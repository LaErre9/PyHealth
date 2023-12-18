from math import sqrt
from typing import Optional, Union, Dict

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_hetero_masks
from torch_geometric.explain.config import MaskType, ModelMode


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    # COEFFICIENTI DI DEFAULT
    # coeffs = {
    #     'edge_size': 0.005,
    #     'edge_reduction': 'sum',
    #     'node_feat_size': 1.0,
    #     'node_feat_reduction': 'mean',
    #     'edge_ent': 1.0,
    #     'node_feat_ent': 0.1,
    #     'EPS': 1e-15,
    # }

    coeffs = {
        'edge_size': 0.001,
        'edge_reduction': 'sum',
        'node_feat_size': 0.9,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[str, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:

        is_hetero = isinstance(x, dict)
        self._train(model, x, edge_index, is_hetero, target=target, index=index, **kwargs)
 
        node_mask = self._extract_mask(self.node_mask, 
                                       self.hard_node_mask, 
                                       is_hetero)
        edge_mask = self._extract_mask(self.edge_mask, 
                                       self.hard_edge_mask, 
                                       is_hetero)

        self._clean_model(model)

        if is_hetero:
            explanation = HeteroExplanation()
            explanation.set_value_dict('node_mask', node_mask)
            explanation.set_value_dict('edge_mask', edge_mask)
        else:
            explanation = Explanation(node_mask=node_mask, edge_mask=edge_mask)

        return explanation

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[str, Tensor]],
        is_hetero: bool,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):

        self._initialize_masks(x, edge_index, is_hetero)

        # Inizializzazione delle maschere rigide per i grafici eterogenei
        if is_hetero:
            self.hard_node_mask = {key: None for key in self.node_mask}
            self.hard_edge_mask = {key: None for key in self.edge_mask}

        parameters = []
        if self.node_mask is not None:
            parameters.extend(self.node_mask.values() if is_hetero else [self.node_mask])
        if self.edge_mask is not None:
            set_hetero_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.extend(self.edge_mask.values() if is_hetero else [self.edge_mask])

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            if is_hetero:
                h = {key: val * self.node_mask[key].sigmoid() for key, val in x.items()}
                y_hat, y = model(h, edge_index, **kwargs), target
            else:
                h = x * self.node_mask.sigmoid() if self.node_mask is not None else x
                y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward(retain_graph=True)
            optimizer.step()

            if i == 0:
                self._update_hard_masks(is_hetero)

    def _initialize_masks(self, x, edge_index, is_hetero: bool):

        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if is_hetero:
            self.node_mask = {key: self._init_node_mask(val, node_mask_type) for key, val in x.items()}
            self.edge_mask = {key: self._init_edge_mask(edge_index[key], edge_mask_type) for key in edge_index}
        else:
            self.node_mask = self._init_node_mask(x, node_mask_type)
            self.edge_mask = self._init_edge_mask(edge_index, edge_mask_type)

    def _init_node_mask(self, x: Tensor, node_mask_type: MaskType) -> Optional[Parameter]:

        device = x.device
        (N, F) = x.size()

        self.N = N # Numero di nodi

        std = 0.1
        if node_mask_type is None:
            return None
        elif node_mask_type == MaskType.object:
            return Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            return Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            return Parameter(torch.randn(1, F, device=device) * std)
        else:
            raise ValueError("Invalid node mask type")

    def _init_edge_mask(self, edge_index: Tensor, edge_mask_type: MaskType) -> Optional[Parameter]:

        device = edge_index.device
        E = edge_index.size(1)

        std = 0.1
        if edge_mask_type is None:
            return None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * self.N))
            return Parameter(torch.randn(E, device=device) * std)
        else:
            raise ValueError("Invalid edge mask type")

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:

        # Calcolo della perdita base (ad esempio, per la classificazione binaria)
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            raise NotImplementedError("Unsupported model mode")

        # Aggiunta della penalità della maschera per i grafici eterogenei
        if isinstance(self.edge_mask, dict):
            for key in self.edge_mask:
                if self.hard_edge_mask[key] is not None:
                    m = self.edge_mask[key][self.hard_edge_mask[key]].sigmoid()
                    edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
                    loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
                    ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                    loss = loss + self.coeffs['edge_ent'] * ent.mean()

            # Analogo per le maschere dei nodi
            for key in self.node_mask:
                if self.hard_node_mask[key] is not None:
                    m = self.node_mask[key][self.hard_node_mask[key]].sigmoid()
                    node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
                    loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
                    ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                    loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        else:
            # Gestione delle maschere per grafici omogenei
            if self.hard_edge_mask is not None:
                m = self.edge_mask[self.hard_edge_mask].sigmoid()
                edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
                loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
                ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                loss = loss + self.coeffs['edge_ent'] * ent.mean()

            if self.hard_node_mask is not None:
                m = self.node_mask[self.hard_node_mask].sigmoid()
                node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
                loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
                ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        print('Loss: ' + str(loss))

        return loss

    def _extract_mask(self, mask, hard_mask, is_hetero: bool):

        if is_hetero:
            return {key: self._post_process_mask(mask[key], hard_mask[key], apply_sigmoid=True) for key in mask}
        else:
            return self._post_process_mask(mask, hard_mask, apply_sigmoid=True)

    def _update_hard_masks(self, is_hetero: bool):

        if is_hetero:
            for key in self.node_mask:
                if self.node_mask[key].grad is None:
                        raise ValueError("Could not compute gradients for node "
                                        "features. Please make sure that node "
                                        "features are used inside the model or "
                                        "disable it via `node_mask_type=None`.")
                self.hard_node_mask[key] = self.node_mask[key].grad != 0.0

            for key in self.edge_mask:
                if self.edge_mask[key].grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask[key] = self.edge_mask[key].grad != 0.0
        else:
            if self.node_mask.grad is None:
                raise ValueError("Could not compute gradients for node "
                                    "features. Please make sure that node "
                                    "features are used inside the model or "
                                    "disable it via `node_mask_type=None`.")
            self.hard_node_mask = self.node_mask.grad != 0.0
            if self.edge_mask.grad is None:
                raise ValueError("Could not compute gradients for edges. "
                                    "Please make sure that edges are used "
                                    "via message passing inside the model or "
                                    "disable it via `edge_mask_type=None`.")
            self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _clean_model(self, model):

        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None