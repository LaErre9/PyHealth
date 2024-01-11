import random
import torch

from torch.nn import functional as F
from typing import Dict, Optional, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, ModelMode, MaskType, ModelReturnType


def stability(
        explainer: Explainer,
        explanation: Explanation,
        subgraph: HeteroData,
        mask: torch.Tensor,
        n: int,
        node_features: Dict[str, torch.Tensor],
) -> float:

    # Verifica se il modello è di tipo regression
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    GES = []
    num_run = 3

    for run in range(num_run):

        with torch.no_grad():
            # Clona i tensori e applica il rumore
            node_features_pert = {k: v.clone().detach() for k, v in node_features.items()}
            noise_visit = torch.normal(0, 0.01, node_features_pert['visit'][subgraph['visit', 'drug'].edge_label_index[:, n][0]].shape)
            noise_drug = torch.normal(0, 0.01, node_features_pert['drug'][subgraph['visit', 'drug'].edge_label_index[:, n][1]].shape)

            node_features_pert['visit'][subgraph['visit', 'drug'].edge_label_index[:, n][0]] += noise_visit
            node_features_pert['drug'][subgraph['visit', 'drug'].edge_label_index[:, n][1]] += noise_drug

        # Rewire edges
        num_nodes_visit = subgraph['visit'].num_nodes - 1
        num_nodes_drug = subgraph['drug'].num_nodes - 1
        num_rewire = 10

        # Inizializza l'insieme dei bordi esistenti
        existing_edges = list([tuple(edge) for edge in subgraph['visit', 'has_received', 'drug'].edge_index.t().tolist()])

        # Genera nuovi bordi
        new_edges = set()
        while len(new_edges) < num_rewire:
            new_edge = (random.randint(0, num_nodes_visit), random.randint(0, num_nodes_drug))
            if new_edge not in existing_edges and new_edge[0] != new_edge[1]:  # Evita self-loops e duplicati
                new_edges.add(new_edge)

        # Converti gli edges rimanenti e i nuovi edges in un edge_index
        # Seleziona casualmente gli edges da rimuovere dall'insieme degli edges esistenti
        edges_to_remove = random.sample(list(enumerate(existing_edges)), num_rewire)
        for index, edge in edges_to_remove:
            existing_edges[index] = new_edges.pop()

        perturbed_edge_index = torch.tensor(existing_edges).t()

        # Copia il dizionario degli indici degli archi esistente
        perturbed_edge_index_dict = subgraph.edge_index_dict.copy()

        # Aggiorna l'indice degli archi per il tipo di relazione specifico con il nuovo indice degli archi
        perturbed_edge_index_dict[('visit', 'has_received', 'drug')] = perturbed_edge_index

        #print(perturbed_edge_index.edge_index_dict)

        pert_explanation = explainer(
            x = node_features_pert,
            edge_index = perturbed_edge_index_dict,
            edge_label_index = subgraph['visit', 'drug'].edge_label_index[:, n],
        )

        ori_exp_mask = torch.zeros_like(explanation['visit', 'drug'].edge_mask)
        ori_exp_mask[explanation['visit', 'drug'].edge_mask > 0] = 1

        pert_exp_mask = torch.zeros_like(pert_explanation['visit', 'drug'].edge_mask)
        pert_exp_mask[pert_explanation['visit', 'drug'].edge_mask > 0] = 1

        GES.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())

    # print(f"GES ({num_run} executions):", GES)

    return max(GES)


def fidelity(
        explainer: Explainer,
        explanation: Explanation,
        subgraph: HeteroData,
        node_features: Dict[str, torch.Tensor],
        mask: torch.Tensor,
) -> Tuple[float, float]:

    # Verifica se il modello è di tipo regression
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    # Estrai node_mask e edge_mask da HeteroExplanation
    node_mask_dict = {k: explanation[k].node_mask for k in explanation.node_types if k != 'edge_mask'}
    edge_mask_dict = {k: explanation[k].edge_mask for k in explanation.edge_types}

    kwargs = {}
    if isinstance(explanation, Explanation):
        kwargs = {key: explanation[key] for key in explanation._model_args}
    else:
        kwargs = {
            'edge_label_index': subgraph['visit', 'drug'].edge_label_index,
        }

    y = subgraph['visit', 'drug'].edge_label
    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat = explainer.get_prediction(
            node_features,
            subgraph.edge_index_dict,
            **kwargs,
        )
        y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        node_features,
        subgraph.edge_index_dict,
        node_mask_dict,
        edge_mask_dict,
        **kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    for key in node_mask_dict.keys():
        node_mask_dict[key] = 1. - node_mask_dict[key]

    for key in edge_mask_dict.keys():
        edge_mask_dict[key] = 1. - edge_mask_dict[key]

    complement_y_hat = explainer.get_masked_prediction(
        node_features,
        subgraph.edge_index_dict,
        node_mask_dict,
        edge_mask_dict,
        **kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1. - (complement_y_hat == y).float().mean()
        neg_fidelity = 1. - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = ((y_hat == y).float() -
                        (complement_y_hat == y).float()).abs().mean()
        neg_fidelity = ((y_hat == y).float() -
                        (explain_y_hat == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)


def sparsity(
        explainer: Explainer,
        explanation: Explanation,
) -> Tuple[float]:

    # Verifica se il modello è di tipo regression
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    # Estrai node_mask e edge_mask da HeteroExplanation
    node_mask_dict = {k: explanation[k].node_mask for k in explanation.node_types if k != 'edge_mask'}

    sparsity_sum = 0
    N = len(node_mask_dict)

    for node_type in node_mask_dict:
        M_i = (node_mask_dict[node_type] > 0).sum()
        G_i = len(node_mask_dict[node_type]) 
        sparsity_sum += (1 - (M_i / G_i))

    sparsity_score = sparsity_sum / N

    return float(sparsity_score)


def unfaithfulness(
        explainer: Explainer,
        explanation: Explanation,
        subgraph: HeteroData,
        node_features: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        top_k: Optional[int] = None,
) -> float:

    # Verifica se il modello è di tipo regression
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    # Verifica se è stato specificato un top-k
    if top_k is not None and explainer.node_mask_type == MaskType.object:
        raise ValueError("Cannot apply top-k feature selection based on a "
                         "node mask of type 'object'")

    # Estrai node_mask e edge_mask da HeteroExplanation
    node_mask_dict = {k: explanation[k].node_mask for k in explanation.node_types if k != 'edge_mask'}
    edge_mask_dict = {k: explanation[k].edge_mask for k in explanation.edge_types}
    x, edge_index = node_features, subgraph.edge_index_dict
    
    kwargs = {}
    if isinstance(explanation, Explanation):
        kwargs = {key: explanation[key] for key in explanation._model_args}
    else:
        kwargs = {
            'edge_label_index': subgraph['visit', 'drug'].edge_label_index,
        }

    y = subgraph['visit', 'drug'].edge_label

    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    feat_importance = {}
    if node_mask_dict is not None and top_k is not None:
        for key in node_mask_dict.keys():
            feat_importance[key] = node_mask_dict[key].sum(dim=0)
            _, top_k_index = feat_importance[key].topk(int(0.25*top_k)) # 2208.09339.pdf (arxiv.org)
            node_mask_dict[key] = torch.zeros_like(node_mask_dict[key])
            node_mask_dict[key][:, top_k_index] = 1.0

    y_hat = explainer.get_masked_prediction(x, edge_index, node_mask_dict,
                                            edge_mask_dict, **kwargs)

    if explanation.get('index') is not None:
        y, y_hat = y[explanation.index], y_hat[explanation.index]

    if explainer.model_config.return_type == ModelReturnType.raw or \
        explainer.model_config.return_type == ModelReturnType.probs:
        y, y_hat = y.softmax(dim=-1), y_hat.softmax(dim=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = y.exp(), y_hat.exp()

    kl_div = F.kl_div(y.log(), y_hat, reduction='sum')

    return 1 - float(torch.exp(-kl_div))