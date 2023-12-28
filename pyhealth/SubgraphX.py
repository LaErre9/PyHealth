import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
import networkx as nx
from typing import List, Tuple, Optional, Dict
from functools import partial
from tqdm import trange
import math
import numpy as np

from pyhealth.models import GNN
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain import HeteroExplanation


class MCTSNode(object):
    """A node in a Monte Carlo Tree Search representing a subgraph."""
    def __init__(self,
                 coalition: Tuple[Tuple[int, int], ...],  # Tuple of node pairs (links)
                 subgraph: HeteroData,  # Changed to HeteroData
                 ori_graph: nx.Graph,
                 c_puct: float,
                 W: float = 0,
                 N: int = 0,
                 P: float = 0
                 ) -> None:
        """Initializes the MCTSNode object.

        :param coalition: A tuple of the nodes in the subgraph represented by this MCTSNode.
        :param data: The full graph.
        :param ori_graph: The original graph in NetworkX format.
        :param W: The sum of the node value.
        :param N: The number of times of arrival at this node.
        :param P: The property score (reward) of this node.
        :param c_puct: The hyperparameter that encourages exploration.
        """
        self.coalition = coalition
        self.subgraph = subgraph
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.W = W
        self.N = N
        self.P = P
        self.children: List[MCTSNode] = []

    def Q(self) -> float:
        """Value that encourages exploitation of nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        """Value that encourages exploration of nodes with few visits."""
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def size(self) -> int:
        """Returns the number of nodes in the subgraph represented by the MCTSNode."""
        return len(self.coalition)

def gnn_score(
        coalition: Tuple[Tuple[int, int], ...],
        subgraph: HeteroData,
        x: Dict[str, Tensor],
        model: torch.nn.Module
        ) -> float:
    """
    Computes the GNN score of the subgraph with the selected coalition of nodes.

    :param coalition: A list of indices of the nodes to retain in the induced subgraph.
    :param subgraph: A HeteroData object containing the full heterogeneous graph.
    :param model: The GNN model to use to compute the score.
    :return: The score of the GNN model applied to the subgraph induced by the provided coalition of nodes.
    """
    print(coalition)
    node_mask_type = {}
    x_mask_type = {}
    # Create node masks for each node type
    for node_type in subgraph.node_types:
        node_mask = torch.zeros(subgraph[node_type].num_nodes, dtype=torch.bool)
        x_mask = torch.zeros(subgraph[node_type].num_nodes, dtype=torch.float)
        if node_type in coalition:
            node_mask[list(coalition[node_type])] = 1
            x_mask[list(coalition[node_type])] = x[node_type][list(coalition[node_type])]
            print(x[node_type][list(coalition[node_type])])
        node_mask_type[node_type] = node_mask
        x_mask_type[node_type] = x_mask

    edge_mask_type = {}
    # Create edge masks for each edge type
    for edge_type in subgraph.edge_types:
        row, col = subgraph[edge_type].edge_index
        edge_mask = (node_mask_type[edge_type[0]][row] == 1) & (node_mask_type[edge_type[2]][col] == 1)
        edge_mask_type[edge_type] = edge_mask

    mask_data = HeteroData()

    # Create node masks for each node type
    for node_type in subgraph.node_types:
        mask_data[node_type].node_id = node_mask_type[node_type]

    # Create edge masks for each edge type
    for edge_type in subgraph.edge_types:
        mask_data[edge_type].edge_index = edge_mask_type[edge_type]

    print(mask_data)
    print(node_mask_type)

    # Apply the model to the masked subgraph
    model.eval()
    with torch.no_grad():
        #FORSE TUTTO SBALLATO
        logits = model(x_mask_type, 
                       mask_data.edge_index_dict,
                       mask_data['visit', 'drug'].edge_label_index,
                       mask_data['visit', 'drug'].edge_label,
                       )
        
    print(logits)
    
    score = torch.sigmoid(logits).item()

    return score


def get_best_mcts_node(results: List[MCTSNode], max_nodes: int) -> MCTSNode:
    """Get the MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.

    :param results: A list of MCTSNodes.
    :param max_nodes: The maximum number of nodes allowed in a subgraph represented by an MCTSNode.
    :return: The MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.
    """
    # Filter subgraphs to only include those with at most max_nodes nodes
    results = [result for result in results if result.size <= max_nodes]

    # Check that there exists a subgraph with at most max_nodes nodes
    if len(results) == 0:
        raise ValueError(f'All subgraphs have more than {max_nodes} nodes.')

    # Sort subgraphs by size in case of a tie (max picks the first one it sees, so the smaller one)
    results = sorted(results, key=lambda result: result.size)

    # Find the subgraph with the highest reward and break ties by preferring a smaller graph
    best_result = max(results, key=lambda result: (result.P, -result.size))

    return best_result


class MCTS(object):
    """An object which runs Monte Carlo Tree Search to find optimal subgraphs for link prediction."""

    def __init__(
        self,
        subgraph: HeteroData,
        x: Dict[str, Tensor], 
        edge_index: Dict[str, Tensor], 
        model: torch.nn.Module,
        num_hops: int,
        n_rollout: int,
        min_nodes: int,
        c_puct: float,
        num_expand_nodes: int,
        high2low: bool
        ) -> None:

        """
        Creates the Monte Carlo Tree Search (MCTS) object.
        :param model: The GNN model to explain.
        :param subgraph: The subgraph data.
        :param device: The device to run the computations on.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree.
        """

        self.x = x
        self.edge_index = edge_index
        self.model = model
        self.num_hops = num_hops
        self.subgraph = subgraph
        self.graph = to_networkx(subgraph.to_homogeneous(), to_undirected=True)
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

        self.root_coalition = {}
        for node_type in self.subgraph.node_types:
            self.root_coalition[node_type] = tuple(range(self.subgraph[node_type].num_nodes))
        
        self.MCTSNodeClass = partial(MCTSNode, subgraph=self.subgraph, ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {self.root.coalition[type]: self.root for type in self.root_coalition.keys()}

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        """Performs a Monte Carlo Tree Search rollout for link prediction.

        :param tree_node: An MCTSNode representing the root of the MCTS search.
        :return: The value (reward) of the rollout.
        """
        # Return the node's value if the minimum number of nodes is reached
        for type in tree_node.coalition.keys():
            if len(tree_node.coalition[type]) <= self.min_nodes:
                return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            # Maintain a set of all the coalitions added as children of the tree
            tree_children_coalitions = set()

            # Get subgraph induced by the tree
            tree_subgraph_pyg = self.subgraph.subgraph({"patient": torch.from_numpy(np.array(tree_node.coalition['patient']))})
            comm = tree_subgraph_pyg.to_homogeneous()
            tree_subgraph = to_networkx(comm, to_undirected=True)

            # Get nodes to try expanding
            all_nodes = sorted(
                tree_subgraph.nodes,
                key=lambda node: tree_subgraph.degree[node],
                reverse=self.high2low
            )
            all_nodes_set = set(all_nodes)

            expand_nodes = all_nodes[:self.num_expand_nodes]

            # For each node, prune it and get the remaining subgraph (only keep the largest connected component)
            for expand_node in expand_nodes:
                subgraph_coalition = all_nodes_set - {expand_node}

                subgraphs = (
                    self.graph.subgraph(connected_component)
                    for connected_component in nx.connected_components(self.graph.subgraph(subgraph_coalition))
                )

                subgraph = max(subgraphs, key=lambda subgraph: subgraph.number_of_nodes())

                new_coalition = tuple(sorted(subgraph.nodes()))

                # Check the state map and merge with an existing subgraph if available
                new_node = self.state_map.setdefault(new_coalition, self.MCTSNodeClass(coalition=new_coalition))

                # Add the subgraph to the children of the tree
                if new_coalition not in tree_children_coalitions:
                    tree_node.children.append(new_node)
                    tree_children_coalitions.add(new_coalition)

            # For each child in the tree, compute its reward using the GNN
            for child in tree_node.children:
                if child.P == 0:
                    child.P = gnn_score(coalition=child.coalition, subgraph=child.subgraph, x=self.x, model=self.model)
                    print("Score: " + str(child.P))

        # Select the best child node and unroll it
        sum_count = sum(child.N for child in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        v = self.mcts_rollout(tree_node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def run_mcts(self) -> List[MCTSNode]:
        """Runs the Monte Carlo Tree search.

        :return: A list of MCTSNode objects representing subgraph explanations sorted from highest to
                 smallest reward (for ties, the smaller graph is first).
        """
        for _ in trange(self.n_rollout):
            self.mcts_rollout(tree_node=self.root)

        explanations = [node for _, node in self.state_map.items()]

        # Sort by highest reward and break ties by preferring a smaller graph
        explanations = sorted(explanations, key=lambda x: (x.P, -x.size), reverse=True)

        return explanations


class SubgraphX(ExplainerAlgorithm):
    """An object which contains methods to explain a GNN's prediction on a graph in terms of subgraphs for link prediction."""

    def __init__(self,
                 subgraph: HeteroData,
                 n_rollout: int = 20,
                 max_nodes: int = 10,
                 min_nodes: int = 5,
                 c_puct: float = 10.0,
                 num_expand_nodes: int = 14,
                 high2low: bool = False,
                 num_hops: Optional[int] = None,
                 ) -> None:
        """Initializes the SubgraphX object.

        :param model: The GNN model to explain.
        :param num_hops: The number of hops to extract the neighborhood of target node.
                         If None, uses the number of MessagePassing layers in the model.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree
                         when extending the child nodes in the search tree.
        """
        super().__init__()

        self.subgraph = subgraph
        self.num_hops = num_hops

        # MCTS hyperparameters
        self.n_rollout = n_rollout
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

    def supports(self) -> bool:
        return True

    def forward(self,
                model: torch.nn.Module,
                x: Dict[str, Tensor],
                edge_index: Dict[str, Tensor],
                *,
                target: Tensor,
                **kwargs,) -> HeteroExplanation:
        """Explain the GNN behavior for the graph using the SubgraphX method.

        :param x: Node feature matrix with shape [num_nodes, dim_node_feature].
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges].
        :param max_nodes: The maximum number of nodes in the final explanation results.
        :return: The MCTSNode corresponding to the subgraph that best explains the model's prediction on the graph
                 (the smallest graph that has the highest reward such that the subgraph has at most max_nodes nodes).
        """
        self.model = model

        if self.num_hops is None:
            self.num_hops = sum(isinstance(module, MessagePassing) for module in self.model.modules())

        # Create an MCTS object with the provided graph
        mcts = MCTS(
            x=x,
            edge_index=edge_index,
            subgraph=self.subgraph,
            model=self.model,
            num_hops=self.num_hops,
            n_rollout=self.n_rollout,
            min_nodes=self.min_nodes,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes,
            high2low=self.high2low
        )

        # Run the MCTS search
        mcts_nodes = mcts.run_mcts()

        # Select the MCTSNode that contains the smallest subgraph that has the highest reward
        # such that the subgraph has at most max_nodes nodes
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_nodes=self.max_nodes)
        print(best_mcts_node.coalition)
        explanation = HeteroExplanation(best_mcts_node)

        return explanation