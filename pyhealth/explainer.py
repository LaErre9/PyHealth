import os
import torch
import datetime
import webbrowser
import pandas as pd
from pyvis.network import Network
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional

from torch.nn import functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
from torch_geometric.explain import CaptumExplainer, Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, ModelMode, MaskType, ModelReturnType
from torch_geometric.explain.metric import characterization_score, fidelity_curve_auc

from pyhealth.models import GNN
from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import InnerMap
from pyhealth.GNNExplainer import GNNExplainer
from pyhealth.SubgraphX import SubgraphX

# Save current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
                'edge_label': subgraph['visit', 'drug'].edge_label,
                'mask': mask,
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

        ######### FIDELITY MODIFICATA PER DATASET SBILANCIATO #########
        # alpha = (((y == explain_y_hat) & (y == 0)).sum().item()) / len(y)

        if explainer.explanation_type == ExplanationType.model:
            # pos_fidelity = 1. - ((1 - alpha) * ((y == complement_y_hat) & (y == 0)).sum().item() +
            #                     alpha * ((y == complement_y_hat) & (y == 1)).sum().item()) / len(y)
            # neg_fidelity = 1. - (alpha * ((y == explain_y_hat) & (y == 0)).sum().item() +
            #                     (1 - alpha) * ((y == explain_y_hat) & (y == 1)).sum().item()) / len(y)
            pos_fidelity = 1. - (complement_y_hat == y).float().mean()
            neg_fidelity = 1. - (explain_y_hat == y).float().mean()
        else:
            pos_fidelity = ((y_hat == y).float() -
                            (complement_y_hat == y).float()).abs().mean()
            neg_fidelity = ((y_hat == y).float() -
                            (explain_y_hat == y).float()).abs().mean()

        return float(pos_fidelity), float(neg_fidelity)

# def sparsity(
#         explainer: Explainer,
#         explanation: Explanation,
# ) -> Tuple[float]:

#     # Estrai node_mask e edge_mask da HeteroExplanation
#     node_mask_dict = {k: explanation[k].node_mask for k in explanation.node_types if k != 'edge_mask'}
    
#     sparsity_sum = 0
#     N = len(node_mask_dict)

#     for node_type in node_mask_dict:
#         M_i = (node_mask_dict[node_type] > 0).sum()
#         G_i = len(node_mask_dict[node_type]) 
#         sparsity_sum += (1 - (M_i / G_i))

#     sparsity_score = sparsity_sum / N

#     return float(sparsity_score)
    

def sparsity(
        subgraph: HeteroData,
        explainer: Explainer,
        explanation: Explanation,
) -> Tuple[float]:

    # Estrai node_mask e edge_mask da HeteroExplanation
    node_mask_dict = {k: explanation[k].node_mask for k in explanation.node_types if k != 'edge_mask'}

    M = 0
    for node_type in node_mask_dict:
        M += (node_mask_dict[node_type] > 0).sum()

    num_nodes = subgraph.num_nodes
    sparsity_score = 1 - (M / num_nodes)
    
    return float(sparsity_score)

def unfaithfulness(
        explainer: Explainer,
        explanation: Explanation,
        subgraph: HeteroData,
        node_features: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        top_k: Optional[int] = None,
) -> float:
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

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
            'edge_label': subgraph['visit', 'drug'].edge_label,
            'mask': mask,
        }

    y = subgraph['visit', 'drug'].edge_label

    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    feat_importance = {}
    if node_mask_dict is not None and top_k is not None:
        for key in node_mask_dict.keys():
            feat_importance[key] = node_mask_dict[key].sum(dim=0)
            _, top_k_index = feat_importance[key].topk(top_k)
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

    kl_div = F.kl_div(y.log(), y_hat, reduction='batchmean')
    return 1 - float(torch.exp(-kl_div))

class HeteroGraphExplainer():
    def __init__(
        self,
        algorithm: str,
        dataset: SampleEHRDataset,
        model: GNN,
        label_key: str,
        threshold_value: float,
        top_k: int,
        root: str="./explainability_results/",
    ):
        self.dataset = dataset
        self.model = model
        self.label_key = label_key
        self.algorithm = algorithm
        self.threshold_value = threshold_value
        self.top_k = top_k
        self.root = root

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_tokenizer = self.model.get_label_tokenizer()
        
        self.proc_df, self.symp_df, self.drug_df, self.diag_df = self.get_dataframe()

        # dizionari dal modello
        self.hadm_dict = self.model.hadm_dict
        self.subject_dict = self.model.subject_dict
        self.icd9_symp_dict = self.model.icd9_symp_dict
        self.icd9_diag_dict = self.model.icd9_diag_dict
        self.icd9_proc_dict = self.model.icd9_proc_dict
        self.atc_pre_dict = self.model.atc_pre_dict

        # Get the subgraph
        self.subgraph = self.get_subgraph()
        self.subgraph = self.generate_neg_samples()
        self.mask = self.generate_mask()

        self.subgraph = self.subgraph.to(device=self.device)

        self.node_features = self.model.x_dict(self.subgraph)

        # Explainer
        if self.algorithm == "IG":
            explainer_algorithm = CaptumExplainer('IntegratedGradients',
                                                  n_steps=25,
                                                  method='riemann_trapezoid'
                                                  )
            type_returned = "probs"
        elif self.algorithm == "GNNExplainer":
            explainer_algorithm = GNNExplainer(epochs=300,
                                               lr=0.3,
                                               )
            type_returned = "raw"
        elif self.algorithm == "SubgraphX":
            explainer_algorithm = SubgraphX(subgraph=self.subgraph,
                                            max_nodes=10,
                                            min_nodes=10,
                                            )
            type_returned = "probs"
        else:
            raise ValueError("Explainer algorithms not yet supported")

        self.explainer = Explainer(
            model=self.model.layer,
            # HYPERPARAMETERS
            algorithm=explainer_algorithm,
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='edge',
                return_type=type_returned,
            ),
            node_mask_type='attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type='topk',
                value=self.threshold_value,
            )
        )

    def get_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Union[None, Dict[str, pd.DataFrame]]]:
        """Gets the dataframe of conditions, procedures, symptoms and drugs of patients.

        Returns:
            dataframe: a `pandas.DataFrame` object.
        """
        PROCEDURES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        SYMPTOMS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        DRUGS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ATC_CODE'])
        DIAGNOSES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])


        # Loop over all patients
        for patient_data in self.dataset:
            subject_id = patient_data['patient_id']
            hadm_id = patient_data['visit_id']

            # PROCEDURES DataFrame
            procedures_data = patient_data['procedures'][-1]
            procedures_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(procedures_data),
                                        'HADM_ID': [hadm_id] * len(procedures_data),
                                        'SEQ_NUM': range(1, len(procedures_data) + 1),
                                        'ICD9_CODE': procedures_data})
            PROCEDURES = pd.concat([PROCEDURES, procedures_df], ignore_index=True)

            # SYMPTOMS DataFrame
            symptoms_data = patient_data['symptoms'][-1]
            symptoms_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(symptoms_data),
                                        'HADM_ID': [hadm_id] * len(symptoms_data),
                                        'SEQ_NUM': range(1, len(symptoms_data) + 1),
                                        'ICD9_CODE': symptoms_data})
            SYMPTOMS = pd.concat([SYMPTOMS, symptoms_df], ignore_index=True)

            if self.label_key == "drugs":
                drugs_data = patient_data['drugs']
                diagnoses_data = patient_data['conditions'][-1]
            else:
                drugs_data = patient_data['drugs'][-1]
                diagnoses_data = patient_data['conditions']

            # DRUGS DataFrame
            drugs_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(drugs_data),
                                    'HADM_ID': [hadm_id] * len(drugs_data),
                                    'ATC_CODE': drugs_data})
            DRUGS = pd.concat([DRUGS, drugs_df], ignore_index=True)

            # DIAGNOSES DataFrame
            diagnoses_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(diagnoses_data),
                                        'HADM_ID': [hadm_id] * len(diagnoses_data),
                                        'SEQ_NUM': range(1, len(diagnoses_data) + 1),
                                        'ICD9_CODE': diagnoses_data})
            DIAGNOSES = pd.concat([DIAGNOSES, diagnoses_df], ignore_index=True)

        return PROCEDURES, SYMPTOMS, DRUGS, DIAGNOSES

    def get_subgraph(self) -> HeteroData:
        """
        Returns a subgraph containing selected patients, visits, symptoms, procedures and diseases.

        Returns:
            subgraph (HeteroData): A subgraph containing selected patients and visits.
        """
        # ==== DATA SELECTION ====
        # Select the patients, visits, symptoms, procedures and diseases from the graph
        self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(self.subject_dict)
        patient = self.symp_df["SUBJECT_ID"].unique()
        select_patient = torch.from_numpy(patient)
 
        self.symp_df['HADM_ID'] = self.symp_df['HADM_ID'].map(self.hadm_dict)
        visit = self.symp_df["HADM_ID"].unique()
        select_visit = torch.from_numpy(visit)
 
        self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)
        symptom = self.symp_df["ICD9_CODE"].unique()
        select_symptom = torch.from_numpy(symptom)
 
        self.diag_df['ICD9_CODE'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
        disease = self.diag_df["ICD9_CODE"].unique()
        select_disease = torch.from_numpy(disease)
 
        self.proc_df['ICD9_CODE'] = self.proc_df['ICD9_CODE'].map(self.icd9_proc_dict)
        procedure = self.proc_df["ICD9_CODE"].unique()
        select_procedure = torch.from_numpy(procedure)
 
        subgraph = self.model.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "disease": select_disease})

        return subgraph

    def generate_neg_samples(self) -> HeteroData:
        """
        Generates negative samples for the subgraph based on the label key.

        Returns:
            HeteroData: The updated subgraph with negative samples.
        """
        if self.label_key == "drugs":
            neg_edges = negative_sampling(self.subgraph['visit', 'has_received', 'drug'].edge_index, num_nodes=(self.subgraph['visit'].num_nodes, self.subgraph['drug'].num_nodes))
            self.subgraph['visit', 'has_received', 'drug'].edge_label_index = self.subgraph['visit', 'has_received', 'drug'].edge_index
            self.subgraph['visit', 'has_received', 'drug'].edge_label = torch.ones(self.subgraph['visit', 'has_received', 'drug'].edge_label_index.shape[1], dtype=torch.float)
            self.subgraph['visit', 'has_received', 'drug'].edge_label_index = torch.cat((self.subgraph['visit', 'has_received', 'drug'].edge_label_index, neg_edges), dim=1)
            self.subgraph['visit', 'has_received', 'drug'].edge_label = torch.cat((self.subgraph['visit', 'has_received', 'drug'].edge_label, torch.zeros(neg_edges.shape[1], dtype=torch.float)), dim=0)
        else:
            neg_edges = negative_sampling(self.subgraph['visit', 'has', 'disease'].edge_index, num_nodes=(self.subgraph['visit'].num_nodes, self.subgraph['drug'].num_nodes))
            self.subgraph['visit', 'has', 'disease'].edge_label_index = self.subgraph['visit', 'has', 'disease'].edge_index
            self.subgraph['visit', 'has', 'disease'].edge_label = torch.ones(self.subgraph['visit', 'has', 'disease'].edge_label_index.shape[1], dtype=torch.float)
            self.subgraph['visit', 'has', 'disease'].edge_label_index = torch.cat((self.subgraph['visit', 'has', 'disease'].edge_label_index, neg_edges), dim=1)
            self.subgraph['visit', 'has', 'disease'].edge_label = torch.cat((self.subgraph['visit', 'has', 'disease'].edge_label, torch.zeros(neg_edges.shape[1], dtype=torch.float)), dim=0)

        return self.subgraph

    def generate_mask(self) -> torch.Tensor:
        """
        Generates a mask for the subgraph edges based on the label key.

        Returns:
            torch.Tensor: The generated mask.
        """
        if self.label_key == "drugs":
            mask = torch.ones_like(self.subgraph['visit', 'has_received', 'drug'].edge_label, dtype=torch.bool, device=self.device)

            # Get all possible edges in the graph
            all_possible_edges = torch.cartesian_prod(torch.arange(self.subgraph['visit'].num_nodes), torch.arange(self.label_tokenizer.get_vocabulary_size()))

            # Filter existing edges in the current graph
            existing_edges = self.subgraph['visit', 'has_received', 'drug'].edge_label_index.t().contiguous()

            # Find missing edges in the current graph
            missing_edges = torch.tensor(list(set(map(tuple, all_possible_edges.tolist())) - set(map(tuple, existing_edges.tolist())))).t().contiguous()

            self.subgraph['visit', 'has_received', 'drug'].edge_label_index = torch.cat([self.subgraph['visit', 'has_received', 'drug'].edge_label_index, missing_edges], dim=1)
            self.subgraph['visit', 'has_received', 'drug'].edge_label = torch.cat([self.subgraph['visit', 'has_received', 'drug'].edge_label, torch.zeros(missing_edges.size(1), dtype=torch.float)], dim=0)
        else:
            mask = torch.ones_like(self.subgraph['visit', 'has', 'disease'].edge_label, dtype=torch.bool, device=self.device)

            # Get all possible edges in the graph
            all_possible_edges = torch.cartesian_prod(torch.arange(self.subgraph['visit'].num_nodes), torch.arange(self.label_tokenizer.get_vocabulary_size()))

            # Filter existing edges in the current graph
            existing_edges = self.subgraph['visit', 'has', 'disease'].edge_label_index.t().contiguous()

            # Find missing edges in the current graph
            missing_edges = torch.tensor(list(set(map(tuple, all_possible_edges.tolist())) - set(map(tuple, existing_edges.tolist())))).t().contiguous()

            self.subgraph['visit', 'has', 'disease'].edge_label_index = torch.cat([self.subgraph['visit', 'has', 'disease'].edge_label_index, missing_edges], dim=1)
            self.subgraph['visit', 'has', 'disease'].edge_label = torch.cat([self.subgraph['visit', 'has', 'disease'].edge_label, torch.zeros(missing_edges.size(1), dtype=torch.float)], dim=0)

        # Extend the mask with False for missing edges
        mask = torch.cat([mask, torch.zeros(missing_edges.size(1), dtype=torch.bool, device=self.device)], dim=0)

        return mask
        
    def explain(
        self,
        n: int
    ):

        self.explanation =self.explainer(
            x = self.node_features,
            edge_index = self.subgraph.edge_index_dict,
            edge_label_index = self.subgraph['visit', 'drug'].edge_label_index[:, n],
            edge_label = self.subgraph['visit', 'drug'].edge_label[n],
            mask = self.mask
        )
        print(f'Generated explanations in {self.explanation.available_explanations}')

        path = f'{self.root}feature_importance_{current_datetime}.png'

        self.explanation.detach()
        self.explanation.visualize_feature_importance(path, top_k=self.top_k)
        self.explanation['prediction'] = F.sigmoid(self.explanation['prediction'])

        print(f"Feature importance plot has been saved to '{path}'")
        print('Edge to predict: ' + str(self.subgraph['visit', 'drug'].edge_label_index[:, n]))
        print('Label to predict: ' + str(self.subgraph['visit', 'drug'].edge_label[n].numpy().astype(int)))
        print('Label predicted: ' + str(1 if self.explanation['prediction'].numpy() > 0.5 else 0) + " because " + str(self.explanation['prediction'].numpy()))

        return

    def explain_graph(
        self,
        k: int = 0,
    ):

        # Creazione del grafo NetworkX
        self.G = nx.DiGraph()

        # Definizione dei colori per i diversi tipi di nodi
        if k >= 2:
            node_colors = {
                'patient': '#add8e6',  # light blue
                'visit': '#ff9999',    # soft red
                'symptom': '#98fb98',  # pale green
                'procedure': '#ffa07a', # light salmon
                'disease': '#f08080',  # light coral
                'drug': '#ffffe0',     # light yellow
                'anatomy': '#f05555',  # light coral
                'pharmaclass': '#ff5fe5',     # light yellow
            }
        else:
            node_colors = {
                'patient': '#add8e6',  # light blue
                'visit': '#ff9999',    # soft red
                'symptom': '#98fb98',  # pale green
                'procedure': '#ffa07a', # light salmon
                'disease': '#f08080',  # light coral
                'drug': '#ffffe0'      # light yellow
            }

        # Assumi che `self.explanation` sia un oggetto definito precedentemente con i dati necessari
        if k >= 2:
            entities = [('patient', self.explanation['patient']),
                        ('visit', self.explanation['visit']),
                        ('symptom', self.explanation['symptom']),
                        ('procedure', self.explanation['procedure']),
                        ('disease', self.explanation['disease']),
                        ('drug', self.explanation['drug']),
                        ('anatomy', self.explanation['anatomy']),
                        ('pharmaclass', self.explanation['pharmaclass'])
                        ]
        else:
            entities = [('patient', self.explanation['patient']),
                        ('visit', self.explanation['visit']),
                        ('symptom', self.explanation['symptom']),
                        ('procedure', self.explanation['procedure']),
                        ('disease', self.explanation['disease']),
                        ('drug', self.explanation['drug'])
                        ]

        # MI PRENDO I NODI PIU' IMPORTANTI
        nodess = []
        for node_type, node_data in self.explanation.node_items():
            for i in range(node_data['node_mask'].shape[0]):
                node_id = f"{node_type}_{i}"
                node_mask = node_data['node_mask'][i]
                if torch.any(node_mask):
                    if node_id not in nodess:
                        nodess.append(node_id)
                    #print(f"{node_id}")
                    #self.G.add_node(node_id, type=node_type)

        for edge_type, edge_data in [(edge[0], edge[1]) for edge in self.explanation.edge_items()]:
            for i in range(edge_data['edge_index'].shape[1]):
                source_id = f"{edge_type[0]}_{edge_data['edge_index'][0, i]}"
                target_id = f"{edge_type[2]}_{edge_data['edge_index'][1, i]}"

                edge_mask = self.explanation[edge_type]['edge_mask'][i]

                if edge_mask > 0:
                    if source_id in nodess and target_id in nodess:
                        # print(source_id + " -> " + target_id)
                        # print(edge_type)
                        # print()
                        self.G.add_edge(source_id, target_id)

        for entity_type, entity_data in entities:
            for i in range(entity_data['x'].shape[0]):
                node_id = f"{entity_type}_{i}"
                if node_id in nodess:
                    node_mask = entity_data['node_mask'][i]

                    # Assicurati che node_mask sia un singolo valore scalare
                    if isinstance(node_mask, torch.Tensor) and node_mask.numel() == 1:
                        node_mask_value = node_mask.item()
                    else:
                        # Se node_mask contiene più valori, scegli un approccio appropriato qui
                        # Ad esempio, potresti voler utilizzare la media o il massimo dei valori
                        node_mask_value = node_mask.mean().item()  # o node_mask.max().item()

                    # Calcola la dimensione del nodo
                    node_size = max(10, node_mask_value * 10)
                    self.G.add_node(node_id, type=entity_type, size=node_size)


        # Converti il grafo NetworkX in un grafo Pyvis
        net = Network(notebook=True, height="750px", width="100%", cdn_resources="remote")
        net.from_nx(self.G)

        # Assegna i colori e le dimensioni ai nodi in Pyvis
        for node in net.nodes:
            node['color'] = node_colors[node['type']]
            if 'size' in self.G.nodes[node['id']]:
                node['size'] = self.G.nodes[node['id']]['size']

        # Crea una legenda HTML e aggiungila al grafo Pyvis
        legend_html = "<div style='position:absolute; right:20px; top:20px; width:200px; background-color:rgba(255,255,255,0.8); padding:10px; border-radius:5px; font-size:14px;'>"
        legend_html += "<strong>Node Types</strong><br>"
        for node_type, color in node_colors.items():
            legend_html += f"<span style='margin-left:10px; color:{color};'>{node_type}</span><br>"
        legend_html += "</div>"
        net.html = legend_html

        # Visualizza il grafo in un file HTML
        filepath = f'{self.root}explain_graph{current_datetime}.html'
        net.show(filepath)
        webbrowser.open(f'{os.getcwd()}/{filepath}')

        return

    def explain_metrics(
        self,
        metrics: List[str] = 'Fidelity',
    ):

        explainer = self.explainer
        explanation = self.explanation

        for metric in metrics:
            if metric == "Fidelity":
                pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, self.mask)
                print("Fidelity Positive: " + str(float(pos_fidelity)))
                print("Fidelity Negative: " + str(float(neg_fidelity)))

            elif metric == "Fidelity_AUC":
                fid_auc = fidelity_curve_auc(torch.tensor([pos_fidelity]), torch.tensor([neg_fidelity]), x=torch.tensor([0]))
                print("Fidelity AUC: " + str(float(fid_auc)))

            elif metric == "Characterization_Score":
                pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, self.mask)
                score = characterization_score(float(pos_fidelity), float(neg_fidelity))
                print("Characterization Score: " + str(score))

            elif metric == "Unfaithfulness":
                unfaithfulness_score = unfaithfulness(explainer, explanation, self.subgraph, self.node_features, self.mask, top_k=10)
                print("Unfaithfulness Score: " + str(unfaithfulness_score))

            elif metric == "Sparsity":
                sparsity_score = sparsity(self.subgraph, explainer, explanation)
                print("Sparsity Score: " + str(sparsity_score))

        return

    def explain_results(self, n: int):

        icd = InnerMap.load("ICD9CM")
        icdpr = InnerMap.load("ICD9PROC")
        atc = InnerMap.load("ATC")

        visit_id = self.subgraph['visit', 'drug'].edge_label_index[:, n][0].item()
        drug_id = self.subgraph['visit', 'drug'].edge_label_index[:, n][1].item()

        # Print prescription decision
        if self.explanation['prediction'].numpy() > 0.5:
            print(f'In visit {visit_id}, the patient received the drug {atc.lookup(list(self.atc_pre_dict.keys())[int(drug_id)])}.')
        else:
            print(f'In visit {visit_id}, the patient did not receive the drug {atc.lookup(list(self.atc_pre_dict.keys())[int(drug_id)])}.')

        print()
        print('Medical Visit Details:')
        print()

        symptoms = []
        procedures = []
        diseases = []
        drugs = []


        # MI PRENDO I NODI PIU' IMPORTANTI
        nodess = []
        for node_type, node_data in self.explanation.node_items():
            for i in range(node_data['node_mask'].shape[0]):
                node_id = f"{node_type}_{i}"
                node_mask = node_data['node_mask'][i]
                if torch.any(node_mask):
                    if node_id not in nodess:
                        nodess.append(node_id)
                    #print(f"{node_id}")
                    #self.G.add_node(node_id, type=node_type)

        for edge_type, edge_data in [(edge[0], edge[1]) for edge in self.explanation.edge_items()]:
            for i in range(edge_data['edge_index'].shape[1]):
                source_id = f"{edge_type[0]}_{edge_data['edge_index'][0, i]}"
                target_id = f"{edge_type[2]}_{edge_data['edge_index'][1, i]}"

                edge_mask = self.explanation[edge_type]['edge_mask'][i]

                if edge_mask > 0:
                    if source_id in nodess and target_id in nodess:
                        source, src_id = str(source_id).split('_')
                        target, tgt_id = str(target_id).split('_')

                        if source == "visit" and src_id == str(visit_id):
                            if target == "symptom":
                                symptoms.append(target_id)
                            elif target == "procedure":
                                procedures.append(target_id)
                            elif target == "disease":
                                diseases.append(target_id)
                            elif target == "drug":
                                drugs.append(target_id)

                        elif target == "visit" and tgt_id == str(visit_id):
                            if source == "symptom":
                                symptoms.append(source_id)
                            elif source == "procedure":
                                procedures.append(source_id)
                            elif source == "disease":
                                diseases.append(source_id)
                            elif source == "drug":
                                drugs.append(source_id)


        # List of symptoms presented by the patient
        print('Symptoms presented by the patient:')
        for symptom in symptoms:
            node_type, id = str(symptom).split('_')
            if node_type == "symptom":
                symptom = icd.lookup(list(self.icd9_symp_dict.keys())[int(id)])
                print(symptom + " " + id)

        print()

        # List of procedures performed on the patient
        print('Procedures performed on the patient:')
        for procedure in procedures:
            node_type, id = str(procedure).split('_')
            if node_type == "procedure":
                procedure = icdpr.lookup(list(self.icd9_proc_dict.keys())[int(id)])
                print(procedure + " " + id)

        print()

        # List of patient's diagnoses
        print('Patient diagnoses:')
        for disease in diseases:
            node_type, id = str(disease).split('_')
            if node_type == "disease":
                disease = icd.lookup(list(self.icd9_diag_dict.keys())[int(id)])
                print(disease + " " + id)

        print()

        # List of drugs administered to the patient
        print('Drugs administered to the patient:')
        for drug in drugs:
            node_type, id = str(drug).split('_')
            if node_type == "drug":
                drug = atc.lookup(list(self.atc_pre_dict.keys())[int(id)])
                print(drug + " " + id)

        return