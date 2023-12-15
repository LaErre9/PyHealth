import os
import torch
import datetime
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network
from pylab import rcParams
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

        y = explanation.target
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
        # print('explain_y_hat: ' + str(explain_y_hat))
        explain_y_hat = explainer.get_target(explain_y_hat)

        # print('explain_y_hat: ' + str(explain_y_hat.shape))

        # print('Before node mask: ' + str(node_mask_dict))
        # print('Before edge mask: ' + str(edge_mask_dict))

        for key in node_mask_dict.keys():
            node_mask_dict[key] = 1. - node_mask_dict[key]

        for key in edge_mask_dict.keys():
            edge_mask_dict[key] = 1. - edge_mask_dict[key]

        # print('After node mask: ' + str(node_mask_dict))
        # print('After edge mask: ' + str(edge_mask_dict))

        complement_y_hat = explainer.get_masked_prediction(
            node_features,
            subgraph.edge_index_dict,
            node_mask_dict,
            edge_mask_dict,
            **kwargs,
        )
        # print('complement_y_hat: ' + str(complement_y_hat))
        complement_y_hat = explainer.get_target(complement_y_hat)

        # print('complement_y_hat: ' + str(complement_y_hat.shape))
        
        y = subgraph['visit', 'drug'].edge_label

        if explanation.get('index') is not None:
            if explainer.explanation_type == ExplanationType.phenomenon:
                y_hat = y_hat[explanation.index]
            explain_y_hat = explain_y_hat[explanation.index]
            complement_y_hat = complement_y_hat[explanation.index]

        # print('complement_y_hat: ' + str(complement_y_hat))
        # print('explain_y_hat: ' + str(explain_y_hat))
        # print('y: ' + str(y))

        if explainer.explanation_type == ExplanationType.model:
            pos_fidelity = 1. - (complement_y_hat == y).float().mean()
            neg_fidelity = 1. - (explain_y_hat == y).float().mean()
        else:
            pos_fidelity = ((y_hat == y).float() -
                            (complement_y_hat == y).float()).abs().mean()
            neg_fidelity = ((y_hat == y).float() -
                            (explain_y_hat == y).float()).abs().mean()

        return pos_fidelity, neg_fidelity

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
    # print('y: ' + str(y))
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
    y_hat = y_hat.softmax(dim=-1)
    y = y.softmax(dim=-1)
    # print('y_hat: ' + str(y_hat))

    if explanation.get('index') is not None:
        y, y_hat = y[explanation.index], y_hat[explanation.index]

    if explainer.model_config.return_type == ModelReturnType.raw:
        y, y_hat = y.softmax(dim=-1), y_hat.softmax(dim=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = y.exp(), y_hat.exp()

    kl_div = F.kl_div(y.log(), y_hat, reduction='batchmean')
    # print('kl_div: ' + str(kl_div))
    return 1 - float(torch.exp(-kl_div))

class HeteroGraphExplainer():
    def __init__(
        self,
        dataset: SampleEHRDataset,
        model: GNN,
        label_key: str,
        k: int,
    ):
        self.dataset = dataset
        self.model = model
        self.label_key = label_key
        self.k = k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_tokenizer = self.model.get_label_tokenizer()
        
        self.proc_df, self.symp_df, self.drug_df, self.diag_df, self.stat_kg_df = self.get_dataframe()

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

        self.explainer = Explainer(
            model=self.model.layer,
            # HYPERPARAMETERS
            algorithm=CaptumExplainer('IntegratedGradients',
                                        n_steps=500,
                                        method='riemann_trapezoid'
                                      ),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='edge',
                return_type='probs',
            ),
            node_mask_type='attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type='topk',
                value=10,
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

        # ==== GRAPH ENRICHMENT ====
        STATIC_KG_DF = {}

        if self.k > 0:
            for relation in self.static_kg:
                if self.k == 1:
                    if relation in ["ANAT_DIAG", "PC_DRUG"]:
                        continue

                # read table
                STATIC_KG_DF[relation] = pd.read_csv(
                    os.path.join(self.root, f"{relation}.csv"),
                    low_memory=False,
                    index_col=0,
                )

        return PROCEDURES, SYMPTOMS, DRUGS, DIAGNOSES, STATIC_KG_DF

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

        self.n = n
        self.explanation =self.explainer(
            x = self.node_features,
            edge_index = self.subgraph.edge_index_dict,
            edge_label_index = self.subgraph['visit', 'drug'].edge_label_index[:, self.n],
            edge_label = self.subgraph['visit', 'drug'].edge_label[self.n],
            mask = self.mask
        )
        print(f'Generated explanations in {self.explanation.available_explanations}')

        path = f'./explainability_results/feature_importance_{current_datetime}.png'

        self.explanation.detach()
        self.explanation.visualize_feature_importance(path, top_k=10)
        self.explanation['prediction'] = F.sigmoid(self.explanation['prediction'])

        print(f"Feature importance plot has been saved to '{path}'")
        print('Edge to predict: ' + str(self.subgraph['visit', 'drug'].edge_label_index[:, n]))
        print('Label to predict: ' + str(self.subgraph['visit', 'drug'].edge_label[n].numpy().astype(int)))
        print('Label predicted: ' + str(1 if self.explanation['prediction'].numpy() > 0.5 else 0) + " because " + str(self.explanation['prediction'].numpy()))

        return

    def explain_graph(
        self
    ):

        rcParams['figure.figsize'] = 14, 10

        # Creazione del grafo NetworkX
        self.G = nx.DiGraph()

        # Definizione dei colori per i diversi tipi di nodi
        node_colors = {
            'patient': 'lightblue',
            'visit': 'red',
            'symptom': 'lightgreen',
            'procedure': 'lightsalmon',
            'disease': 'lightcoral',
            'drug': 'yellow'
        }

        # Assumi che `self.explanation` sia un oggetto definito precedentemente con i dati necessari
        entities = [('patient', self.explanation['patient']),
                    ('visit', self.explanation['visit']),
                    ('symptom', self.explanation['symptom']),
                    ('procedure', self.explanation['procedure']),
                    ('disease', self.explanation['disease']),
                    ('drug', self.explanation['drug'])]

        nodess = []
        for edge_type, edge_data in [(edge[0], edge[1]) for edge in self.explanation.edge_items()]:
            for i in range(edge_data['edge_index'].shape[1]):
                source_id = f"{edge_type[0]}_{edge_data['edge_index'][0, i]}"
                target_id = f"{edge_type[2]}_{edge_data['edge_index'][1, i]}"
                
                edge_mask = self.explanation[edge_type]['edge_mask'][i]

                if edge_mask > 0:
                    if source_id not in nodess:
                        nodess.append(source_id)
                    if target_id not in nodess:
                        nodess.append(target_id)
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
                    node_size = max(10, node_mask_value * 10000)
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
        filepath = f'./explainability_results/explain_graph{current_datetime}.html'
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

        return

    def explain_results(
        self
    ):

        icd = InnerMap.load("ICD9CM")
        icdpr = InnerMap.load("ICD9PROC")
        atc = InnerMap.load("ATC")

        for node in self.G.nodes:
            node_type, id = str(node).split('_')
            if node_type == "patient":
                print(node)
            if node_type == "visit":
                print(node)
            if node_type == "disease":
                print(str(node) + ": " + str(icd.lookup(list(self.icd9_diag_dict.keys())[int(id)])))
            if node_type == "symptom":
                print(str(node) + ": " + str(icd.lookup(list(self.icd9_symp_dict.keys())[int(id)])))
            if node_type == "procedure":
                print(str(node) + ": " + str(icdpr.lookup(list(self.icd9_proc_dict.keys())[int(id)])))
            if node_type == "drug":
                print(str(node) + ": " + str(atc.lookup(list(self.atc_pre_dict.keys())[int(id)])))

        return