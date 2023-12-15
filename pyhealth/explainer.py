import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union

import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.utils import to_networkx
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset

from torch_geometric.explain import CaptumExplainer, Explainer

from pyvis.network import Network
from pylab import rcParams
import webbrowser

from torch import Tensor
from torch_geometric.explain import Explainer, HeteroExplanation, Explanation
from torch_geometric.explain.config import ExplanationType, ModelMode
from torch_geometric.explain.metric import characterization_score

class HeteroGraphExplainer():
    def __init__(
        self,
        dataset: SampleEHRDataset,
        model: BaseModel,
        label_key: str,
        k: int,

    ):
        # super(HeteroGraphExplainer, self).__init__(
        #     dataset=dataset,
        #     label_key=label_key,
        # )

        self.dataset = dataset
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_key = label_key
        self.label_tokenizer = self.model.get_label_tokenizer()
        self.k = k
        
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

        # self.edge_index_patient_to_visit, self.edge_index_visit_to_symptom, \
        #     self.edge_index_visit_to_disease, self.edge_index_visit_to_procedure, \
        #     self.edge_index_visit_to_drug, self.edge_index_disease_to_symptom, \
        #     self.edge_index_anatomy_to_diagnosis, self.edge_index_diagnosis_to_drug, \
        #     self.edge_index_pharma_to_drug, self.edge_index_symptom_to_drug = self.get_edge_index()

        # self.graph = self.graph_definition()
        # self.x_dict = self.model.x_dict
        self.explainer = Explainer(
                    model=self.model.layer,
                    algorithm=CaptumExplainer('IntegratedGradients',
                                               n_steps=300,
                                               method='riemann_trapezoid',
                                               internal_batch_size=self.subgraph['visit', 'drug'].edge_index.shape[1]), # HYPERPARAMETERS
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
                        value=3,
                    ),
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
            # target = test_data['patient', 'disease'].edge_label[n].unsqueeze(dim=0).long(),
            edge_label_index = self.subgraph['visit', 'drug'].edge_label_index[:, self.n],
            edge_label = self.subgraph['visit', 'drug'].edge_label[self.n],
            mask = self.mask
        )
        print(f'Generated explanations in {self.explanation.available_explanations}')

        path = 'feature_importance1.png'
        self.explanation.detach()
        self.explanation.visualize_feature_importance(path, top_k=10)
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
        G = nx.DiGraph()

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
                source_node_mask = self.explanation[edge_type[0]]['node_mask'][edge_data['edge_index'][0, i]]
                target_node_mask = self.explanation[edge_type[2]]['node_mask'][edge_data['edge_index'][1, i]]

                if edge_mask > 0:
                    if source_id not in nodess:
                        nodess.append(source_id)
                    if target_id not in nodess:
                        nodess.append(target_id)
                    # Aggiungi l'arco con il peso limitato
                    G.add_edge(source_id, target_id, weight=min(100, 20000 * edge_mask.numpy()))

        for entity_type, entity_data in entities:
            for i in range(entity_data['x'].shape[0]):
                node_id = f"{entity_type}_{i}"
                if node_id in nodess:
                    G.add_node(node_id, type=entity_type)

        # Converti il grafo NetworkX in un grafo Pyvis
        net = Network(notebook=True, height="750px", width="100%")
        net.from_nx(G)

        # Assegna i colori ai nodi in Pyvis
        for node in net.nodes:
            node['color'] = node_colors[node['type']]

        # Crea una legenda HTML
        legend_html = "<div style='position:absolute; right:20px; top:20px; width:200px; background-color:rgba(255,255,255,0.8); padding:10px; border-radius:5px; font-size:14px;'>"
        legend_html += "<strong>Node Types</strong><br>"
        for node_type, color in node_colors.items():
            legend_html += f"<span style='margin-left:10px; color:{color};'>{node_type}</span><br>"
        legend_html += "</div>"

        # Aggiungi la legenda al grafo Pyvis
        net.html = legend_html

        # Visualizza il grafo in un file HTML
        net.show('nx.html')
        webbrowser.open('nx.html')

        plt.show()

    def explain_metrics(
        self,
        metrics: List[str] = 'Fidelity',
    ):

        explainer = self.explainer
        explanation = self.explanation

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
                'edge_label_index': self.subgraph['visit', 'drug'].edge_label_index,
                'edge_label': self.subgraph['visit', 'drug'].edge_label,
                'mask': self.mask,
            }

        y = explanation.target
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = explainer.get_prediction(
                self.node_features,
                self.subgraph.edge_index_dict,
                **kwargs,
            )
            y_hat = explainer.get_target(y_hat)

        explain_y_hat = explainer.get_masked_prediction(
            self.node_features,
            self.subgraph.edge_index_dict,
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
            self.node_features,
            self.subgraph.edge_index_dict,
            node_mask_dict,
            edge_mask_dict,
            **kwargs,
        )
        # print('complement_y_hat: ' + str(complement_y_hat))
        complement_y_hat = explainer.get_target(complement_y_hat)

        # print('complement_y_hat: ' + str(complement_y_hat.shape))
        
        y = self.subgraph['visit', 'drug'].edge_label
            
        if explanation.get('index') is not None:
            if explainer.explanation_type == ExplanationType.phenomenon:
                y_hat = y_hat[explanation.index]
            explain_y_hat = explain_y_hat[explanation.index]
            complement_y_hat = complement_y_hat[explanation.index]

        print('complement_y_hat: ' + str(complement_y_hat))
        print('explain_y_hat: ' + str(explain_y_hat))
        print('y: ' + str(y))
        
        if explainer.explanation_type == ExplanationType.model:
            pos_fidelity = 1. - (complement_y_hat == y).float().mean()
            neg_fidelity = 1. - (explain_y_hat == y).float().mean()
        else:
            pos_fidelity = ((y_hat == y).float() -
                            (complement_y_hat == y).float()).abs().mean()
            neg_fidelity = ((y_hat == y).float() -
                            (explain_y_hat == y).float()).abs().mean()

        
        for metric in metrics:
            if metric == "Fidelity":
                print("Fidelity Positive: " + str(float(pos_fidelity)))
                print("Fidelity Negative: " + str(float(neg_fidelity)))
            elif metric == "Characterization_Score":
                score = characterization_score(float(pos_fidelity), float(neg_fidelity))
                print("Characterization Score: " + str(score))
                
        return 
























    # def get_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Get the edge indices for the graph.

    #     Returns:
    #         A tuple of torch.Tensor containing the edge indices for different relationships in the graph:
    #         - edge_index_patient_to_visit: Edge indices pointing from patients to visits.
    #         - edge_index_visit_to_symptom: Edge indices pointing from visits to symptoms.
    #         - edge_index_visit_to_disease: Edge indices pointing from visits to diseases.
    #         - edge_index_visit_to_procedure: Edge indices pointing from visits to procedures.
    #         - edge_index_visit_to_drug: Edge indices pointing from visits to drugs.
    #         - edge_index_disease_to_symptom: Edge indices pointing from diseases to symptoms.
    #         - edge_index_anatomy_to_diagnosis: Edge indices pointing from anatomy to diagnosis.
    #         - edge_index_diagnosis_to_drug: Edge indices pointing from diagnosis to drugs.
    #         - edge_index_pharma_to_drug: Edge indices pointing from pharma to drugs.
    #         - edge_index_symptom_to_drug: Edge indices pointing from symptoms to drugs.
    #     """
    #     # =============== MAPPING VISITS ===========================
    #     # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
    #     self.symp_df['HADM_ID'] = self.symp_df['HADM_ID'].map(self.hadm_dict)
        
    #     # Substituting the values in the 'SUBJECT_ID' column with the corresponding indices in the vocabulary
    #     self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(self.subject_dict)

    #     has_patient_id = torch.from_numpy(self.symp_df['SUBJECT_ID'].values)
    #     has_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)

    #     # Create the edge index for the relationship 'has' between patients and visits
    #     edge_index_patient_to_visit = torch.stack([has_patient_id, has_visit_id], dim=0)

    #     # =============== MAPPING SYMPTOMS ===========================
    #     # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
    #     self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)

    #     presents_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)
    #     presents_symptom_id = torch.from_numpy(self.symp_df['ICD9_CODE'].values)

    #     # Create the edge index for the relationship 'presents' between visits and symptoms
    #     edge_index_visit_to_symptom = torch.stack([presents_visit_id, presents_symptom_id], dim=0)

    #     # =============== MAPPING DIAGNOSES ===========================
    #     # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
    #     self.diag_df['ICD9_CODE_DIAG'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
    #     # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
    #     self.diag_df['HADM_ID'] = self.diag_df['HADM_ID'].map(self.hadm_dict)

    #     # Drop the 'ICD9_CODE' column that is no longer needed
    #     self.diag_df.drop('ICD9_CODE', axis=1, inplace=True)

    #     hasdisease_visit_id = torch.from_numpy(self.diag_df['HADM_ID'].values)
    #     hasdisease_disease_id = torch.from_numpy(self.diag_df['ICD9_CODE_DIAG'].values)

    #     # Create the edge index for the relationship 'has' between visits and diseases
    #     edge_index_visit_to_disease = torch.stack([hasdisease_visit_id, hasdisease_disease_id], dim=0)

    #     # =============== MAPPING PROCEDURES ===========================
    #     # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
    #     self.proc_df['ICD9_CODE_PROC'] = self.proc_df['ICD9_CODE'].map(self.icd9_proc_dict)
    #     # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
    #     self.proc_df['HADM_ID'] = self.proc_df['HADM_ID'].map(self.hadm_dict)

    #     # Drop the 'ICD9_CODE' column that is no longer needed
    #     self.proc_df.drop('ICD9_CODE', axis=1, inplace=True)

    #     hastreat_visit_id = torch.from_numpy(self.proc_df['HADM_ID'].values)
    #     hastreat_procedure_id = torch.from_numpy(self.proc_df['ICD9_CODE_PROC'].values)

    #     # Create the edge index for the relationship 'has_treat' between visits and procedures
    #     edge_index_visit_to_procedure = torch.stack([hastreat_visit_id, hastreat_procedure_id], dim=0)

    #     # =============== MAPPING DRUGS ===========================
    #     # Substituting the values in the 'ATC_CODE' column with the corresponding indices in the vocabulary
    #     self.drug_df['ATC_CODE_PRE'] = self.drug_df['ATC_CODE'].map(self.atc_pre_dict)
    #     # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
    #     self.drug_df['HADM_ID'] = self.drug_df['HADM_ID'].map(self.hadm_dict)

    #     # Drop the 'ATC_CODE' column that is no longer needed
    #     self.drug_df.drop('ATC_CODE', axis=1, inplace=True)

    #     hasreceived_visit_id = torch.from_numpy(self.drug_df['HADM_ID'].values)
    #     hasreceived_drug_id = torch.from_numpy(self.drug_df['ATC_CODE_PRE'].values)

    #     # Create the edge index for the relationship 'has_received' between visits and drugs
    #     edge_index_visit_to_drug = torch.stack([hasreceived_visit_id, hasreceived_drug_id], dim=0)

    #     # ==== GRAPH ENRICHMENT ====
    #     edge_index_disease_to_symptom = None
    #     edge_index_anatomy_to_diagnosis = None
    #     edge_index_diagnosis_to_drug = None
    #     edge_index_pharma_to_drug = None
    #     edge_index_symptom_to_drug = None

    #     if self.k > 0:
    #         for relation in self.static_kg:
    #             if relation == "DIAG_SYMP":
    #                 # =============== MAPPING DIAG_SYMP ===========================
    #                 # Copy the dataframe with the relationship DIAG_SYMP
    #                 diag_symp_df = self.stat_kg_df[relation].astype(str).copy()
    #                 diag_symp_df = diag_symp_df[diag_symp_df["DIAG"].isin(self.icd9_diag_dict.keys())]

    #                 # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
    #                 diag_symp_df['DIAG'] = diag_symp_df['DIAG'].map(self.icd9_diag_dict)

    #                 # Lookup the indices of the symptoms in the vocabulary
    #                 last_index = max(self.icd9_symp_dict.values())

    #                 # Add the new symptoms to the dictionary with consecutive indices
    #                 for symptom_code in diag_symp_df['SYMP'].unique():
    #                     if symptom_code not in self.icd9_symp_dict:
    #                         last_index += 1
    #                         self.icd9_symp_dict[symptom_code] = last_index
    #                         self.symp_df = pd.concat([self.symp_df, pd.DataFrame({'SUBJECT_ID': [0], 'HADM_ID': [0], 'SEQ_NUM': [0], 'ICD9_CODE': [symptom_code]})], ignore_index=True)
    #                 diag_symp_df['SYMP'] = diag_symp_df['SYMP'].map(self.icd9_symp_dict)

    #                 if not anat_diag_df.empty:
    #                     hasbeencaused_diag_id = torch.from_numpy(diag_symp_df['DIAG'].values)
    #                     hasbeencaused_symp_id = torch.from_numpy(diag_symp_df['SYMP'].values)
    #                     edge_index_disease_to_symptom = torch.stack([hasbeencaused_diag_id, hasbeencaused_symp_id], dim=0)
    #                 else:
    #                     # Initialize edge_index_disease_to_symptom as empty if the DataFrame is empty
    #                     edge_index_disease_to_symptom = torch.empty((2, 0), dtype=torch.int64)

    #             elif (relation == "ANAT_DIAG") and self.k == 2:
    #                 # =============== MAPPING ANAT_DIAG ===========================
    #                 # Copy the dataframe with the relationship ANAT_DIAG
    #                 anat_diag_df = self.stat_kg_df[relation].astype(str).copy()
    #                 anat_diag_df = anat_diag_df[anat_diag_df["DIAG"].isin(self.icd9_diag_dict.keys())]

    #                 # Create a unique vocabulary from the codici UBERON
    #                 uberon_anat_vocab = anat_diag_df['ANAT'].unique()
    #                 # Create a dictionary that maps the codici UBERON to their index in the vocabulary
    #                 uberon_anat_dict = {code: i for i, code in enumerate(uberon_anat_vocab)}

    #                 # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
    #                 anat_diag_df['DIAG'] = anat_diag_df['DIAG'].map(self.icd9_diag_dict)
    #                 # Substituting the values in the 'ANAT' column with the corresponding indices in the vocabulary
    #                 anat_diag_df['ANAT'] = anat_diag_df['ANAT'].map(uberon_anat_dict)

    #                 if not anat_diag_df.empty:
    #                     localizes_diag_id = torch.from_numpy(anat_diag_df['DIAG'].values)
    #                     localizes_anat_id = torch.from_numpy(anat_diag_df['ANAT'].values)
    #                     edge_index_anatomy_to_diagnosis = torch.stack([localizes_diag_id, localizes_anat_id], dim=0)
    #                 else:
    #                     # Initialize edge_index_anatomy_to_diagnosis as empty if the DataFrame is empty
    #                     edge_index_anatomy_to_diagnosis = torch.empty((2, 0), dtype=torch.int64)

    #             elif relation == "DRUG_DIAG":
    #                 # =============== MAPPING DRUG_DIAG ===========================
    #                 # Copy the dataframe with the relationship DRUG_DIAG
    #                 drug_diag_df = self.stat_kg_df[relation].astype(str).copy()
    #                 drug_diag_df = drug_diag_df[drug_diag_df["DIAG"].isin(self.icd9_diag_dict.keys())]
    #                 drug_diag_df = drug_diag_df[drug_diag_df["DRUG"].isin(self.atc_pre_dict.keys())]

    #                 # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
    #                 drug_diag_df['DIAG'] = drug_diag_df['DIAG'].map(self.icd9_diag_dict)
    #                 # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
    #                 drug_diag_df['DRUG'] = drug_diag_df['DRUG'].map(self.atc_pre_dict)

    #                 if not drug_diag_df.empty:
    #                     treats_diag_id = torch.from_numpy(drug_diag_df['DIAG'].values)
    #                     treats_drug_id = torch.from_numpy(drug_diag_df['DRUG'].values)
    #                     edge_index_diagnosis_to_drug = torch.stack([treats_diag_id, treats_drug_id], dim=0)
    #                 else:
    #                     # Initialize edge_index_diagnosis_to_drug as empty if the DataFrame is empty
    #                     edge_index_diagnosis_to_drug = torch.empty((2, 0), dtype=torch.int64)

    #             elif (relation == "PC_DRUG") and (self.k == 2):
    #                 # =============== MAPPING PC_DRUG ===========================
    #                 # Copy the dataframe with the relationship PC_DRUG
    #                 pc_drug_df = self.stat_kg_df[relation].astype(str).copy()
    #                 pc_drug_df = pc_drug_df[pc_drug_df["DRUG"].isin(self.atc_pre_dict.keys())]

    #                 # Create a unique vocabulary from the codici PHARMACLASS
    #                 ndc_pc_vocab = pc_drug_df['PHARMACLASS'].unique()
    #                 # Create a dictionary that maps the codici PHARMACLASS to their index in the vocabulary
    #                 ndc_pc_dict = {code: i for i, code in enumerate(ndc_pc_vocab)}

    #                 # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
    #                 pc_drug_df['DRUG'] = pc_drug_df['DRUG'].map(self.atc_pre_dict)
    #                 # Substituting the values in the 'PHARMACLASS' column with the corresponding indices in the vocabulary
    #                 pc_drug_df['PHARMACLASS'] = pc_drug_df['PHARMACLASS'].map(ndc_pc_dict)

    #                 if not pc_drug_df.empty:
    #                     includes_pharma_id = torch.from_numpy(pc_drug_df['PHARMACLASS'].values)
    #                     includes_drug_id = torch.from_numpy(pc_drug_df['DRUG'].values)
    #                     edge_index_pharma_to_drug = torch.stack([includes_pharma_id, includes_drug_id], dim=0)
    #                 else:
    #                     # Initialize edge_index_pharma_to_drug as empty if the DataFrame is empty
    #                     edge_index_pharma_to_drug = torch.empty((2, 0), dtype=torch.int64)

    #             elif relation == "SYMP_DRUG":
    #                 # =============== MAPPING SYMP_DRUG ===========================
    #                 # Copy the dataframe with the relationship SYMP_DRUG
    #                 symp_drug_df = self.stat_kg_df[relation].astype(str).copy()
    #                 symp_drug_df = symp_drug_df[symp_drug_df["DRUG"].isin(self.atc_pre_dict.keys())] ###OCCHIO QUI

    #                 # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
    #                 symp_drug_df['DRUG'] = symp_drug_df['DRUG'].map(self.atc_pre_dict)

    #                 # Lookup the indices of the symptoms in the vocabulary
    #                 last_index = max(self.icd9_symp_dict.values())

    #                 # Add the new symptoms to the dictionary with consecutive indices
    #                 for symptom_code in symp_drug_df['SYMP'].unique():
    #                     if symptom_code not in self.icd9_symp_dict:
    #                         last_index += 1
    #                         self.icd9_symp_dict[symptom_code] = last_index
    #                         self.symp_df = pd.concat([self.symp_df, pd.DataFrame({'SUBJECT_ID': [0], 'HADM_ID': [0], 'SEQ_NUM': [0], 'ICD9_CODE': [symptom_code]})], ignore_index=True)
    #                 symp_drug_df['SYMP'] = symp_drug_df['SYMP'].map(self.icd9_symp_dict)

    #                 if not symp_drug_df.empty:
    #                     causes_symp_id = torch.from_numpy(symp_drug_df['SYMP'].values)
    #                     causes_drug_id = torch.from_numpy(symp_drug_df['DRUG'].values)
    #                     edge_index_symptom_to_drug = torch.stack([causes_symp_id, causes_drug_id], dim=0)
    #                 else:
    #                     # Initialize edge_index_symptom_to_drug as empty if the DataFrame is empty
    #                     edge_index_symptom_to_drug = torch.empty((2, 0), dtype=torch.int64)

    #     return edge_index_patient_to_visit, edge_index_visit_to_symptom, edge_index_visit_to_disease, \
    #             edge_index_visit_to_procedure, edge_index_visit_to_drug, edge_index_disease_to_symptom, \
    #             edge_index_anatomy_to_diagnosis, edge_index_diagnosis_to_drug, edge_index_pharma_to_drug, \
    #             edge_index_symptom_to_drug


    # def graph_definition(self) -> HeteroData:
    #     """
    #     Defines the graph structure for the GNN model.

    #     Returns:
    #         HeteroData: The graph structure with node and edge indices.
    #     """
    #     # Graph definition:
    #     graph = HeteroData()

    #     # Save node indices:
    #     graph["patient"].node_id = torch.arange(len(self.symp_df['SUBJECT_ID'].unique()))
    #     graph["visit"].node_id = torch.arange(len(self.symp_df['HADM_ID'].unique()))
    #     graph["symptom"].node_id = torch.arange(len(self.symp_df['ICD9_CODE'].unique()))
    #     graph["procedure"].node_id = torch.arange(len(self.proc_df['ICD9_CODE_PROC'].unique()))

    #     # Nodes of Static KG
    #     if self.k == 2:
    #         for relation in self.static_kg:
    #             if relation == "ANAT_DIAG":
    #                 graph["anatomy"].node_id = torch.arange(len(self.stat_kg_df[relation]['ANAT'].unique()))
    #             if relation == "PC_DRUG":
    #                 graph["pharmaclass"].node_id = torch.arange(len(self.stat_kg_df[relation]['PHARMACLASS'].unique()))

    #     if self.label_key == "conditions":
    #         graph["disease"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())
    #         graph["drug"].node_id = torch.arange(len(self.drug_df['ATC_CODE_PRE'].unique()))
    #     else:
    #         graph["disease"].node_id = torch.arange(len(self.diag_df['ICD9_CODE_DIAG'].unique()))
    #         graph["drug"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())

    #     # Add the edge indices:
    #     graph["patient", "has", "visit"].edge_index = self.edge_index_patient_to_visit
    #     graph["visit", "presents", "symptom"].edge_index = self.edge_index_visit_to_symptom
    #     graph["visit", "has", "disease"].edge_index = self.edge_index_visit_to_disease
    #     graph["visit", "has_treat", "procedure"].edge_index = self.edge_index_visit_to_procedure
    #     graph["visit", "has_received", "drug"].edge_index = self.edge_index_visit_to_drug

    #     # Edges of Static KG
    #     if self.k > 0:
    #         for relation in self.static_kg:
    #             if relation == "DIAG_SYMP":
    #                 graph["disease", "has_been_caused_by", "symptom"].edge_index = self.edge_index_disease_to_symptom
    #             if (relation == "ANAT_DIAG") and (self.k == 2):
    #                 graph["disease", "localizes", "anatomy"].edge_index = self.edge_index_anatomy_to_diagnosis
    #             if relation == "DRUG_DIAG":
    #                 graph["disease", "treats", "drug"].edge_index = self.edge_index_diagnosis_to_drug
    #             if (relation == "PC_DRUG") and (self.k == 2):
    #                 graph["pharmaclass", "includes", "drug"].edge_index = self.edge_index_pharma_to_drug
    #             if relation == "SYMP_DRUG":
    #                 graph["symptom", "causes", "drug"].edge_index = self.edge_index_symptom_to_drug


    #     # We also need to make sure to add the reverse edges from movies to users
    #     # in order to let a GNN be able to pass messages in both directions.
    #     # We can leverage the `T.ToUndirected()` transform for this from PyG:
    #     graph = T.ToUndirected()(graph)

    #     return graph