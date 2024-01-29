import os
import torch
import datetime
import webbrowser
import pandas as pd
from pyvis.network import Network
import networkx as nx
from typing import Dict, List, Tuple, Union

from torch.nn import functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
from torch_geometric.explain import CaptumExplainer, DummyExplainer, Explainer
from torch_geometric.explain.metric import characterization_score, fidelity_curve_auc

from pyhealth.models import GNN
from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import InnerMap
from pyhealth.GNNExplainer import GNNExplainer
from pyhealth.explainer_metrics import *

# Save current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class HeteroGraphExplainer():
    def __init__(
        self,
        algorithm: str,
        dataset: SampleEHRDataset,
        model: GNN,
        label_key: str,
        threshold_value: float,
        top_k: int,
        feat_size: int = 32,
        root: str="./explainability_results/",
    ):
        self.dataset = dataset
        self.model = model
        self.label_key = label_key
        self.algorithm = algorithm
        self.threshold_value = threshold_value
        self.top_k = top_k
        self.feat_size = feat_size
        self.root = root

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_tokenizer = self.model.get_label_tokenizer()

        self.proc_df, self.symp_df, self.medication_df, self.diag_df = self.get_dataframe()

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
        elif self.algorithm == "Saliency":
            explainer_algorithm = CaptumExplainer('Saliency',
                                                 )
            type_returned = "probs"
        elif self.algorithm == "GNNExplainer":
            explainer_algorithm = GNNExplainer(epochs=100,
                                               lr=0.2,
                                               )
            type_returned = "raw"
        elif self.algorithm == "DummyExplainer":
            explainer_algorithm = DummyExplainer()
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

    def get_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                     Union[None, Dict[str, pd.DataFrame]]]:
        """Gets the dataframe of diagnosis, procedures, symptoms and medications of patients.

        Returns:
            dataframe: a `pandas.DataFrame` object.
        """
        PROCEDURES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        SYMPTOMS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        MEDICATIONS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ATC_CODE'])
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

            if self.label_key == "medications":
                medications_data = patient_data['medications']
                diagnoses_data = patient_data['diagnosis'][-1]
            elif self.label_key == "diagnosis":
                medications_data = patient_data['medications'][-1]
                diagnoses_data = patient_data['diagnosis']

            # MEDICATIONS DataFrame
            medications_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(medications_data),
                                    'HADM_ID': [hadm_id] * len(medications_data),
                                    'ATC_CODE': medications_data})
            MEDICATIONS = pd.concat([MEDICATIONS, medications_df], ignore_index=True)

            # DIAGNOSES DataFrame
            diagnoses_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(diagnoses_data),
                                        'HADM_ID': [hadm_id] * len(diagnoses_data),
                                        'SEQ_NUM': range(1, len(diagnoses_data) + 1),
                                        'ICD9_CODE': diagnoses_data})
            DIAGNOSES = pd.concat([DIAGNOSES, diagnoses_df], ignore_index=True)

        return PROCEDURES, SYMPTOMS, MEDICATIONS, DIAGNOSES

    def get_subgraph(self) -> HeteroData:
        """
        Returns a subgraph containing selected patients, visits, symptoms, procedures and diagnosis.

        Returns:
            subgraph (HeteroData): A subgraph containing selected patients and visits.
        """
        # ==== DATA SELECTION ====
        # Select the patients, visits, symptoms, procedures and diagnosis from the graph
        self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(self.subject_dict)
        patient = self.symp_df["SUBJECT_ID"].unique()
        select_patient = torch.from_numpy(patient)
 
        self.symp_df['HADM_ID'] = self.symp_df['HADM_ID'].map(self.hadm_dict)
        visit = self.symp_df["HADM_ID"].unique()
        select_visit = torch.from_numpy(visit)
 
        self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)
        symptom = self.symp_df["ICD9_CODE"].unique()
        select_symptom = torch.from_numpy(symptom)
 
        self.proc_df['ICD9_CODE'] = self.proc_df['ICD9_CODE'].map(self.icd9_proc_dict)
        procedure = self.proc_df["ICD9_CODE"].unique()
        select_procedure = torch.from_numpy(procedure)
 
        if self.label_key == "medications":
            self.diag_df['ICD9_CODE'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
            diagnosis = self.diag_df["ICD9_CODE"].unique()
            select_diagnosis = torch.from_numpy(diagnosis)

            subgraph = self.model.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "diagnosis": select_diagnosis})

        elif self.label_key == "diagnosis":
            self.medication_df['ATC_CODE'] = self.medication_df['ATC_CODE'].map(self.atc_pre_dict)
            medication = self.medication_df["ATC_CODE"].unique()
            select_medication = torch.from_numpy(medication)

            subgraph = self.model.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "medication": select_medication})

        return subgraph

    def generate_neg_samples(self) -> HeteroData:
        """
        Generates negative samples for the subgraph based on the label key.

        Returns:
            HeteroData: The updated subgraph with negative samples.
        """
        if self.label_key == "medications":
            neg_edges = negative_sampling(self.subgraph['visit', 'has_received', 'medication'].edge_index, num_nodes=(self.subgraph['visit'].num_nodes, self.subgraph['medication'].num_nodes))
            self.subgraph['visit', 'has_received', 'medication'].edge_label_index = self.subgraph['visit', 'has_received', 'medication'].edge_index
            self.subgraph['visit', 'has_received', 'medication'].edge_label = torch.ones(self.subgraph['visit', 'has_received', 'medication'].edge_label_index.shape[1], dtype=torch.float)
            self.subgraph['visit', 'has_received', 'medication'].edge_label_index = torch.cat((self.subgraph['visit', 'has_received', 'medication'].edge_label_index, neg_edges), dim=1)
            self.subgraph['visit', 'has_received', 'medication'].edge_label = torch.cat((self.subgraph['visit', 'has_received', 'medication'].edge_label, torch.zeros(neg_edges.shape[1], dtype=torch.float)), dim=0)

        elif self.label_key == "diagnosis":
            neg_edges = negative_sampling(self.subgraph['visit', 'has', 'diagnosis'].edge_index, num_nodes=(self.subgraph['visit'].num_nodes, self.subgraph['diagnosis'].num_nodes))
            self.subgraph['visit', 'has', 'diagnosis'].edge_label_index = self.subgraph['visit', 'has', 'diagnosis'].edge_index
            self.subgraph['visit', 'has', 'diagnosis'].edge_label = torch.ones(self.subgraph['visit', 'has', 'diagnosis'].edge_label_index.shape[1], dtype=torch.float)
            self.subgraph['visit', 'has', 'diagnosis'].edge_label_index = torch.cat((self.subgraph['visit', 'has', 'diagnosis'].edge_label_index, neg_edges), dim=1)
            self.subgraph['visit', 'has', 'diagnosis'].edge_label = torch.cat((self.subgraph['visit', 'has', 'diagnosis'].edge_label, torch.zeros(neg_edges.shape[1], dtype=torch.float)), dim=0)

        return self.subgraph

    def generate_mask(self) -> torch.Tensor:
        """
        Generates a mask for the subgraph edges based on the label key.

        Returns:
            torch.Tensor: The generated mask.
        """
        if self.label_key == "medications":
            mask = torch.ones_like(self.subgraph['visit', 'has_received', 'medication'].edge_label, dtype=torch.bool, device=self.device)

            # Get all possible edges in the graph
            all_possible_edges = torch.cartesian_prod(torch.arange(self.subgraph['visit'].num_nodes), torch.arange(self.label_tokenizer.get_vocabulary_size()))

            # Filter existing edges in the current graph
            existing_edges = self.subgraph['visit', 'has_received', 'medication'].edge_label_index.t().contiguous()

            # Find missing edges in the current graph
            missing_edges = torch.tensor(list(set(map(tuple, all_possible_edges.tolist())) - set(map(tuple, existing_edges.tolist())))).t().contiguous()

            self.subgraph['visit', 'has_received', 'medication'].edge_label_index = torch.cat([self.subgraph['visit', 'has_received', 'medication'].edge_label_index, missing_edges], dim=1)
            self.subgraph['visit', 'has_received', 'medication'].edge_label = torch.cat([self.subgraph['visit', 'has_received', 'medication'].edge_label, torch.zeros(missing_edges.size(1), dtype=torch.float)], dim=0)

        elif self.label_key == "diagnosis":
            mask = torch.ones_like(self.subgraph['visit', 'has', 'diagnosis'].edge_label, dtype=torch.bool, device=self.device)

            # Get all possible edges in the graph
            all_possible_edges = torch.cartesian_prod(torch.arange(self.subgraph['visit'].num_nodes), torch.arange(self.label_tokenizer.get_vocabulary_size()))

            # Filter existing edges in the current graph
            existing_edges = self.subgraph['visit', 'has', 'diagnosis'].edge_label_index.t().contiguous()

            # Find missing edges in the current graph
            missing_edges = torch.tensor(list(set(map(tuple, all_possible_edges.tolist())) - set(map(tuple, existing_edges.tolist())))).t().contiguous()

            self.subgraph['visit', 'has', 'diagnosis'].edge_label_index = torch.cat([self.subgraph['visit', 'has', 'diagnosis'].edge_label_index, missing_edges], dim=1)
            self.subgraph['visit', 'has', 'diagnosis'].edge_label = torch.cat([self.subgraph['visit', 'has', 'diagnosis'].edge_label, torch.zeros(missing_edges.size(1), dtype=torch.float)], dim=0)

        # Extend the mask with False for missing edges
        mask = torch.cat([mask, torch.zeros(missing_edges.size(1), dtype=torch.bool, device=self.device)], dim=0)

        return mask

    def explain(
        self,
        n: int
    ):
        self.n = n
        if self.label_key == "medications":
            edge_label_index = self.subgraph['visit', 'medication'].edge_label_index[:, n]

        elif self.label_key == "diagnosis":
            edge_label_index = self.subgraph['visit', 'diagnosis'].edge_label_index[:, n]

        self.explanation =self.explainer(
            x = self.node_features,
            edge_index = self.subgraph.edge_index_dict,
            edge_label_index = edge_label_index,
        )
        print(f'Generated explanations in {self.explanation.available_explanations}')

        path = f'{self.root}feature_importance_{current_datetime}.png'

        self.explanation.detach()
        self.explanation.visualize_feature_importance(path, top_k=self.top_k)
        self.explanation['prediction'] = F.sigmoid(self.explanation['prediction'])

        print(f"Feature importance plot has been saved to '{path}'")
        if self.label_key == "medications":
            print('Edge to predict: ' + str(self.subgraph['visit', 'medication'].edge_label_index[:, n]))
            print('Label to predict: ' + str(self.subgraph['visit', 'medication'].edge_label[n].cpu().numpy().astype(int)))

        elif self.label_key == "diagnosis":
            print('Edge to predict: ' + str(self.subgraph['visit', 'diagnosis'].edge_label_index[:, n]))
            print('Label to predict: ' + str(self.subgraph['visit', 'diagnosis'].edge_label[n].cpu().numpy().astype(int)))

        print('Label predicted: ' + str(1 if self.explanation['prediction'].cpu().numpy() > 0.5 else 0) + 
              " because " + str(self.explanation['prediction'].cpu().numpy()))

        return

    def explain_graph(
        self,
        k: int = 0,
        human_readable: bool = False,
        dashboard: bool = False,
    ):

        # Creazione del grafo NetworkX
        self.G = nx.DiGraph()

        icd = InnerMap.load("ICD9CM")
        icdpr = InnerMap.load("ICD9PROC")
        atc = InnerMap.load("ATC")

        # Definizione dei colori per i diversi tipi di nodi
        node_colors = {
            'patient': '#20b2aa',  # light sea green
            'visit': '#fa8072',    # salmon
            'symptom': '#98fb98',  # pale green
            'procedure': '#da70d6', # orchid
            'diagnosis': '#cd853f',  # peru
            'medication': '#87ceeb',      # sky blue
        }
        if k >= 2:
            node_colors['anatomy'] = '#bc8f8f'  # rosy brown
            node_colors['pharmaclass'] = '#483d8b'  # dark slate blue

        # Definizione dei colori per i diversi tipi di archi
        edge_colors = {
            ('patient', 'visit'): '#20b2aa',
            ('visit', 'symptom'): '#98fb98',
            ('visit', 'procedure'): '#da70d6',
            ('visit', 'diagnosis'): '#cd853f',
            ('visit', 'medication'): '#87ceeb',
            ('visit', 'patient'): '#20b2aa',
            ('symptom', 'visit'): '#98fb98',
            ('procedure', 'visit'): '#da70d6',
            ('diagnosis', 'visit'): '#cd853f',
            ('medication', 'visit'): '#87ceeb',
        }
        if k >= 2:
            edge_colors[('visit', 'anatomy')] = '#bc8f8f'
            edge_colors[('visit', 'pharmaclass')] = '#483d8b'
            edge_colors[('anatomy', 'visit')] = '#bc8f8f'
            edge_colors[('pharmaclass', 'visit')] = '#483d8b'

        atc_list = sorted(self.atc_pre_dict.keys(), key=lambda x: self.atc_pre_dict[x])
        icd9_diag_list = sorted(self.icd9_diag_dict.keys(), key=lambda x: self.icd9_diag_dict[x])
        icd9_proc_list = sorted(self.icd9_proc_dict.keys(), key=lambda x: self.icd9_proc_dict[x])
        icd9_symp_list = sorted(self.icd9_symp_dict.keys(), key=lambda x: self.icd9_symp_dict[x])

        # MI PRENDO I NODI PIU' IMPORTANTI
        nodess = {}
        for node_type, node_data in self.explanation.node_items():
            for i in range(node_data['node_mask'].shape[0]):
                node_id = f"{node_type}_{i}"
                node_mask = node_data['node_mask'][i]

                if node_mask.max() > 0:
                    if node_id not in nodess:
                        nodess.update({node_id: node_mask.max().item()})

                        if human_readable:
                            node_type_d, id = str(node_id).split('_')
                            if node_type_d == "symptom":
                                symptom_icd = icd.lookup(icd9_symp_list[int(id)])
                                print(f"{node_type} {id} {symptom_icd} Importance: " + str(node_mask.max()))
                                # Calcola la dimensione del nodo
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(symptom_icd, type=node_type, size=node_size)
                            
                            elif node_type_d == "procedure":
                                procedure_icd = icdpr.lookup(icd9_proc_list[int(id)])
                                print(f"{node_type} {id} {procedure_icd} Importance: " + str(node_mask.max()))
                                # Calcola la dimensione del nodo
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(procedure_icd, type=node_type, size=node_size)

                            elif node_type_d == "diagnosis":
                                diagnosis_icd = icd.lookup(icd9_diag_list[int(id)])
                                print(f"{node_type} {id} {diagnosis_icd} Importance: " + str(node_mask.max()))
                                # Calcola la dimensione del nodo
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(diagnosis_icd, type=node_type, size=node_size)

                            elif node_type_d == "medication":
                                medication_atc = atc.lookup(atc_list[int(id)])
                                print(f"{node_type} {id} {medication_atc} Importance: " + str(node_mask.max()))
                                # Calcola la dimensione del nodo
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(medication_atc, type=node_type, size=node_size)

                            # if k >= 2:
                            #     if node_type_d == "anatomy":
                            #         anatomy_atc = atc.lookup(list(self.atc_pre_dict.keys())[int(id)])
                            #         # Calcola la dimensione del nodo
                            #         node_size = max(10, node_mask.max().item() * 20)
                            #         self.G.add_node(anatomy_atc, type=node_type, size=node_size)

                            #     elif node_type_d == "pharmaclass":
                            #         pharmaclass_atc = atc.lookup(list(self.atc_pre_dict.keys())[int(id)])
                            #         # Calcola la dimensione del nodo
                            #         node_size = max(10, node_mask.max().item() * 20)
                            #         self.G.add_node(pharmaclass_atc, type=node_type, size=node_size)
                            
                            elif node_type_d == "patient":
                                print(f"{node_type} {id} Importance: " + str(node_mask.max()))
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(node_id, type=node_type, size=node_size)

                            elif node_type_d == "visit":
                                print(f"{node_type} {id} Importance: " + str(node_mask.max()))
                                node_size = max(10, node_mask.max().item() * 20)
                                self.G.add_node(node_id, type=node_type, size=node_size)
                        else:
                            print(f"{node_type} {i} Importance: " + str(node_mask.max()))
                            # Calcola la dimensione del nodo
                            node_size = max(10, node_mask.max().item() * 20)
                            self.G.add_node(node_id, type=node_type, size=node_size)

        self.nodess = nodess
        
        for edge_type, edge_data in [(edge[0], edge[1]) for edge in self.explanation.edge_items()]:
            for i in range(edge_data['edge_index'].shape[1]):
                source_id = f"{edge_type[0]}_{edge_data['edge_index'][0, i]}"
                target_id = f"{edge_type[2]}_{edge_data['edge_index'][1, i]}"

                # edge_mask = self.explanation[edge_type]['edge_mask'][i]

                # if edge_mask > 0:
                    # print(source_id + " -> " + target_id + " Importance: " + str(edge_mask))
                if human_readable:
                    if source_id in self.nodess.keys() and target_id in self.nodess.keys():
                        source_id_d, id_s = str(source_id).split('_')
                        target_id_d, id_t = str(target_id).split('_')

                        ############ SOURCE ID ###########
                        if source_id_d == "symptom":
                            symptom_icd = icd.lookup(icd9_symp_list[int(id_s)])
                            source_id = symptom_icd

                        elif source_id_d == "procedure":
                            procedure_icd = icdpr.lookup(icd9_proc_list[int(id_s)])
                            source_id = procedure_icd

                        elif source_id_d == "diagnosis":
                            diagnosis_icd = icd.lookup(icd9_diag_list[int(id_s)])
                            source_id = diagnosis_icd

                        elif source_id_d == "medication":
                            medication_atc = atc.lookup(atc_list[int(id_s)])
                            source_id = medication_atc

                        elif source_id_d == "patient":
                            source_id = source_id

                        elif source_id_d == "visit":
                            source_id = source_id

                        ############ TARGET ID ############
                        if target_id_d == "symptom":
                            symptom_icd = icd.lookup(icd9_symp_list[int(id_t)])
                            target_id = symptom_icd

                        elif target_id_d == "procedure":
                            procedure_icd = icdpr.lookup(icd9_proc_list[int(id_t)])
                            target_id = procedure_icd

                        elif target_id_d == "diagnosis":
                            diagnosis_icd = icd.lookup(icd9_diag_list[int(id_t)])
                            target_id = diagnosis_icd

                        elif target_id_d == "medication":
                            medication_atc = atc.lookup(atc_list[int(id_t)])
                            target_id = medication_atc

                        elif target_id_d == "patient":
                            target_id = target_id

                        elif target_id_d == "visit":
                            target_id = target_id

                        self.G.add_edge(source_id, target_id, type=(edge_type[0], edge_type[2]))

                else:
                    if source_id in self.nodess.keys() and target_id in self.nodess.keys():
                        self.G.add_edge(source_id, target_id, type=(edge_type[0], edge_type[2]))

        # # Add Legend Nodes
        # step = 120
        # x = 550
        # y = -350

        # legend_nodes = [
        #     (
        #         len(self.nodess) + i, 
        #         {
        #             'group': i, 
        #             'label': legend_type,
        #             'type': legend_type,
        #             'size': 90, 
        #             'fixed': True, 
        #             'physics': False, 
        #             'x': f'{x + i*step}px', 
        #             'y': y, 
        #             'shape': 'box', 
        #             'widthConstraint': 90, 
        #             'font': {'size': 15}
        #         }
        #     )
        #     for i, legend_type in enumerate(node_colors.keys())
        # ]
        # self.G.add_nodes_from(legend_nodes)

        # Converti il grafo NetworkX in un grafo Pyvis
        net = Network(notebook=True, height="500px", filter_menu=False, 
                      cdn_resources="remote")
        net.set_options('{"layout": {"randomSeed":5}}')
        net.from_nx(self.G)
        

        # Assegna i colori, le dimensioni e l'opacità ai nodi in Pyvis
        for i, node in enumerate(net.nodes):
            node['color'] = node_colors[node['type']]
            if 'size' in self.G.nodes[node['id']] and node['id'] in self.nodess.keys():
                node['size'] = self.G.nodes[node['id']]['size']
                if self.algorithm == "GNNExplainer":
                    node['opacity'] = self.nodess[node['id']] * 20
                else:
                    node['opacity'] = self.nodess[node['id']] * 20

        # Assegna i colori agli archi in Pyvis
        for edge in net.edges:
            edge['color'] = edge_colors[edge['type']]

        # Visualizza il grafo in un file HTML
        if dashboard:
            filepath = f'{self.root}explain_graph.html'
            net.show(filepath)
        else:
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
                if self.label_key == "medications":
                    pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'medication'].edge_label_index, 
                                                        self.subgraph['visit', 'medication'].edge_label)

                elif self.label_key == "diagnosis":
                    pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label_index, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label)
                print("Fidelity Positive: " + str(float(pos_fidelity)))
                print("Fidelity Negative: " + str(float(neg_fidelity)))

            elif metric == "Fidelity_F1":
                if self.label_key == "medications":
                    pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'medication'].edge_label_index, 
                                                        self.subgraph['visit', 'medication'].edge_label)

                elif self.label_key == "diagnosis":
                    pos_fidelity, neg_fidelity = fidelity(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label_index, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label)
                score = (2 * pos_fidelity * (1-neg_fidelity)) / (pos_fidelity + (1-neg_fidelity))
                print("Fidelity (weighted): " + str(score))

            elif metric == "Unfaithfulness":
                if self.label_key == "medications":
                    unfaithfulness_score = unfaithfulness(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'medication'].edge_label_index, 
                                                        self.subgraph['visit', 'medication'].edge_label,
                                                        top_k=self.feat_size)

                elif self.label_key == "diagnosis":
                    unfaithfulness_score = unfaithfulness(explainer, explanation, self.subgraph, self.node_features, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label_index, 
                                                        self.subgraph['visit', 'diagnosis'].edge_label,
                                                        top_k=self.feat_size)
                print("Unfaithfulness Score: " + str(unfaithfulness_score))

            elif metric == "Sparsity":
                sparsity_score = sparsity(explainer, explanation)
                print("Sparsity Score: " + str(sparsity_score))

            elif metric == "Instability":
                stability_score = stability(explainer, explanation, self.subgraph, self.n, self.node_features, 
                                            self.label_key)
                print("Instability Score: " + str(stability_score))

        
        return

    def explain_results(
                self,
                n: int,
                doctor_type: str = "Doctor_Recruiter"
    ):

            icd = InnerMap.load("ICD9CM")
            icdpr = InnerMap.load("ICD9PROC")
            atc = InnerMap.load("ATC")

            if self.label_key == "medications":
                visit_id = self.subgraph['visit', 'has_received', 'medication'].edge_label_index[:, n][0].item()
                medication_id = self.subgraph['visit', 'has_received', 'medication'].edge_label_index[:, n][1].item()
            elif self.label_key == "diagnosis":
                visit_id = self.subgraph['visit', 'has', 'diagnosis'].edge_label_index[:, n][0].item()
                diagnosis_id = self.subgraph['visit', 'has', 'diagnosis'].edge_label_index[:, n][1].item()

            prompt_recruiter_doctors = """"""
            prompt_internist_doctor = """"""
            medical_scenario = """"""

            atc_list = sorted(self.atc_pre_dict.keys(), key=lambda x: self.atc_pre_dict[x])
            icd9_diag_list = sorted(self.icd9_diag_dict.keys(), key=lambda x: self.icd9_diag_dict[x])
            icd9_proc_list = sorted(self.icd9_proc_dict.keys(), key=lambda x: self.icd9_proc_dict[x])
            icd9_symp_list = sorted(self.icd9_symp_dict.keys(), key=lambda x: self.icd9_symp_dict[x])

            if doctor_type == "Doctor_Recruiter":
                # Create dynamic text
                if self.label_key == "medications":
                    # Prescription decision
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_recruiter_doctors += f"""You are a MEDICAL EXPERT specialised in classifying a specific medical scenario in specific areas of medicine. \n"""
                        prompt_recruiter_doctors += f"""Generate a JSON file that lists a maximum of 5 MOST RELEVANT and COMPETENT doctors/specialists in the administration of the medication:"""
                        prompt_recruiter_doctors += f"""\n"{atc.lookup(atc_list[int(medication_id)])}" at visit {visit_id} and on the patient's condition. \n"""

                    else:
                        prompt_recruiter_doctors += f"""You are a MEDICAL EXPERT specialised in classifying a specific medical scenario in specific areas of medicine. \n"""
                        prompt_recruiter_doctors += f"""Generate a JSON file that lists a maximum of 5 MOST RELEVANT and COMPETENT doctors/specialists in the NON-administration of the medication:"""
                        prompt_recruiter_doctors += f"""\n"{atc.lookup(atc_list[int(medication_id)])}" at visit {visit_id} and on the patient's condition. \n"""

                elif self.label_key == "diagnosis":
                    # Prescription decision
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_recruiter_doctors += f"""You are a MEDICAL EXPERT specialised in classifying a specific medical scenario in specific areas of medicine. \n"""
                        prompt_recruiter_doctors += f"""Generate a JSON file that lists a maximum of 5 MOST RELEVANT and COMPETENT doctors/specialists in the prediction of the diagnosis:"""
                        prompt_recruiter_doctors += f"""\n"{icd.lookup(icd9_diag_list[int(diagnosis_id)])}" at visit {visit_id} and on the patient's condition. \n"""

                    else:
                        prompt_recruiter_doctors += f"""You are a MEDICAL EXPERT specialised in classifying a specific medical scenario in specific areas of medicine. \n"""
                        prompt_recruiter_doctors += f"""Generate a JSON file that lists a maximum of 5 MOST RELEVANT and COMPETENT doctors/specialists in the NON-prediction of the diagnosis:"""
                        prompt_recruiter_doctors += f"""\n"{icd.lookup(icd9_diag_list[int(diagnosis_id)])}" at visit {visit_id} and on the patient's condition. \n"""

            elif doctor_type == "Internist_Doctor":

                if self.label_key == "medications":
                    # Prescription decision
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_internist_doctor += f"""ANALYZE the medical scenario of Visit {visit_id}, in which the MEDICATION RECOMMENDATION system recommended: "{atc.lookup(atc_list[int(medication_id)])}".\n"""
                        prompt_internist_doctor += f"""USE MEDICAL EXPERTISE to EXPLAIN the recommendation and evaluate its CORRECTNESS. \n"""
                        prompt_internist_doctor += f"""Provide guidance on the ALIGNMENT between the patient's condition in the medical scenario and the recommended medication, emphasizing key factors. \n"""
                        prompt_internist_doctor += f"""Ensure clarity and conciseness in the analysis in 100 words."""
                        
                    else:
                        prompt_internist_doctor += f"""ANALYZE the medical scenario of Visit {visit_id}, in which the medication recommendation system did NOT recommend: {atc.lookup(atc_list[int(medication_id)])}.\n"""
                        prompt_internist_doctor += f"""USE MEDICAL EXPERTISE to EXPLAIN the NO-recommendation and evaluate its CORRECTNESS. \n"""
                        prompt_internist_doctor += f"""Provide guidance on the ALIGNMENT between the patient's condition in the medical scenario and the NON-recommended medication, emphasizing key factors. \n"""
                        prompt_internist_doctor += f"""Ensure clarity and conciseness in the analysis in 100 words."""
                        
                elif self.label_key == "diagnosis":
                    # Prescription decision
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_internist_doctor += f"""ANALYZE the medical scenario of Visit {visit_id}, in which the diagnosis recommendation system recommended: {icd.lookup(icd9_diag_list[int(diagnosis_id)])}.\n"""
                        prompt_internist_doctor += f"""USE MEDICAL EXPERTISE to EXPLAIN the recommendation and evaluate its CORRECTNESS. \n"""
                        prompt_internist_doctor += f"""Provide guidance on the ALIGNMENT between the patient's condition in the medical scenario and the recommended diagnosis, emphasizing key factors. \n"""
                        prompt_internist_doctor += f"""Ensure clarity and conciseness in the analysis in 100 words."""
                    else:
                        prompt_internist_doctor += f"""ANALYZE the medical scenario of Visit {visit_id}, in which the diagnosis recommendation system did NOT recommend: {icd.lookup(icd9_diag_list[int(diagnosis_id)])}.\n"""
                        prompt_internist_doctor += f"""USE MEDICAL EXPERTISE to EXPLAIN the NO-recommendation and evaluate its CORRECTNESS. \n"""
                        prompt_internist_doctor += f"""Provide guidance on the ALIGNMENT between the patient's condition in the medical scenario and the NON-recommended diagnosis, emphasizing key factors. \n"""
                        prompt_internist_doctor += f"""Ensure clarity and conciseness in the analysis in 100 words."""

            symptoms = []
            procedures = []
            diagnosis = []
            medications = []

            for edge_type, edge_data in [(edge[0], edge[1]) for edge in self.explanation.edge_items()]:
                for i in range(edge_data['edge_index'].shape[1]):
                    source_id = f"{edge_type[0]}_{edge_data['edge_index'][0, i]}"
                    target_id = f"{edge_type[2]}_{edge_data['edge_index'][1, i]}"

                    edge_mask = self.explanation[edge_type]['edge_mask'][i]

                    if edge_mask > 0:
                        if source_id in self.nodess.keys() and target_id in self.nodess.keys():
                            source, src_id = str(source_id).split('_')
                            target, tgt_id = str(target_id).split('_')

                            if source == "visit" and src_id == str(visit_id):
                                if target == "symptom":
                                    symptoms.append(target_id)
                                elif target == "procedure":
                                    procedures.append(target_id)
                                elif target == "diagnosis":
                                    if self.label_key == "medications":
                                        diagnosis.append(target_id)
                                    elif self.label_key == "diagnosis":
                                        if tgt_id != str(diagnosis_id):
                                            diagnosis.append(target_id)
                                elif target == "medication":
                                    if self.label_key == "medications":
                                        if tgt_id != str(medication_id):
                                            medications.append(target_id)
                                    elif self.label_key == "diagnosis":
                                        medications.append(target_id)

                            elif target == "visit" and tgt_id == str(visit_id):
                                if source == "symptom":
                                    symptoms.append(source_id)
                                elif source == "procedure":
                                    procedures.append(source_id)
                                elif source == "diagnosis":
                                    if self.label_key == "medications":
                                        diagnosis.append(source_id)
                                    elif self.label_key == "diagnosis":
                                        if src_id != str(diagnosis_id):
                                            diagnosis.append(source_id)
                                elif source == "medication":
                                    if self.label_key == "medications":
                                        if src_id != str(medication_id):
                                            medications.append(source_id)
                                    elif self.label_key == "diagnosis":
                                        medications.append(source_id)


            symptoms = sorted(set(symptoms), key=lambda x: self.nodess[x], reverse=True)
            procedures = sorted(set(procedures), key=lambda x: self.nodess[x], reverse=True)
            diagnosis = sorted(set(diagnosis), key=lambda x: self.nodess[x], reverse=True)
            medications = sorted(set(medications), key=lambda x: self.nodess[x], reverse=True)

            prompt_recruiter_doctors += f"\nTHE FOLLOWING MEDICAL SCENARIO of the patient in the visit {visit_id}, in which the importance values of each condition are highlighted, is obtained from the explainability phase of the recommendation system, which aims to provide the conditions that the system has deemed important for recommendation purposes. In particular, the scenario includes:\n\n"
            prompt_internist_doctor += f"\n\nTHE FOLLOWING MEDICAL SCENARIO of the patient in the visit {visit_id}, in which the importance values of each condition are highlighted, is obtained from the explainability phase of the recommendation system, which aims to provide the conditions that the system has deemed important for recommendation purposes. In particular, the scenario includes:\n\n"

            # List of symptoms presented by the patient
            medical_scenario += f"**Symptoms** presented by the patient found to be important from the system (ordered by level of importance):\n\n"


            if symptoms == []:
                medical_scenario += "- No symptoms found\n"

            for symptom in symptoms:
                node_type, id = str(symptom).split('_')
                if node_type == "symptom":
                    symptom_icd = icd.lookup(icd9_symp_list[int(id)])
                    medical_scenario += "- " + symptom_icd + " - Importance level: " + str(round(self.nodess[symptom], 4)) + "\n"


            # List of procedures performed on the patient
            medical_scenario += f"\n**Procedures** performed on the patient results important from the system (ordered by level of importance):\n\n"

            if procedures == []:
                medical_scenario += "- No procedures found\n"

            for procedure in procedures:
                node_type, id = str(procedure).split('_')
                if node_type == "procedure":
                    procedure_icd = icdpr.lookup(icd9_proc_list[int(id)])
                    medical_scenario += "- " + procedure_icd + " - Importance level: " + str(round(self.nodess[procedure], 4)) + "\n"


            # List of patient's diagnoses
            medical_scenario += f"\nPatient **diagnosis** important from the system (ordered by level of importance):\n\n"

            if diagnosis == []:
                medical_scenario += "- No diagnosis found\n"

            for diagnosis in diagnosis:
                node_type, id = str(diagnosis).split('_')
                if node_type == "diagnosis":
                    diagnosis_icd = icd.lookup(icd9_diag_list[int(id)])
                    medical_scenario += "- " + diagnosis_icd + " - Importance level: " + str(round(self.nodess[diagnosis], 4)) + "\n"

            # List of medications administered to the patient
            medical_scenario += f"\n**Medications** already administered to the patient found important from the system (ordered by level of importance):\n\n"

            if medications == []:
                medical_scenario += "- No medications found\n"

            for medication in medications:
                node_type, id = str(medication).split('_')
                if node_type == "medication":
                    medication_atc = atc.lookup(atc_list[int(id)])
                    medical_scenario += "- " + medication_atc + " - Importance level: " + str(round(self.nodess[medication], 4)) + "\n"

            with open(f'{self.root}medical_scenario.txt', 'w') as file:
                file.write(medical_scenario)

            prompt_recruiter_doctors += medical_scenario
            prompt_internist_doctor += medical_scenario

            if self.label_key == "medications":
                if doctor_type == "Doctor_Recruiter":
                    prompt_recruiter_doctors += f"""\nFor each doctor in the JSON file, include:"""
                    prompt_recruiter_doctors += f"""\n- 'role': 'Specify medical speciality'"""
                    prompt_recruiter_doctors += f"""\n- 'description': 'You are a [role identified] with expertise in [describe skills]'"""
                    prompt_recruiter_doctors += f"""\n\nThe structure of the JSON:\n'doctors': [\n\t'role': \n\t'description': \n]"""

                    print(prompt_recruiter_doctors)
                    with open(f'{self.root}prompt_recruiter_doctors.txt', 'w') as file:
                        file.write(prompt_recruiter_doctors)

                    return prompt_recruiter_doctors

                elif doctor_type == "Internist_Doctor":
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_internist_doctor += f"""\nCOMPOSE a summary JUSTIFYING the recommendation of the medication. HIGHLIGHT the positive and negative aspects of administering the medication to the patient taking into account the medical scenario. \n"""
                        prompt_internist_doctor += f"""Clearly articulate the rationale for administering the prescribed medication. \n"""
                        prompt_internist_doctor += f"""In the absence of a discernible correlation between the patient's condition and the recommended medication, explain why the recommendation is not justified. \n"""
                    else:
                        prompt_internist_doctor += f"""\nCompose a summary JUSTIFYING the NOT recommendation of the medication. HIGHLIGHT the positive and negative aspects of administering the medication to the patient taking into account the medical scenario. \n"""
                        prompt_internist_doctor += f"""Clearly articulate the rationale for NOT administering the prescribed medication. \n"""
                        prompt_internist_doctor += f"""In the absence of a discernible correlation between the patient's condition and the recommended medication, explain why the recommendation is justified."""

                    print(prompt_internist_doctor)
                    with open(f'{self.root}prompt_internist_doctor.txt', 'w') as file:
                        file.write(prompt_internist_doctor)

                    return prompt_internist_doctor

            elif self.label_key == "diagnosis":
                if doctor_type == "Doctor_Recruiter":
                    prompt_recruiter_doctors += f"""For each doctor in the JSON file, include: """
                    prompt_recruiter_doctors += f"""\n- 'role': 'Specify medical speciality'"""
                    prompt_recruiter_doctors += f"""\n- 'description': 'You are a [role identified] with expertise in [describe skills]'"""
                    prompt_recruiter_doctors += f"""\n\nThe structure of the JSON:\n'doctors': [\n\t'role': \n\t'description': \n]"""

                    print(prompt_recruiter_doctors)
                    with open(f'{self.root}prompt_recruiter_doctors.txt', 'w') as file:
                        file.write(prompt_recruiter_doctors)

                    return prompt_recruiter_doctors

                elif doctor_type == "Internist_Doctor":
                    if self.explanation['prediction'].numpy() > 0.5:
                        prompt_internist_doctor += f"""\nCOMPOSE a summary JUSTIFYING the recommendation of the diagnosis. HIGHLIGHT the positive and negative aspects of prediction the diagnosis to the patient taking into account the medical scenario. \n"""
                        prompt_internist_doctor += f"""Clearly articulate the rationale for predicting the prescribed diagnosis. \n"""
                        prompt_internist_doctor += f"""In the absence of a discernible correlation between the patient's condition and the prediction diagnosis, explain why the prediction is not justified."""
                    else:
                        prompt_internist_doctor += f"""\nCOMPOSE a summary JUSTIFYING the NOT recommendation of diagnosis. HIGHLIGHT the positive and negative aspects of prediction the diagnosis to the patient taking into account the medical scenario. \n"""
                        prompt_internist_doctor += f"""Clearly articulate the rationale for NOT predicting the prescribed diagnosis. \n"""
                        prompt_internist_doctor += f"""In the absence of a discernible correlation between the patient's condition and the prediction diagnosis, explain why the prediction is justified."""

                    print(prompt_internist_doctor)
                    with open(f'{self.root}prompt_internist_doctor.txt', 'w') as file:
                        file.write(prompt_internist_doctor)

            return prompt_internist_doctor