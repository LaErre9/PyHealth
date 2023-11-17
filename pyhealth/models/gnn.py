import torch
import math
import pkg_resources
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth.models.utils import batch_to_multihot
from pyhealth.datasets import SampleEHRDataset


#### Define a simple GNN model:
class GNN_Conv(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = F.tanh(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_patient: torch.Tensor, x_drug: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_patient = x_patient[edge_label_index[0]]
        edge_feat_drug = x_drug[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_patient * edge_feat_drug).sum(dim=-1)

class GNNLayer(torch.nn.Module):
    """GNN model.

    Our Model.

    This layer is used in the GNN model. But it can also be used as a
    standalone layer.

    Args:
        hidden_channels: hidden feature size.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int,
        **kwargs,
    ):
        super(GNNLayer, self).__init__()

        # Since the dataset does not come with rich features, we also learn four
        # embedding matrices for patients, symptoms, procedures and diseases:
        self.pat_emb = torch.nn.Embedding(data["patient"].num_nodes, hidden_channels)
        self.symp_emb = torch.nn.Embedding(data["symptom"].num_nodes, hidden_channels)
        self.proc_emb = torch.nn.Embedding(data["procedure"].num_nodes, hidden_channels)
        self.dis_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
        self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN_Conv(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def calculate_loss(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates the loss.
        """
        loss = F.binary_cross_entropy_with_logits(pred, y_true)

        return loss

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = {
          "patient": self.pat_emb(data["patient"].node_id),
          "symptom": self.symp_emb(data["symptom"].node_id),
          "procedure": self.proc_emb(data["procedure"].node_id),
          "disease": self.dis_emb(data["disease"].node_id),
          "drug": self.drug_emb(data["drug"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["patient"],
            x_dict["drug"],
            data["patient", "has_received", "drug"].edge_label_index,
        )

        loss = self.calculate_loss(pred, data["patient", "has_received", "drug"].edge_label)

        return loss, pred

class GNN(BaseModel):
    """GNN model.

    Our Model.

    Note:
        This model is only for diagnoses prediction / drug recommendation
        which takes conditions, procedures, symptoms as feature_keys, 
        and drugs as label_key. It only operates on the visit level.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_channels: the hidden channels. Default is 32.
        **kwargs: other parameters for the GNN layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 128,
        hidden_channels: int = 32,
        **kwargs,
    ):
        super(GNN, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures", "symptoms"],
            label_key="drugs",
            mode="multilabel",
        )

        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels

        self.proc_df, self.symp_df, self.drug_df, self.diag_df = self.get_dataframe()

        self.edge_index_patient_to_symptom, self.edge_index_patient_to_disease, \
            self.edge_index_patient_to_procedure, self.edge_index_patient_to_drug = self.get_edge_index()

        self.graph = self.graph_definition()

        self.train_loader = self.get_batches()

        self.layer = GNNLayer(self.train_loader, self.hidden_channels)

    def get_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gets the dataframe of conditions, procedures, symptoms and drugs of patients.

        Returns:
            dataframe: a `pandas.DataFrame` object.
        """
        PROCEDURES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        SYMPTOMS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        DRUGS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ATC_CODE'])
        DIAGNOSES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])


        # Loop attraverso i pazienti e aggiungi le informazioni ai DataFrame
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

            # DRUGS DataFrame
            drugs_data = patient_data['drugs']
            drugs_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(drugs_data),
                                    'HADM_ID': [hadm_id] * len(drugs_data),
                                    'ATC_CODE': drugs_data})
            DRUGS = pd.concat([DRUGS, drugs_df], ignore_index=True)

            # DIAGNOSES DataFrame
            diagnoses_data = patient_data['conditions'][-1]
            diagnoses_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(diagnoses_data),
                                        'HADM_ID': [hadm_id] * len(diagnoses_data),
                                        'SEQ_NUM': range(1, len(diagnoses_data) + 1),
                                        'ICD9_CODE': diagnoses_data})
            DIAGNOSES = pd.concat([DIAGNOSES, diagnoses_df], ignore_index=True)

        return PROCEDURES, SYMPTOMS, DRUGS, DIAGNOSES

    def get_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the graph.

        Returns:
            graph: a `torch_geometric.data.HeteroData` object.
        """

        ### =============== MAPPING SYMPTOMS ===========================
        print('Num. Patients: ' + str(len(self.symp_df['SUBJECT_ID'].unique())))
        print('Num. Symptoms: ' + str(len(self.symp_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_symp_vocab = self.symp_df['ICD9_CODE'].unique()
        # Creazione di un vocabolario unico dai SUBJECT_ID
        subject_vocab = self.symp_df['SUBJECT_ID'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_symp_dict = {code: i for i, code in enumerate(icd9_symp_vocab)}
        # Creazione di un dizionario che mappa il SUBJECT_ID al suo indice nel vocabolario
        subject_dict = {code: i for i, code in enumerate(subject_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(icd9_symp_dict)
        # Sostituzione dei valori nella colonna 'SUBJECT_ID' con i corrispondenti indici nel vocabolario
        self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(subject_dict)

        presents_patient_id = torch.from_numpy(self.symp_df['SUBJECT_ID'].values)
        presents_symptom_id = torch.from_numpy(self.symp_df['ICD9_CODE'].values)

        edge_index_patient_to_symptom = torch.stack([presents_patient_id, presents_symptom_id], dim=0)

        print('Dimension of edge index: ' + str(edge_index_patient_to_symptom.shape))
        print("Final edge indices pointing from patients to symptoms:")
        print("=================================================")
        print(edge_index_patient_to_symptom)
        print("=================================================")

        ### =============== MAPPING DIAGNOSES ===========================
        print('Num. Patients: ' + str(len(self.diag_df['SUBJECT_ID'].unique())))
        print('Num. Diseases: ' + str(len(self.diag_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_diag_vocab = self.diag_df['ICD9_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_diag_dict = {code: i for i, code in enumerate(icd9_diag_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.diag_df['ICD9_CODE_DIAG'] = self.diag_df['ICD9_CODE'].map(icd9_diag_dict)
        # Sostituzione dei valori nella colonna 'SUBJECT_ID' con i corrispondenti indici nel vocabolario
        self.diag_df['SUBJECT_ID'] = self.diag_df['SUBJECT_ID'].map(subject_dict)

        self.diag_df.drop('ICD9_CODE', axis=1, inplace=True)

        hasdisease_patient_id = torch.from_numpy(self.diag_df['SUBJECT_ID'].values)
        hasdisease_disease_id = torch.from_numpy(self.diag_df['ICD9_CODE_DIAG'].values)

        edge_index_patient_to_disease = torch.stack([hasdisease_patient_id, hasdisease_disease_id], dim=0)

        print('Dimension of edge index: ' + str(edge_index_patient_to_disease.shape))
        print("Final edge indices pointing from patients to diseases:")
        print("=================================================")
        print(edge_index_patient_to_disease)
        print("=================================================")

        ### =============== MAPPING PROCEDURES ===========================
        print('Num. Patients: ' + str(len(self.proc_df['SUBJECT_ID'].unique())))
        print('Num. Procedures: ' + str(len(self.proc_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_proc_vocab = self.proc_df['ICD9_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_proc_dict = {code: i for i, code in enumerate(icd9_proc_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.proc_df['ICD9_CODE_PROC'] = self.proc_df['ICD9_CODE'].map(icd9_proc_dict)
        # Sostituzione dei valori nella colonna 'SUBJECT_ID' con i corrispondenti indici nel vocabolario
        self.proc_df['SUBJECT_ID'] = self.proc_df['SUBJECT_ID'].map(subject_dict)

        self.proc_df.drop('ICD9_CODE', axis=1, inplace=True)
        
        hastreat_patient_id = torch.from_numpy(self.proc_df['SUBJECT_ID'].values)
        hastreat_procedure_id = torch.from_numpy(self.proc_df['ICD9_CODE_PROC'].values)

        edge_index_patient_to_procedure = torch.stack([hastreat_patient_id, hastreat_procedure_id], dim=0)

        print('Dimension of edge index: ' + str(edge_index_patient_to_procedure.shape))
        print("Final edge indices pointing from patients to procedures:")
        print("=================================================")
        print(edge_index_patient_to_procedure)
        print("=================================================")

        ### =============== MAPPING DRUGS ===========================
        print('Num. Patients: ' + str(len(self.drug_df['SUBJECT_ID'].unique())))
        print('Num. Drugs: ' + str(len(self.drug_df['ATC_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ATC
        atc_pre_vocab = self.drug_df['ATC_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ATC al suo indice nel vocabolario
        atc_pre_dict = {code: i for i, code in enumerate(atc_pre_vocab)}

        # Sostituzione dei valori nella colonna 'ATC' con i corrispondenti indici nel vocabolario
        self.drug_df['ATC_CODE_PRE'] = self.drug_df['ATC_CODE'].map(atc_pre_dict)
        # Sostituzione dei valori nella colonna 'SUBJECT_ID' con i corrispondenti indici nel vocabolario
        self.drug_df['SUBJECT_ID'] = self.drug_df['SUBJECT_ID'].map(subject_dict)

        self.drug_df.drop('ATC_CODE', axis=1, inplace=True)
        # self.drug_df.drop(['SUBJECT_ID','ATC_CODE_PRE'], axis=1, inplace=True)

        hasreceived_patient_id = torch.from_numpy(self.drug_df['SUBJECT_ID'].values)
        hasreceived_drug_id = torch.from_numpy(self.drug_df['ATC_CODE_PRE'].values)

        edge_index_patient_to_drug = torch.stack([hasreceived_patient_id, hasreceived_drug_id], dim=0)

        print('Dimension of edge index: ' + str(edge_index_patient_to_drug.shape))
        print("Final edge indices pointing from patients to drugs:")
        print("=================================================")
        print(edge_index_patient_to_drug)

        return edge_index_patient_to_symptom, edge_index_patient_to_disease, edge_index_patient_to_procedure, edge_index_patient_to_drug
    
    def graph_definition(self) -> HeteroData:
        # Creazione del grafo
        graph = HeteroData()

        # Save node indices:
        graph["patient"].node_id = torch.arange(len(self.symp_df['SUBJECT_ID'].unique()))
        graph["symptom"].node_id = torch.arange(len(self.symp_df['ICD9_CODE'].unique()))
        graph["disease"].node_id = torch.arange(len(self.diag_df['ICD9_CODE_DIAG'].unique()))
        graph["procedure"].node_id = torch.arange(len(self.proc_df['ICD9_CODE_PROC'].unique()))
        graph["drug"].node_id = torch.arange(len(self.drug_df['ATC_CODE_PRE'].unique()))

        # Add the edge indices:
        graph["patient", "presents", "symptom"].edge_index = self.edge_index_patient_to_symptom
        graph["patient", "has", "disease"].edge_index = self.edge_index_patient_to_disease
        graph["patient", "has_treat", "procedure"].edge_index = self.edge_index_patient_to_procedure
        graph["patient", "has_received", "drug"].edge_index = self.edge_index_patient_to_drug

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        graph = T.ToUndirected()(graph)

        print("=================================================")
        print("Final graph:")
        print(graph)

        return graph

    def get_batches(self) -> HeteroData:
        ### ========== RANDOM LINK SPLIT ==========================
        transform = T.RandomLinkSplit(
            num_val=0.0,
            num_test=0.0,
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=[('patient', 'presents', 'symptom'),('patient', 'has', 'disease'),('patient', 'has_treat', 'procedure'),('patient', 'has_received', 'drug')],
            rev_edge_types=[('symptom', 'rev_presents', 'patient'),('disease', 'rev_has', 'patient'),('procedure', 'rev_has_treat', 'patient'),('drug', 'rev_has_received', 'patient')],
        )

        train_data, val_data, test_data = transform(self.graph)

        ### Per prova stampo
        print("Training data:")
        print("==============")
        print(train_data)

        # ### ========== LINK NEIGHBOR LOADER ==========================
        # # Define seed edges:
        # edge_label_index = train_data["patient", "has_received", "drug"].edge_label_index
        # edge_label = train_data["patient", "has_received", "drug"].edge_label

        # train_loader = LinkNeighborLoader(
        #     data=train_data,
        #     num_neighbors=[20, 10],
        #     neg_sampling_ratio=2.0,
        #     edge_label_index=(("patient", "has_received", "drug"), edge_label_index),
        #     edge_label=edge_label,
        #     batch_size=128,
        #     shuffle=True,
        # )

        # # Inspect a sample:
        # sampled_data = next(iter(train_loader))

        # print("Sampled mini-batch:")
        # print("===================")
        # print(sampled_data)

        return train_data

    def forward(
        self,
        patient_id: List[str],
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        symptoms: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        symp_df = pd.DataFrame({
            'SUBJECT_ID': patient_id,
            'ICD9_CODE': symptoms,
        })
        symp_df = symp_df.explode('ICD9_CODE')
        self.symp_df = symp_df.explode('ICD9_CODE')

        proc_df = pd.DataFrame({
                    'SUBJECT_ID': patient_id,
                    'ICD9_CODE': procedures,
                })
        proc_df = proc_df.explode('ICD9_CODE')
        self.proc_df = proc_df.explode('ICD9_CODE')

        drug_df = pd.DataFrame({
                    'SUBJECT_ID': patient_id,
                    'ATC_CODE': drugs,
                })
        drug_df = drug_df.explode('ATC_CODE')
        self.drug_df = drug_df.explode('ATC_CODE')

        diag_df = pd.DataFrame({
                    'SUBJECT_ID': patient_id,
                    'ICD9_CODE': conditions,
                })
        diag_df = diag_df.explode('ICD9_CODE')
        self.diag_df = diag_df.explode('ICD9_CODE')

        self.edge_index_patient_to_symptom, self.edge_index_patient_to_disease, \
            self.edge_index_patient_to_procedure, self.edge_index_patient_to_drug = self.get_edge_index()

        self.graph = self.graph_definition()

        self.train_loader = self.get_batches()
        
        loss, pred = self.layer(
            self.train_loader
        )

        y_prob = self.prepare_y_prob(pred)
        
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": self.train_loader["patient", "has_received", "drug"].edge_label
        }