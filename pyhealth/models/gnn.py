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
from torch_geometric.utils import coalesce
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv, SAGEConv, FiLMConv, to_hetero

from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset


#### Weights for the loss function to handle unbalanced classes:
def compute_class_weights(y_true):

    zero_count = torch.sum(y_true == 0)
    zero_ratio = (1 / zero_count) * (len(y_true) / 2.0)
    # print("Zero ratio: ", zero_ratio)

    one_count = torch.sum(y_true == 1)
    one_ratio = (1 / one_count) * (len(y_true) / 2.0)
    # print("One ratio: ", one_ratio)

    class_weights = torch.tensor([zero_ratio, one_ratio]).to(y_true.device)

    return class_weights

# Since the dataset does not come with rich features, we also learn four
# embedding matrices for patients, symptoms, procedures and diagnosis:
class X_Dict(torch.nn.Module):
  def __init__(self, k: int, static_kg: List[str], data_enrich: bool, data: HeteroData, embedding_dim: int):
    super().__init__()

    self.k = k
    self.static_kg = static_kg
    self.data_enrich = data_enrich

    self.pat_emb = torch.nn.Embedding(data["patient"].num_nodes, embedding_dim)
    self.vis_emb = torch.nn.Embedding(data["visit"].num_nodes, embedding_dim)
    if self.data_enrich:
        self.symp_emb = torch.nn.Embedding(data["symptom"].num_nodes, embedding_dim)
    self.proc_emb = torch.nn.Embedding(data["procedure"].num_nodes, embedding_dim)
    self.dis_emb = torch.nn.Embedding(data["diagnosis"].num_nodes, embedding_dim)
    self.medication_emb = torch.nn.Embedding(data["medication"].num_nodes, embedding_dim)

    # Embedding of nodes of Static KG
    if self.k == 2:
        for relation in self.static_kg:
            if relation == "ANAT_DIAG":
                self.anat_emb = torch.nn.Embedding(data["anatomy"].num_nodes, embedding_dim)
            if relation == "PC_DRUG":
                self.pharma_emb = torch.nn.Embedding(data["pharmaclass"].num_nodes, embedding_dim)

  def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
    x_dict = {
        'patient': self.pat_emb(batch['patient']['node_id']),
        'visit': self.vis_emb(batch['visit']['node_id']),
        'procedure': self.proc_emb(batch['procedure']['node_id']),
        'diagnosis': self.dis_emb(batch['diagnosis']['node_id']),
        'medication': self.medication_emb(batch['medication']['node_id']),
    }
    if self.data_enrich:
        x_dict['symptom'] = self.symp_emb(batch['symptom']['node_id'])

    if self.k == 2:
        for relation in self.static_kg:
            if relation == "ANAT_DIAG":
                x_dict['anatomy'] = self.anat_emb(batch['anatomy']['node_id'])
            if relation == "PC_DRUG":
                x_dict['pharmaclass'] = self.pharma_emb(batch['pharmaclass']['node_id'])

    return x_dict

#### Define a simple GNN model:
class GNN_Conv(torch.nn.Module):
    def __init__(self, hidden_channels: int, convlayer: str):
        super().__init__()
        self.convlayer = convlayer

        if self.convlayer == "GraphConv":
            self.conv1 = GraphConv(hidden_channels, hidden_channels, aggr="mean", add_self_loops=False)
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr="mean", add_self_loops=False)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr="mean", add_self_loops=False)
            self.dropout = torch.nn.Dropout(p=0.3)
        elif self.convlayer == "SAGEConv":
            self.conv1 = SAGEConv((-1, -1), hidden_channels, normalize=True)
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv2 = SAGEConv((-1, -1), hidden_channels, normalize=True)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv3 = SAGEConv((-1, -1), hidden_channels)
        elif self.convlayer == "FiLMConv":
            self.conv1 = FiLMConv(hidden_channels, hidden_channels)
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv2 = FiLMConv(hidden_channels, hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv3 = FiLMConv(hidden_channels, hidden_channels)
            self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv4 = FiLMConv(hidden_channels, hidden_channels)
            self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.convlayer == "GraphConv":
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = self.dropout(x)
        elif self.convlayer == "SAGEConv":
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
        elif self.convlayer == "FiLMConv":
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
            x = self.dropout(x)

        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_visit: torch.Tensor, x_label: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_visit = x_visit[edge_label_index[0]]
        edge_feat_label = x_label[edge_label_index[1]]

        z = torch.cat([edge_feat_visit, edge_feat_label], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)

        return z.view(-1)

class GNNLayer(torch.nn.Module):
    """GNN Model.

    This layer is used in the GNN model. But it can also be used as a
    standalone layer.

    Args:
        hidden_channels: hidden feature size.
    """
    def __init__(
        self,
        data: HeteroData,
        convlayer: str,
        label_key: str,
        static_kg: List[str],
        k: int,
        hidden_channels: int,
        **kwargs,
    ):
        super(GNNLayer, self).__init__()

        self.convlayer = convlayer
        self.label_key = label_key
        self.static_kg = static_kg
        self.k = k
        self.hidden_channels = hidden_channels

        # Instantiate homogeneous GNN:
        self.gnn = GNN_Conv(self.hidden_channels, self.convlayer)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        # Instantiate edge classifier:
        self.classifier = Classifier(self.hidden_channels)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor], 
                edge_label_index: torch.Tensor) -> torch.Tensor:

        z_dict = self.gnn(x_dict, edge_index_dict)

        if self.label_key == "medications":
            pred = self.classifier(
                z_dict["visit"],
                z_dict["medication"],
                edge_label_index,
            )
        elif self.label_key == "diagnosis":
            pred = self.classifier(
                z_dict["visit"],
                z_dict["diagnosis"],
                edge_label_index,
            )

        return pred

class GNN(BaseModel):
    """GNN Model.

    Note:
        This model is only for diagnosis prediction / medication recommendation
        which takes diagnosis/medications, procedures, symptoms as feature_keys,
        and diagnosis/medications as label_key. It only operates on the visit level.

    Note:
        This model accepts every ATC level as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_channels: the hidden channels. Default is 128.
        **kwargs: other parameters for the GNN layer.
    """
    def __init__(
        self,
        dataset: SampleEHRDataset,
        convlayer: str,
        feature_keys: List[str],
        label_key: str,
        root: str = "static-kg/",
        static_kg: List[str] = ["DIAG_SYMP", "SYMP_DRUG", "DRUG_DIAG", "ANAT_DIAG", "PC_DRUG"],
        k: int = 0,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        **kwargs,
    ):
        super(GNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode="multilabel",
        )

        if k not in range(0, 3):
            raise ValueError("k must be 0, 1 or 2.")

        for relation in static_kg:
            if relation not in ["DIAG_SYMP", "SYMP_DRUG", "DRUG_DIAG", "ANAT_DIAG", "PC_DRUG"]:
                raise ValueError("static_kg must be one of the following: DIAG_SYMP, SYMP_DRUG, DRUG_DIAG, ANAT_DIAG, PC_DRUG.")

        if convlayer not in ["GraphConv", "SAGEConv", "FiLMConv"]:
            raise ValueError("ConvLayer must be one of the following: GraphConv, SAGEConv, FiLMConv.")

        self.data_enrich = False
        for feature in feature_keys:
            if feature == "symptoms":
                self.data_enrich = True

        self.convlayer = convlayer
        self.root = root
        self.static_kg = static_kg
        self.k = k
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.label_tokenizer = self.get_label_tokenizer()

        self.proc_df, self.symp_df, self.medication_df, self.diag_df, self.stat_kg_df = self.get_dataframe()

        self.hadm_dict, self.subject_dict, self.icd9_symp_dict, self.icd9_diag_dict, \
            self.icd9_proc_dict, self.atc_pre_dict = self.mapping_nodes()

        self.edge_index_patient_to_visit, self.edge_index_visit_to_symptom, \
            self.edge_index_visit_to_diagnosis, self.edge_index_visit_to_procedure, \
            self.edge_index_visit_to_medication, self.edge_index_diagnosis_to_symptom, \
            self.edge_index_anatomy_to_diagnosis, self.edge_index_diagnosis_to_medication, \
            self.edge_index_pharma_to_medication, self.edge_index_symptom_to_medication = self.get_edge_index()

        self.graph = self.graph_definition()

        self.x_dict = X_Dict(self.k, self.static_kg, self.data_enrich, self.graph, self.embedding_dim)

        self.layer = GNNLayer(self.graph, self.convlayer, self.label_key, self.static_kg, self.k, 
                              self.hidden_channels)

    def get_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Union[None, Dict[str, pd.DataFrame]]]:
        """Gets the dataframe of diagnosis, procedures, symptoms and medications of patients.

        Returns:
            dataframe: a `pandas.DataFrame` object.
        """
        PROCEDURES = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        SYMPTOMS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        DRUGS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'ATC_CODE'])
        DIAGNOSIS = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])


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

            # SYMPTOMS DataFrame - DATA ENRICHMENT
            symptoms_data = patient_data['symptoms'][-1]
            symptoms_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(symptoms_data),
                                        'HADM_ID': [hadm_id] * len(symptoms_data),
                                        'SEQ_NUM': range(1, len(symptoms_data) + 1),
                                        'ICD9_CODE': symptoms_data})
            SYMPTOMS = pd.concat([SYMPTOMS, symptoms_df], ignore_index=True)

            if self.label_key == "medications":
                medications_data = patient_data['medications']
                diagnosis_data = patient_data['diagnosis'][-1]
            elif self.label_key == "diagnosis":
                medications_data = patient_data['medications'][-1]
                diagnosis_data = patient_data['diagnosis']

            # DRUGS DataFrame
            medications_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(medications_data),
                                    'HADM_ID': [hadm_id] * len(medications_data),
                                    'ATC_CODE': medications_data})
            DRUGS = pd.concat([DRUGS, medications_df], ignore_index=True)

            # DIAGNOSIS DataFrame
            diagnosis_df = pd.DataFrame({'SUBJECT_ID': [subject_id] * len(diagnosis_data),
                                        'HADM_ID': [hadm_id] * len(diagnosis_data),
                                        'SEQ_NUM': range(1, len(diagnosis_data) + 1),
                                        'ICD9_CODE': diagnosis_data})
            DIAGNOSIS = pd.concat([DIAGNOSIS, diagnosis_df], ignore_index=True)

        # ==== GRAPH ENRICHMENT ====
        STATIC_KG = {}

        if self.k > 0:
            for relation in self.static_kg:
                if self.k == 1:
                    if relation in ["ANAT_DIAG", "PC_DRUG"]:
                        continue

                # read table
                STATIC_KG[relation] = pd.read_csv(
                    os.path.join(self.root, f"{relation}.csv"),
                    low_memory=False,
                    index_col=0,
                )

        return PROCEDURES, SYMPTOMS, DRUGS, DIAGNOSIS, STATIC_KG

    def mapping_nodes(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Maps different entities to their corresponding indices in the vocabulary.

        Returns:
            A tuple of dictionaries containing the mappings for HADM_ID, SUBJECT_ID, ICD9_CODE (symptoms),
            ICD9_CODE (diagnosis), ICD9_CODE (procedures), and ATC_CODE (medications).
        """
        # VOCABULARY OF VISITS
        # Create a unique vocabulary from the HADM_ID
        hadm_vocab = self.symp_df['HADM_ID'].unique()
        # Create a dictionary that maps the HADM_ID to its index in the vocabulary
        hadm_dict = {code: i for i, code in enumerate(hadm_vocab)}

        # VOCABULARY OF PATIENTS
        # Create a unique vocabulary from the SUBJECT_ID
        subject_vocab = self.symp_df['SUBJECT_ID'].unique()
        # Create a dictionary that maps the SUBJECT_ID to its index in the vocabulary
        subject_dict = {code: i for i, code in enumerate(subject_vocab)}

        # VOCABULARY OF SYMPTOMS
        # Create a unique vocabulary from the ICD9_CODE
        icd9_symp_vocab = self.symp_df['ICD9_CODE'].unique()
        # Create a dictionary that maps the ICD9_CODE to its index in the vocabulary
        icd9_symp_dict = {code: i for i, code in enumerate(icd9_symp_vocab)}

        # VOCABULARY OF DIAGNOSIS
        # Create a unique vocabulary from the ICD9_CODE
        icd9_diag_vocab = self.diag_df['ICD9_CODE'].unique()
        # Create a dictionary that maps the ICD9_CODE to its index in the vocabulary
        if self.label_key == "diagnosis":
            icd9_diag_dict = {code: self.label_tokenizer.vocabulary(code) for code in icd9_diag_vocab}
        else:
            icd9_diag_dict = {code: i for i, code in enumerate(icd9_diag_vocab)}

        # VOCABULARY OF PROCEDURES
        # Create a unique vocabulary from the ICD9_CODE
        icd9_proc_vocab = self.proc_df['ICD9_CODE'].unique()
        # Create a dictionary that maps the ICD9_CODE to its index in the vocabulary
        icd9_proc_dict = {code: i for i, code in enumerate(icd9_proc_vocab)}

        # VOCABULARY OF DRUGS
        # Create a unique vocabulary from the ATC_CODE
        atc_pre_vocab = self.medication_df['ATC_CODE'].unique()
        # Create a dictionary that maps the ATC_CODE to its index in the vocabulary
        if self.label_key == "medications":
            atc_pre_dict = {code: self.label_tokenizer.vocabulary(code) for code in atc_pre_vocab}
        else:
            atc_pre_dict = {code: i for i, code in enumerate(atc_pre_vocab)}

        return hadm_dict, subject_dict, icd9_symp_dict, icd9_diag_dict, icd9_proc_dict, atc_pre_dict

    def get_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the edge indices for the graph.

        Returns:
            A tuple of torch.Tensor containing the edge indices for different relationships in the graph:
            - edge_index_patient_to_visit: Edge indices pointing from patients to visits.
            - edge_index_visit_to_symptom: Edge indices pointing from visits to symptoms.
            - edge_index_visit_to_diagnosis: Edge indices pointing from visits to diagnosis.
            - edge_index_visit_to_procedure: Edge indices pointing from visits to procedures.
            - edge_index_visit_to_medication: Edge indices pointing from visits to medications.
            - edge_index_diagnosis_to_symptom: Edge indices pointing from diagnosis to symptoms.
            - edge_index_anatomy_to_diagnosis: Edge indices pointing from anatomy to diagnosis.
            - edge_index_diagnosis_to_medication: Edge indices pointing from diagnosis to medications.
            - edge_index_pharma_to_medication: Edge indices pointing from pharma to medications.
            - edge_index_symptom_to_medication: Edge indices pointing from symptoms to medications.
        """
        # =============== MAPPING VISITS ===========================
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.symp_df['HADM_ID'] = self.symp_df['HADM_ID'].map(self.hadm_dict)
        
        # Substituting the values in the 'SUBJECT_ID' column with the corresponding indices in the vocabulary
        self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(self.subject_dict)

        has_patient_id = torch.from_numpy(self.symp_df['SUBJECT_ID'].values)
        has_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)

        # Create the edge index for the relationship 'has' between patients and visits
        edge_index_patient_to_visit = torch.stack([has_patient_id, has_visit_id], dim=0)

        # Remove duplicate patient_id and visit_id pairs
        edge_index_patient_to_visit = coalesce(edge_index_patient_to_visit)

        # =============== MAPPING SYMPTOMS ===========================
        # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
        if self.data_enrich:
            self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)

            presents_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)
            presents_symptom_id = torch.from_numpy(self.symp_df['ICD9_CODE'].values)

            # Create the edge index for the relationship 'presents' between visits and symptoms
            edge_index_visit_to_symptom = torch.stack([presents_visit_id, presents_symptom_id], dim=0)

            edge_index_visit_to_symptom = coalesce(edge_index_visit_to_symptom)
        else:
            edge_index_visit_to_symptom = torch.empty((2, 0), dtype=torch.int64)

        # =============== MAPPING DIAGNOSIS ===========================
        # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
        self.diag_df['ICD9_CODE_DIAG'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.diag_df['HADM_ID'] = self.diag_df['HADM_ID'].map(self.hadm_dict)

        # Drop the 'ICD9_CODE' column that is no longer needed
        self.diag_df.drop('ICD9_CODE', axis=1, inplace=True)

        hasdiagnosis_visit_id = torch.from_numpy(self.diag_df['HADM_ID'].values)
        hasdiagnosis_diagnosis_id = torch.from_numpy(self.diag_df['ICD9_CODE_DIAG'].values)

        # Create the edge index for the relationship 'has' between visits and diagnosis
        edge_index_visit_to_diagnosis = torch.stack([hasdiagnosis_visit_id, hasdiagnosis_diagnosis_id], dim=0)

        edge_index_visit_to_diagnosis = coalesce(edge_index_visit_to_diagnosis)

        # =============== MAPPING PROCEDURES ===========================
        # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
        self.proc_df['ICD9_CODE_PROC'] = self.proc_df['ICD9_CODE'].map(self.icd9_proc_dict)
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.proc_df['HADM_ID'] = self.proc_df['HADM_ID'].map(self.hadm_dict)

        # Drop the 'ICD9_CODE' column that is no longer needed
        self.proc_df.drop('ICD9_CODE', axis=1, inplace=True)

        hastreat_visit_id = torch.from_numpy(self.proc_df['HADM_ID'].values)
        hastreat_procedure_id = torch.from_numpy(self.proc_df['ICD9_CODE_PROC'].values)

        # Create the edge index for the relationship 'has_treat' between visits and procedures
        edge_index_visit_to_procedure = torch.stack([hastreat_visit_id, hastreat_procedure_id], dim=0)

        edge_index_visit_to_procedure = coalesce(edge_index_visit_to_procedure)

        # =============== MAPPING DRUGS ===========================
        # Substituting the values in the 'ATC_CODE' column with the corresponding indices in the vocabulary
        self.medication_df['ATC_CODE_PRE'] = self.medication_df['ATC_CODE'].map(self.atc_pre_dict)
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.medication_df['HADM_ID'] = self.medication_df['HADM_ID'].map(self.hadm_dict)

        # Drop the 'ATC_CODE' column that is no longer needed
        self.medication_df.drop('ATC_CODE', axis=1, inplace=True)

        hasreceived_visit_id = torch.from_numpy(self.medication_df['HADM_ID'].values)
        hasreceived_medication_id = torch.from_numpy(self.medication_df['ATC_CODE_PRE'].values)

        # Create the edge index for the relationship 'has_received' between visits and medications
        edge_index_visit_to_medication = torch.stack([hasreceived_visit_id, hasreceived_medication_id], dim=0)

        edge_index_visit_to_medication = coalesce(edge_index_visit_to_medication)

        # ==== GRAPH ENRICHMENT ====
        edge_index_diagnosis_to_symptom = None
        edge_index_anatomy_to_diagnosis = None
        edge_index_diagnosis_to_medication = None
        edge_index_pharma_to_medication = None
        edge_index_symptom_to_medication = None

        if self.k > 0:
            for relation in self.static_kg:
                if relation == "DIAG_SYMP" and self.data_enrich:
                    # =============== MAPPING DIAG_SYMP ===========================
                    # Copy the dataframe with the relationship DIAG_SYMP
                    diag_symp_df = self.stat_kg_df[relation].astype(str).copy()
                    diag_symp_df = diag_symp_df[diag_symp_df["DIAG"].isin(self.icd9_diag_dict.keys())]

                    # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
                    diag_symp_df['DIAG'] = diag_symp_df['DIAG'].map(self.icd9_diag_dict)

                    # Lookup the indices of the symptoms in the vocabulary
                    last_index = max(self.icd9_symp_dict.values())

                    # Add the new symptoms to the dictionary with consecutive indices
                    for symptom_code in diag_symp_df['SYMP'].unique():
                        if symptom_code not in self.icd9_symp_dict:
                            last_index += 1
                            self.icd9_symp_dict[symptom_code] = last_index
                            self.symp_df = pd.concat([self.symp_df, pd.DataFrame({'SUBJECT_ID': [0], 'HADM_ID': [0], 'SEQ_NUM': [0], 'ICD9_CODE': [symptom_code]})], ignore_index=True)
                    diag_symp_df['SYMP'] = diag_symp_df['SYMP'].map(self.icd9_symp_dict)

                    if not diag_symp_df.empty:
                        hasbeencaused_diag_id = torch.from_numpy(diag_symp_df['DIAG'].values)
                        hasbeencaused_symp_id = torch.from_numpy(diag_symp_df['SYMP'].values)
                        edge_index_diagnosis_to_symptom = torch.stack([hasbeencaused_diag_id, hasbeencaused_symp_id], dim=0)
                    else:
                        # Initialize edge_index_diagnosis_to_symptom as empty if the DataFrame is empty
                        edge_index_diagnosis_to_symptom = torch.empty((2, 0), dtype=torch.int64)

                elif (relation == "ANAT_DIAG") and self.k == 2:
                    # =============== MAPPING ANAT_DIAG ===========================
                    # Copy the dataframe with the relationship ANAT_DIAG
                    anat_diag_df = self.stat_kg_df[relation].astype(str).copy()
                    anat_diag_df = anat_diag_df[anat_diag_df["DIAG"].isin(self.icd9_diag_dict.keys())]

                    # Create a unique vocabulary from the codici UBERON
                    uberon_anat_vocab = anat_diag_df['ANAT'].unique()
                    # Create a dictionary that maps the codici UBERON to their index in the vocabulary
                    uberon_anat_dict = {code: i for i, code in enumerate(uberon_anat_vocab)}

                    # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
                    anat_diag_df['DIAG'] = anat_diag_df['DIAG'].map(self.icd9_diag_dict)
                    # Substituting the values in the 'ANAT' column with the corresponding indices in the vocabulary
                    anat_diag_df['ANAT'] = anat_diag_df['ANAT'].map(uberon_anat_dict)

                    if not anat_diag_df.empty:
                        localizes_diag_id = torch.from_numpy(anat_diag_df['DIAG'].values)
                        localizes_anat_id = torch.from_numpy(anat_diag_df['ANAT'].values)
                        edge_index_anatomy_to_diagnosis = torch.stack([localizes_diag_id, localizes_anat_id], dim=0)
                    else:
                        # Initialize edge_index_anatomy_to_diagnosis as empty if the DataFrame is empty
                        edge_index_anatomy_to_diagnosis = torch.empty((2, 0), dtype=torch.int64)

                elif relation == "DRUG_DIAG":
                    # =============== MAPPING DRUG_DIAG ===========================
                    # Copy the dataframe with the relationship DRUG_DIAG
                    medication_diag_df = self.stat_kg_df[relation].astype(str).copy()
                    medication_diag_df = medication_diag_df[medication_diag_df["DIAG"].isin(self.icd9_diag_dict.keys())]
                    medication_diag_df = medication_diag_df[medication_diag_df["DRUG"].isin(self.atc_pre_dict.keys())]

                    # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
                    medication_diag_df['DIAG'] = medication_diag_df['DIAG'].map(self.icd9_diag_dict)
                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    medication_diag_df['DRUG'] = medication_diag_df['DRUG'].map(self.atc_pre_dict)

                    if not medication_diag_df.empty:
                        treats_diag_id = torch.from_numpy(medication_diag_df['DIAG'].values)
                        treats_medication_id = torch.from_numpy(medication_diag_df['DRUG'].values)
                        edge_index_diagnosis_to_medication = torch.stack([treats_diag_id, treats_medication_id], dim=0)
                    else:
                        # Initialize edge_index_diagnosis_to_medication as empty if the DataFrame is empty
                        edge_index_diagnosis_to_medication = torch.empty((2, 0), dtype=torch.int64)

                elif (relation == "PC_DRUG") and (self.k == 2):
                    # =============== MAPPING PC_DRUG ===========================
                    # Copy the dataframe with the relationship PC_DRUG
                    pc_medication_df = self.stat_kg_df[relation].astype(str).copy()
                    pc_medication_df = pc_medication_df[pc_medication_df["DRUG"].isin(self.atc_pre_dict.keys())]

                    # Create a unique vocabulary from the codici PHARMACLASS
                    ndc_pc_vocab = pc_medication_df['PHARMACLASS'].unique()
                    # Create a dictionary that maps the codici PHARMACLASS to their index in the vocabulary
                    ndc_pc_dict = {code: i for i, code in enumerate(ndc_pc_vocab)}

                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    pc_medication_df['DRUG'] = pc_medication_df['DRUG'].map(self.atc_pre_dict)
                    # Substituting the values in the 'PHARMACLASS' column with the corresponding indices in the vocabulary
                    pc_medication_df['PHARMACLASS'] = pc_medication_df['PHARMACLASS'].map(ndc_pc_dict)

                    if not pc_medication_df.empty:
                        includes_pharma_id = torch.from_numpy(pc_medication_df['PHARMACLASS'].values)
                        includes_medication_id = torch.from_numpy(pc_medication_df['DRUG'].values)
                        edge_index_pharma_to_medication = torch.stack([includes_pharma_id, includes_medication_id], dim=0)
                    else:
                        # Initialize edge_index_pharma_to_medication as empty if the DataFrame is empty
                        edge_index_pharma_to_medication = torch.empty((2, 0), dtype=torch.int64)

                elif relation == "SYMP_DRUG" and self.data_enrich:
                    # =============== MAPPING SYMP_DRUG ===========================
                    # Copy the dataframe with the relationship SYMP_DRUG
                    symp_medication_df = self.stat_kg_df[relation].astype(str).copy()
                    symp_medication_df = symp_medication_df[symp_medication_df["DRUG"].isin(self.atc_pre_dict.keys())] ###OCCHIO QUI

                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    symp_medication_df['DRUG'] = symp_medication_df['DRUG'].map(self.atc_pre_dict)

                    # Lookup the indices of the symptoms in the vocabulary
                    last_index = max(self.icd9_symp_dict.values())

                    # Add the new symptoms to the dictionary with consecutive indices
                    for symptom_code in symp_medication_df['SYMP'].unique():
                        if symptom_code not in self.icd9_symp_dict:
                            last_index += 1
                            self.icd9_symp_dict[symptom_code] = last_index
                            self.symp_df = pd.concat([self.symp_df, pd.DataFrame({'SUBJECT_ID': [0], 'HADM_ID': [0], 'SEQ_NUM': [0], 'ICD9_CODE': [symptom_code]})], ignore_index=True)
                    symp_medication_df['SYMP'] = symp_medication_df['SYMP'].map(self.icd9_symp_dict)

                    if not symp_medication_df.empty:
                        causes_symp_id = torch.from_numpy(symp_medication_df['SYMP'].values)
                        causes_medication_id = torch.from_numpy(symp_medication_df['DRUG'].values)
                        edge_index_symptom_to_medication = torch.stack([causes_symp_id, causes_medication_id], dim=0)
                    else:
                        # Initialize edge_index_symptom_to_medication as empty if the DataFrame is empty
                        edge_index_symptom_to_medication = torch.empty((2, 0), dtype=torch.int64)

        return edge_index_patient_to_visit, edge_index_visit_to_symptom, edge_index_visit_to_diagnosis, \
                edge_index_visit_to_procedure, edge_index_visit_to_medication, edge_index_diagnosis_to_symptom, \
                edge_index_anatomy_to_diagnosis, edge_index_diagnosis_to_medication, edge_index_pharma_to_medication, \
                edge_index_symptom_to_medication

    def graph_definition(self) -> HeteroData:
        """
        Defines the graph structure for the GNN model.

        Returns:
            HeteroData: The graph structure with node and edge indices.
        """
        # Graph definition:
        graph = HeteroData()

        # Save node indices:
        graph["patient"].node_id = torch.arange(len(self.symp_df['SUBJECT_ID'].unique()))
        graph["visit"].node_id = torch.arange(len(self.symp_df['HADM_ID'].unique()))
        if self.data_enrich:
            graph["symptom"].node_id = torch.arange(len(self.symp_df['ICD9_CODE'].unique()))
        graph["procedure"].node_id = torch.arange(len(self.proc_df['ICD9_CODE_PROC'].unique()))

        # Nodes of Static KG
        if self.k == 2:
            for relation in self.static_kg:
                if relation == "ANAT_DIAG":
                    graph["anatomy"].node_id = torch.arange(len(self.stat_kg_df[relation]['ANAT'].unique()))
                if relation == "PC_DRUG":
                    graph["pharmaclass"].node_id = torch.arange(len(self.stat_kg_df[relation]['PHARMACLASS'].unique()))

        if self.label_key == "diagnosis":
            graph["diagnosis"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())
            graph["medication"].node_id = torch.arange(len(self.medication_df['ATC_CODE_PRE'].unique()))
        else:
            graph["diagnosis"].node_id = torch.arange(len(self.diag_df['ICD9_CODE_DIAG'].unique()))
            graph["medication"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())

        # Add the edge indices:
        graph["patient", "has", "visit"].edge_index = self.edge_index_patient_to_visit
        if self.data_enrich:
            graph["visit", "presents", "symptom"].edge_index = self.edge_index_visit_to_symptom
        graph["visit", "has", "diagnosis"].edge_index = self.edge_index_visit_to_diagnosis
        graph["visit", "has_treat", "procedure"].edge_index = self.edge_index_visit_to_procedure
        graph["visit", "has_received", "medication"].edge_index = self.edge_index_visit_to_medication

        # Edges of Static KG
        if self.k > 0:
            for relation in self.static_kg:
                if relation == "DIAG_SYMP" and self.data_enrich:
                    graph["diagnosis", "has_been_caused_by", "symptom"].edge_index = self.edge_index_diagnosis_to_symptom
                if (relation == "ANAT_DIAG") and (self.k == 2):
                    graph["diagnosis", "localizes", "anatomy"].edge_index = self.edge_index_anatomy_to_diagnosis
                if relation == "DRUG_DIAG":
                    graph["diagnosis", "treats", "medication"].edge_index = self.edge_index_diagnosis_to_medication
                if (relation == "PC_DRUG") and (self.k == 2):
                    graph["pharmaclass", "includes", "medication"].edge_index = self.edge_index_pharma_to_medication
                if relation == "SYMP_DRUG" and self.data_enrich:
                    graph["symptom", "causes", "medication"].edge_index = self.edge_index_symptom_to_medication


        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        graph = T.ToUndirected()(graph)

        return graph

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

    def convert_batches(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts the data into separate dataframes for procedures, symptoms, medications, and diagnosis.

        Returns:
            A tuple of four pandas DataFrames representing the converted data:
            - proc_df: DataFrame containing procedure data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
            - symp_df: DataFrame containing symptom data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
            - medication_df: DataFrame containing medication data with columns 'SUBJECT_ID', 'HADM_ID', and 'ATC_CODE'.
            - diag_df: DataFrame containing diagnosis data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
        """
        # ==== DATA CONVERSION ====
        # SYMPTOMS DataFrame
        symp_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.symptoms,
        })
        symp_df = symp_df.explode('ICD9_CODE')
        symp_df = symp_df.explode('ICD9_CODE')

        # PROCEDURES DataFrame
        proc_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.procedures,
        })
        proc_df = proc_df.explode('ICD9_CODE')
        proc_df = proc_df.explode('ICD9_CODE')

        # DRUGS DataFrame
        medication_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ATC_CODE': self.medications,
        })
        medication_df = medication_df.explode('ATC_CODE')
        medication_df = medication_df.explode('ATC_CODE')

        # CONDITIONS DataFrame
        diag_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.diagnosis,
        })
        diag_df = diag_df.explode('ICD9_CODE')
        diag_df = diag_df.explode('ICD9_CODE')

        return proc_df, symp_df, medication_df, diag_df

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
 
        if self.data_enrich:
            self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)
            symptom = self.symp_df["ICD9_CODE"].unique()
            select_symptom = torch.from_numpy(symptom)
 
        if self.label_key == "medications":
            self.diag_df['ICD9_CODE'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
            diagnosis = self.diag_df["ICD9_CODE"].unique()
            select_diagnosis = torch.from_numpy(diagnosis)
        else:
            self.medication_df['ATC_CODE'] = self.medication_df['ATC_CODE'].map(self.atc_pre_dict)
            medication = self.medication_df["ATC_CODE"].unique()
            select_medication = torch.from_numpy(medication)
 
        self.proc_df['ICD9_CODE'] = self.proc_df['ICD9_CODE'].map(self.icd9_proc_dict)
        procedure = self.proc_df["ICD9_CODE"].unique()
        select_procedure = torch.from_numpy(procedure)
 
        if self.label_key == "medications":
            if self.data_enrich:
                subgraph = self.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "diagnosis": select_diagnosis})
            else:
                subgraph = self.graph.subgraph({"patient": select_patient, "visit": select_visit, "procedure": select_procedure, "diagnosis": select_diagnosis})
        elif self.label_key == "diagnosis":
            if self.data_enrich:
                subgraph = self.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "medication": select_medication})
            else:
                subgraph = self.graph.subgraph({"patient": select_patient, "visit": select_visit, "procedure": select_procedure, "medication": select_medication})

        return subgraph

    def create_y_prob_mat(self) -> torch.Tensor:
        """
        Create a probability matrix based on the given label key.

        Returns:
            torch.Tensor: The probability matrix.
        """
        if self.label_key == "medications":
            edge_label_full = self.subgraph["visit", "has_received", "medication"].edge_label_index
        elif self.label_key == "diagnosis":
            edge_label_full = self.subgraph["visit", "has", "diagnosis"].edge_label_index

        # Get the probability values from the model
        prob_full = self.y_prob.detach()

        # Get the unique indices and labels
        unique_visits, indices = torch.unique(edge_label_full[0], return_inverse=True)
        unique_labels, label_indices = torch.unique(edge_label_full[1], return_inverse=True)

        # Sort the indices and labels
        combined_indices = indices * len(unique_labels) + label_indices
        sorted_combined_indices = torch.argsort(combined_indices)

        # Get the unique sorted indices and labels
        unique_combined_indices, unique_indices = torch.unique(sorted_combined_indices, return_inverse=True)

        # Calculate the sorted visits and labels
        sorted_visits = unique_combined_indices // len(unique_labels)
        sorted_labels = unique_combined_indices % len(unique_labels)

        # Create a tensor of zeros with the correct shape
        y_prob_mat = torch.zeros(len(unique_visits), len(unique_labels), device=self.device)

        # Indexing directly into the tensor to fill values
        y_prob_mat[sorted_visits, sorted_labels] = prob_full[sorted_combined_indices]

        return y_prob_mat

    def visualize_graph(self, patients_id: List[int] = 0) -> None:
        """
        ##DEPRECATED##
        Visualizes the graph by drawing its nodes and edges using networkx and matplotlib.

        This method selects a specific patient and visit from the graph, removes isolated nodes,
        converts the subgraph to a homogeneous graph, and then uses networkx to draw the graph.
        Each node is assigned a color based on its node type.

        Args:
            patients_id (List[int]): The list of patient IDs to visualize.

        Returns:
            None
        """
        # Select a specific patient and visit
        select_patient = torch.tensor(patients_id)

        # Retrieve the subgraph
        data_view = self.graph.subgraph({"patient": select_patient})
        data_view = T.RemoveIsolatedNodes()(data_view)

        # Mapping between node_type values and labels
        node_type_mapping = {
            0: 'patient',
            1: 'visit',
            2: 'symptom',
            3: 'diagnosis',
            4: 'procedure',
            5: 'medication'
        }

        # Convert to homogeneous
        data_homogeneous = data_view.to_homogeneous()
        g = to_networkx(data_homogeneous)

        # Use node types as color map
        colour_map = data_homogeneous.node_type
        pos = nx.spring_layout(g)

        # Separate nodes by node type and add some randomness to separate the nodes
        for i in range(0, len(colour_map)):
            if colour_map[i] != 0:
                pos[i][0] += np.cos(colour_map[i] / 2) * 10 + random.randint(-1, 1)
                pos[i][1] += np.sin(colour_map[i] / 2) * 10 + random.randint(-1, 1)
            else:
                pos[i][0] += random.randint(-3, 3)
                pos[i][1] += random.randint(-3, 3)

        # Draw the graph
        nx.draw_networkx(g, pos=pos, node_color=colour_map * 40, cmap=plt.cm.tab20)

        # Add custom legend
        legend_labels = {v: k for k, v in node_type_mapping.items()}
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(legend_labels[label] * 4), markersize=10, label=label) for label in node_type_mapping.values()]

        plt.legend(handles=legend_handles, loc="lower right")
        plt.show()

    def calculate_loss(self, pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates the loss."""
        # Calculate class weights
        class_weights = compute_class_weights(y_true)
        
        # Create a tensor of weights for each datapoint
        sample_weights = class_weights[y_true.long()]
 
        # Calculate the loss for each prediction
        loss = F.binary_cross_entropy_with_logits(pred, y_true, weight=sample_weights)
        # Calculate the loss for each prediction -  Version with mask
        # loss = F.binary_cross_entropy_with_logits(pred[mask], y_true[mask])

        return loss

    def forward(
        self,
        patient_id: List[str],
        visit_id: List[str],
        diagnosis: List[List[List[str]]],
        procedures: List[List[List[str]]],
        symptoms: List[List[List[str]]],
        medications: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            patient_id (List[str]): List of patient IDs.
            visit_id (List[str]): List of visit IDs.
            diagnosis (List[List[List[str]]]): List of diagnosis.
            procedures (List[List[List[str]]]): List of procedures.
            symptoms (List[List[List[str]]]): List of symptoms.
            medications (List[List[str]]): List of medications.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss, predicted probabilities, and true labels.
        """
        # Set the model to training mode
        self.patient_id = patient_id
        self.visit_id = visit_id
        self.diagnosis = diagnosis
        self.procedures = procedures
        self.symptoms = symptoms
        self.medications = medications

        # Prepare the labels
        if self.label_key == "medications":
            y_true = self.prepare_labels(self.medications, self.label_tokenizer)
        elif self.label_key == "diagnosis":
            y_true = self.prepare_labels(self.diagnosis, self.label_tokenizer)

        # Convert the data into separate dataframes for procedures, symptoms, medications, and diagnosis
        self.proc_df, self.symp_df, self.medication_df, self.diag_df = self.convert_batches()
        
        # Get the subgraph
        self.subgraph = self.get_subgraph()
        self.subgraph = self.generate_neg_samples()
        self.mask = self.generate_mask()

        self.subgraph = self.subgraph.to(device=self.device)

        self.node_features = self.x_dict(self.subgraph)

        # Get the loss and predicted probabilities
        if self.label_key == "medications":
            pred = self.layer(self.node_features, self.subgraph.edge_index_dict, 
                                self.subgraph['visit', 'medication'].edge_label_index)
            loss = self.calculate_loss(pred, self.subgraph['visit', 'medication'].edge_label, self.mask)
        elif self.label_key == "diagnosis":
            pred = self.layer(self.node_features, self.subgraph.edge_index_dict, 
                                self.subgraph['visit', 'diagnosis'].edge_label_index)
            loss = self.calculate_loss(pred, self.subgraph['visit', 'diagnosis'].edge_label, self.mask)

        # Prepare the predicted probabilities applying the sigmoid function
        self.y_prob = self.prepare_y_prob(pred)
        # Create the probability matrix
        y_prob_mat = self.create_y_prob_mat()

        loss = loss.to(device=self.device)
        y_prob_mat = y_prob_mat.to(device=self.device)
        y_true = y_true.to(device=self.device)

        return {
            "loss": loss,
            "y_prob": y_prob_mat,
            "y_true": y_true
        }