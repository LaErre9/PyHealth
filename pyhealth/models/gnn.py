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


#### Weights for the loss function to handle unbalanced classes:
def compute_class_weights(y_true):
    class_counts = torch.bincount(y_true.long())
    total_samples = y_true.size(0)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights

# Since the dataset does not come with rich features, we also learn four
# embedding matrices for patients, symptoms, procedures and diseases:
class X_Dict(torch.nn.Module):
  def __init__(self, k: int, static_kg: List[str], data: HeteroData, embedding_dim):
    super().__init__()

    self.k = k
    self.static_kg = static_kg

    self.pat_emb = torch.nn.Embedding(data["patient"].num_nodes, embedding_dim)
    self.vis_emb = torch.nn.Embedding(data["visit"].num_nodes, embedding_dim)
    self.symp_emb = torch.nn.Embedding(data["symptom"].num_nodes, embedding_dim)
    self.proc_emb = torch.nn.Embedding(data["procedure"].num_nodes, embedding_dim)
    self.dis_emb = torch.nn.Embedding(data["disease"].num_nodes, embedding_dim)
    self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, embedding_dim)

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
        'symptom': self.symp_emb(batch['symptom']['node_id']),
        'procedure': self.proc_emb(batch['procedure']['node_id']),
        'disease': self.dis_emb(batch['disease']['node_id']),
        'drug': self.drug_emb(batch['drug']['node_id']),
    }

    if self.k == 2:
        for relation in self.static_kg:
            if relation == "ANAT_DIAG":
                x_dict['anatomy'] = self.anat_emb(batch['anatomy']['node_id'])
            if relation == "PC_DRUG":
                x_dict['pharmaclass'] = self.pharma_emb(batch['pharmaclass']['node_id'])

    return x_dict

#### Define a simple GNN model:
class GNN_Conv(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.4):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv4 = SAGEConv((-1, -1), hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.dropout(x)

        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels):
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
        label_key: str,
        static_kg: List[str],
        k: int,
        hidden_channels: int,
        embedding_dim: int,
        **kwargs,
    ):
        super(GNNLayer, self).__init__()

        self.label_key = label_key
        self.static_kg = static_kg
        self.k = k

        # Instantiate homogeneous GNN:
        self.gnn = GNN_Conv(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier(hidden_channels)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor], 
                edge_label_index: torch.Tensor, edge_label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        z_dict = self.gnn(x_dict, edge_index_dict)

        if self.label_key == "drugs":
            pred = self.classifier(
                z_dict["visit"],
                z_dict["drug"],
                edge_label_index,
            )
        else:
            pred = self.classifier(
                z_dict["visit"],
                z_dict["disease"],
                edge_label_index,
            )

        return F.sigmoid(pred)

class GNN(BaseModel):
    """GNN Model.

    Note:
        This model is only for diagnoses prediction / drug recommendation
        which takes conditions, procedures, symptoms as feature_keys,
        and drugs as label_key. It only operates on the visit level.

    Note:
        This model accepts every ATC level as medication codes.

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
            raise ValueError("k must be 0, 1 or 2. By default, will be set k = 0.")

        for relation in static_kg:
            if relation not in ["DIAG_SYMP", "SYMP_DRUG", "DRUG_DIAG", "ANAT_DIAG", "PC_DRUG"]:
                raise ValueError("static_kg must be one of the following: DIAG_SYMP, SYMP_DRUG, DRUG_DIAG, ANAT_DIAG, PC_DRUG. By default, will be set static_kg = ['DIAG_SYMP', 'SYMP_DRUG', 'DRUG_DIAG', 'ANAT_DIAG', 'PC_DRUG'].")

        self.root = root
        self.static_kg = static_kg
        self.k = k
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.label_tokenizer = self.get_label_tokenizer()

        self.proc_df, self.symp_df, self.drug_df, self.diag_df, self.stat_kg_df = self.get_dataframe()

        self.hadm_dict, self.subject_dict, self.icd9_symp_dict, self.icd9_diag_dict, \
            self.icd9_proc_dict, self.atc_pre_dict = self.mapping_nodes()

        self.edge_index_patient_to_visit, self.edge_index_visit_to_symptom, \
            self.edge_index_visit_to_disease, self.edge_index_visit_to_procedure, \
            self.edge_index_visit_to_drug, self.edge_index_disease_to_symptom, \
            self.edge_index_anatomy_to_diagnosis, self.edge_index_diagnosis_to_drug, \
            self.edge_index_pharma_to_drug, self.edge_index_symptom_to_drug = self.get_edge_index()

        self.graph = self.graph_definition()

        self.x_dict = X_Dict(self.k, self.static_kg, self.graph, self.embedding_dim)

        self.layer = GNNLayer(self.graph, self.label_key, self.static_kg, self.k, self.hidden_channels, self.embedding_dim)

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

    def mapping_nodes(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Maps different entities to their corresponding indices in the vocabulary.

        Returns:
            A tuple of dictionaries containing the mappings for HADM_ID, SUBJECT_ID, ICD9_CODE (symptoms),
            ICD9_CODE (diagnoses), ICD9_CODE (procedures), and ATC_CODE (drugs).
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

        # VOCABULARY OF DIAGNOSES
        # Create a unique vocabulary from the ICD9_CODE
        icd9_diag_vocab = self.diag_df['ICD9_CODE'].unique()
        # Create a dictionary that maps the ICD9_CODE to its index in the vocabulary
        if self.label_key == "conditions":
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
        atc_pre_vocab = self.drug_df['ATC_CODE'].unique()
        # Create a dictionary that maps the ATC_CODE to its index in the vocabulary
        if self.label_key == "drugs":
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
            - edge_index_visit_to_disease: Edge indices pointing from visits to diseases.
            - edge_index_visit_to_procedure: Edge indices pointing from visits to procedures.
            - edge_index_visit_to_drug: Edge indices pointing from visits to drugs.
            - edge_index_disease_to_symptom: Edge indices pointing from diseases to symptoms.
            - edge_index_anatomy_to_diagnosis: Edge indices pointing from anatomy to diagnosis.
            - edge_index_diagnosis_to_drug: Edge indices pointing from diagnosis to drugs.
            - edge_index_pharma_to_drug: Edge indices pointing from pharma to drugs.
            - edge_index_symptom_to_drug: Edge indices pointing from symptoms to drugs.
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

        # =============== MAPPING SYMPTOMS ===========================
        # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
        self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(self.icd9_symp_dict)

        presents_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)
        presents_symptom_id = torch.from_numpy(self.symp_df['ICD9_CODE'].values)

        # Create the edge index for the relationship 'presents' between visits and symptoms
        edge_index_visit_to_symptom = torch.stack([presents_visit_id, presents_symptom_id], dim=0)

        # =============== MAPPING DIAGNOSES ===========================
        # Substituting the values in the 'ICD9_CODE' column with the corresponding indices in the vocabulary
        self.diag_df['ICD9_CODE_DIAG'] = self.diag_df['ICD9_CODE'].map(self.icd9_diag_dict)
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.diag_df['HADM_ID'] = self.diag_df['HADM_ID'].map(self.hadm_dict)

        # Drop the 'ICD9_CODE' column that is no longer needed
        self.diag_df.drop('ICD9_CODE', axis=1, inplace=True)

        hasdisease_visit_id = torch.from_numpy(self.diag_df['HADM_ID'].values)
        hasdisease_disease_id = torch.from_numpy(self.diag_df['ICD9_CODE_DIAG'].values)

        # Create the edge index for the relationship 'has' between visits and diseases
        edge_index_visit_to_disease = torch.stack([hasdisease_visit_id, hasdisease_disease_id], dim=0)

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

        # =============== MAPPING DRUGS ===========================
        # Substituting the values in the 'ATC_CODE' column with the corresponding indices in the vocabulary
        self.drug_df['ATC_CODE_PRE'] = self.drug_df['ATC_CODE'].map(self.atc_pre_dict)
        # Substituting the values in the 'HADM_ID' column with the corresponding indices in the vocabulary
        self.drug_df['HADM_ID'] = self.drug_df['HADM_ID'].map(self.hadm_dict)

        # Drop the 'ATC_CODE' column that is no longer needed
        self.drug_df.drop('ATC_CODE', axis=1, inplace=True)

        hasreceived_visit_id = torch.from_numpy(self.drug_df['HADM_ID'].values)
        hasreceived_drug_id = torch.from_numpy(self.drug_df['ATC_CODE_PRE'].values)

        # Create the edge index for the relationship 'has_received' between visits and drugs
        edge_index_visit_to_drug = torch.stack([hasreceived_visit_id, hasreceived_drug_id], dim=0)

        # ==== GRAPH ENRICHMENT ====
        edge_index_disease_to_symptom = None
        edge_index_anatomy_to_diagnosis = None
        edge_index_diagnosis_to_drug = None
        edge_index_pharma_to_drug = None
        edge_index_symptom_to_drug = None

        if self.k > 0:
            for relation in self.static_kg:
                if relation == "DIAG_SYMP":
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

                    if not anat_diag_df.empty:
                        hasbeencaused_diag_id = torch.from_numpy(diag_symp_df['DIAG'].values)
                        hasbeencaused_symp_id = torch.from_numpy(diag_symp_df['SYMP'].values)
                        edge_index_disease_to_symptom = torch.stack([hasbeencaused_diag_id, hasbeencaused_symp_id], dim=0)
                    else:
                        # Initialize edge_index_disease_to_symptom as empty if the DataFrame is empty
                        edge_index_disease_to_symptom = torch.empty((2, 0), dtype=torch.int64)

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
                    drug_diag_df = self.stat_kg_df[relation].astype(str).copy()
                    drug_diag_df = drug_diag_df[drug_diag_df["DIAG"].isin(self.icd9_diag_dict.keys())]
                    drug_diag_df = drug_diag_df[drug_diag_df["DRUG"].isin(self.atc_pre_dict.keys())]

                    # Substituting the values in the 'DIAG' column with the corresponding indices in the vocabulary
                    drug_diag_df['DIAG'] = drug_diag_df['DIAG'].map(self.icd9_diag_dict)
                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    drug_diag_df['DRUG'] = drug_diag_df['DRUG'].map(self.atc_pre_dict)

                    if not drug_diag_df.empty:
                        treats_diag_id = torch.from_numpy(drug_diag_df['DIAG'].values)
                        treats_drug_id = torch.from_numpy(drug_diag_df['DRUG'].values)
                        edge_index_diagnosis_to_drug = torch.stack([treats_diag_id, treats_drug_id], dim=0)
                    else:
                        # Initialize edge_index_diagnosis_to_drug as empty if the DataFrame is empty
                        edge_index_diagnosis_to_drug = torch.empty((2, 0), dtype=torch.int64)

                elif (relation == "PC_DRUG") and (self.k == 2):
                    # =============== MAPPING PC_DRUG ===========================
                    # Copy the dataframe with the relationship PC_DRUG
                    pc_drug_df = self.stat_kg_df[relation].astype(str).copy()
                    pc_drug_df = pc_drug_df[pc_drug_df["DRUG"].isin(self.atc_pre_dict.keys())]

                    # Create a unique vocabulary from the codici PHARMACLASS
                    ndc_pc_vocab = pc_drug_df['PHARMACLASS'].unique()
                    # Create a dictionary that maps the codici PHARMACLASS to their index in the vocabulary
                    ndc_pc_dict = {code: i for i, code in enumerate(ndc_pc_vocab)}

                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    pc_drug_df['DRUG'] = pc_drug_df['DRUG'].map(self.atc_pre_dict)
                    # Substituting the values in the 'PHARMACLASS' column with the corresponding indices in the vocabulary
                    pc_drug_df['PHARMACLASS'] = pc_drug_df['PHARMACLASS'].map(ndc_pc_dict)

                    if not pc_drug_df.empty:
                        includes_pharma_id = torch.from_numpy(pc_drug_df['PHARMACLASS'].values)
                        includes_drug_id = torch.from_numpy(pc_drug_df['DRUG'].values)
                        edge_index_pharma_to_drug = torch.stack([includes_pharma_id, includes_drug_id], dim=0)
                    else:
                        # Initialize edge_index_pharma_to_drug as empty if the DataFrame is empty
                        edge_index_pharma_to_drug = torch.empty((2, 0), dtype=torch.int64)

                elif relation == "SYMP_DRUG":
                    # =============== MAPPING SYMP_DRUG ===========================
                    # Copy the dataframe with the relationship SYMP_DRUG
                    symp_drug_df = self.stat_kg_df[relation].astype(str).copy()
                    symp_drug_df = symp_drug_df[symp_drug_df["DRUG"].isin(self.atc_pre_dict.keys())] ###OCCHIO QUI

                    # Substituting the values in the 'DRUG' column with the corresponding indices in the vocabulary
                    symp_drug_df['DRUG'] = symp_drug_df['DRUG'].map(self.atc_pre_dict)

                    # Lookup the indices of the symptoms in the vocabulary
                    last_index = max(self.icd9_symp_dict.values())

                    # Add the new symptoms to the dictionary with consecutive indices
                    for symptom_code in symp_drug_df['SYMP'].unique():
                        if symptom_code not in self.icd9_symp_dict:
                            last_index += 1
                            self.icd9_symp_dict[symptom_code] = last_index
                            self.symp_df = pd.concat([self.symp_df, pd.DataFrame({'SUBJECT_ID': [0], 'HADM_ID': [0], 'SEQ_NUM': [0], 'ICD9_CODE': [symptom_code]})], ignore_index=True)
                    symp_drug_df['SYMP'] = symp_drug_df['SYMP'].map(self.icd9_symp_dict)

                    if not symp_drug_df.empty:
                        causes_symp_id = torch.from_numpy(symp_drug_df['SYMP'].values)
                        causes_drug_id = torch.from_numpy(symp_drug_df['DRUG'].values)
                        edge_index_symptom_to_drug = torch.stack([causes_symp_id, causes_drug_id], dim=0)
                    else:
                        # Initialize edge_index_symptom_to_drug as empty if the DataFrame is empty
                        edge_index_symptom_to_drug = torch.empty((2, 0), dtype=torch.int64)

        return edge_index_patient_to_visit, edge_index_visit_to_symptom, edge_index_visit_to_disease, \
                edge_index_visit_to_procedure, edge_index_visit_to_drug, edge_index_disease_to_symptom, \
                edge_index_anatomy_to_diagnosis, edge_index_diagnosis_to_drug, edge_index_pharma_to_drug, \
                edge_index_symptom_to_drug

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
        graph["symptom"].node_id = torch.arange(len(self.symp_df['ICD9_CODE'].unique()))
        graph["procedure"].node_id = torch.arange(len(self.proc_df['ICD9_CODE_PROC'].unique()))

        # Nodes of Static KG
        if self.k == 2:
            for relation in self.static_kg:
                if relation == "ANAT_DIAG":
                    graph["anatomy"].node_id = torch.arange(len(self.stat_kg_df[relation]['ANAT'].unique()))
                if relation == "PC_DRUG":
                    graph["pharmaclass"].node_id = torch.arange(len(self.stat_kg_df[relation]['PHARMACLASS'].unique()))

        if self.label_key == "conditions":
            graph["disease"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())
            graph["drug"].node_id = torch.arange(len(self.drug_df['ATC_CODE_PRE'].unique()))
        else:
            graph["disease"].node_id = torch.arange(len(self.diag_df['ICD9_CODE_DIAG'].unique()))
            graph["drug"].node_id = torch.arange(self.label_tokenizer.get_vocabulary_size())

        # Add the edge indices:
        graph["patient", "has", "visit"].edge_index = self.edge_index_patient_to_visit
        graph["visit", "presents", "symptom"].edge_index = self.edge_index_visit_to_symptom
        graph["visit", "has", "disease"].edge_index = self.edge_index_visit_to_disease
        graph["visit", "has_treat", "procedure"].edge_index = self.edge_index_visit_to_procedure
        graph["visit", "has_received", "drug"].edge_index = self.edge_index_visit_to_drug

        # Edges of Static KG
        if self.k > 0:
            for relation in self.static_kg:
                if relation == "DIAG_SYMP":
                    graph["disease", "has_been_caused_by", "symptom"].edge_index = self.edge_index_disease_to_symptom
                if (relation == "ANAT_DIAG") and (self.k == 2):
                    graph["disease", "localizes", "anatomy"].edge_index = self.edge_index_anatomy_to_diagnosis
                if relation == "DRUG_DIAG":
                    graph["disease", "treats", "drug"].edge_index = self.edge_index_diagnosis_to_drug
                if (relation == "PC_DRUG") and (self.k == 2):
                    graph["pharmaclass", "includes", "drug"].edge_index = self.edge_index_pharma_to_drug
                if relation == "SYMP_DRUG":
                    graph["symptom", "causes", "drug"].edge_index = self.edge_index_symptom_to_drug


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

    def convert_batches(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts the data into separate dataframes for procedures, symptoms, drugs, and conditions.

        Returns:
            A tuple of four pandas DataFrames representing the converted data:
            - proc_df: DataFrame containing procedure data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
            - symp_df: DataFrame containing symptom data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
            - drug_df: DataFrame containing drug data with columns 'SUBJECT_ID', 'HADM_ID', and 'ATC_CODE'.
            - diag_df: DataFrame containing condition data with columns 'SUBJECT_ID', 'HADM_ID', and 'ICD9_CODE'.
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
        drug_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ATC_CODE': self.drugs,
        })
        drug_df = drug_df.explode('ATC_CODE')
        drug_df = drug_df.explode('ATC_CODE')

        # CONDITIONS DataFrame
        diag_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.conditions,
        })
        diag_df = diag_df.explode('ICD9_CODE')
        diag_df = diag_df.explode('ICD9_CODE')

        return proc_df, symp_df, drug_df, diag_df

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
 
        subgraph = self.graph.subgraph({"patient": select_patient, "visit": select_visit, "symptom": select_symptom, "procedure": select_procedure, "disease": select_disease})

        return subgraph

    def create_y_prob_mat(self) -> torch.Tensor:
        """
        Create a probability matrix based on the given label key.

        Returns:
            torch.Tensor: The probability matrix.
        """
        if self.label_key == "drugs":
            edge_label_full = self.subgraph["visit", "has_received", "drug"].edge_label_index
        else:
            edge_label_full = self.subgraph["visit", "has", "disease"].edge_label_index

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
        Visualizes the graph by drawing its nodes and edges using networkx and matplotlib.

        This method selects a specific patient and visit from the graph, removes isolated nodes,
        converts the subgraph to a homogeneous graph, and then uses networkx to draw the graph.
        Each node is assigned a color based on its node type.

        Args:
            None

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
            3: 'disease',
            4: 'procedure',
            5: 'drug'
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
        # # Calculate class weights
        # class_weights = compute_class_weights(y_true)
 
        # # Create a tensor of weights for each datapoint
        class_weights = torch.tensor([0.6, 0.9])
        sample_weights = class_weights[y_true.long()]
 
        # Calculate the loss for each prediction
        loss = F.binary_cross_entropy(pred, y_true, weight=sample_weights)

        return loss

    def forward(
        self,
        patient_id: List[str],
        visit_id: List[str],
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        symptoms: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            patient_id (List[str]): List of patient IDs.
            visit_id (List[str]): List of visit IDs.
            conditions (List[List[List[str]]]): List of conditions.
            procedures (List[List[List[str]]]): List of procedures.
            symptoms (List[List[List[str]]]): List of symptoms.
            drugs (List[List[str]]): List of drugs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss, predicted probabilities, and true labels.
        """
        # Set the model to training mode
        self.patient_id = patient_id
        self.visit_id = visit_id
        self.conditions = conditions
        self.procedures = procedures
        self.symptoms = symptoms
        self.drugs = drugs

        # Prepare the labels
        if self.label_key == "drugs":
            y_true = self.prepare_labels(self.drugs, self.label_tokenizer)
        else:
            y_true = self.prepare_labels(self.conditions, self.label_tokenizer)

        # Convert the data into separate dataframes for procedures, symptoms, drugs, and conditions
        self.proc_df, self.symp_df, self.drug_df, self.diag_df = self.convert_batches()
        
        # Get the subgraph
        self.subgraph = self.get_subgraph()
        self.subgraph = self.generate_neg_samples()
        self.mask = self.generate_mask()

        self.subgraph = self.subgraph.to(device=self.device)

        self.node_features = self.x_dict(self.subgraph)

        # Get the loss and predicted probabilities
        if self.label_key == "drugs":
            pred = self.layer(self.node_features, self.subgraph.edge_index_dict, 
                                self.subgraph['visit', 'drug'].edge_label_index, 
                                self.subgraph['visit', 'drug'].edge_label, self.mask)
            loss = self.calculate_loss(pred, self.subgraph['visit', 'drug'].edge_label, self.mask)
        else:
            pred = self.layer(self.node_features, self.subgraph.edge_index_dict, 
                                self.subgraph['visit', 'disease'].edge_label_index, 
                                self.subgraph['visit', 'disease'].edge_label, self.mask)
            loss = self.calculate_loss(pred, self.subgraph['visit', 'disease'].edge_label, self.mask)

        # Prepare the predicted probabilities applying the sigmoid function
        self.y_prob = pred
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