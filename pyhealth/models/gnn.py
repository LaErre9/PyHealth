import torch
import math
import pkg_resources
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union
from rdkit import Chem
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth.models.utils import batch_to_multihot
from pyhealth.metrics import ddi_rate_score
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset

class GNNLayer(torch.nn.Module):
    """GNN model.

    Our Model.

    This layer is used in the GNN model. But it can also be used as a
    standalone layer.

    Args:
        hidden_size: hidden feature size.
    """

    def __init__(
        self,
        hidden_size: int,
        **kwargs,
    ):
        super(GNNLayer, self).__init__()

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
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the GNN layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super(GNN, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures", "symptoms"],
            label_key="drugs",
            mode="multilabel",
        )

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

        self.proc_df, self.symp_df, self.drug_df, self.diag_df = get_dataframe(self)

        self.edge_index_patient_to_symptom, self.edge_index_patient_to_disease, \
            self.edge_index_patient_to_procedure, self.edge_index_patient_to_drug = get_edge_index(self)
        
        self.graph = graph_definition(self)