import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch.nn.functional as F
from torch.nn import Linear

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

#### Define a simple GNN model:
class GNN_Conv(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.conv4 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = self.dropout(x)

        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    
    def forward(self, x_visit: torch.Tensor, x_drug: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_visit = x_visit[edge_label_index[0]]
        edge_feat_drug = x_drug[edge_label_index[1]]

        z = torch.cat([edge_feat_visit, edge_feat_drug], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)

        return z.view(-1)
    """
    def forward(self, x_visit: torch.Tensor, x_drug: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_visit = x_visit[edge_label_index[0]]
        edge_feat_drug = x_drug[edge_label_index[1]]

        # Calculate cosine similarity between edge features:
        # cosine_sim = F.cosine_similarity(edge_feat_patient, edge_feat_drug, dim=-1) # PROVA COSINE SIMILARITY
        return (edge_feat_visit * edge_feat_drug).sum(dim=-1)

        # return cosine_sim
    """

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
        mask: torch.Tensor,
        hidden_channels: int,
        embedding_dim: int,
        **kwargs,
    ):
        super(GNNLayer, self).__init__()

        # Since the dataset does not come with rich features, we also learn four
        # embedding matrices for patients, symptoms, procedures and diseases:
        self.pat_emb = torch.nn.Embedding(data["patient"].num_nodes, embedding_dim)
        self.vis_emb = torch.nn.Embedding(data["visit"].num_nodes, embedding_dim)
        self.symp_emb = torch.nn.Embedding(data["symptom"].num_nodes, embedding_dim)
        self.proc_emb = torch.nn.Embedding(data["procedure"].num_nodes, embedding_dim)
        self.dis_emb = torch.nn.Embedding(data["disease"].num_nodes, embedding_dim)
        self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, embedding_dim)

        # Instantiate homogeneous GNN:
        self.gnn = GNN_Conv(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier(hidden_channels)

    def calculate_loss(self, pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates the loss."""
        # print('y_true: ' + str(y_true[mask].shape))
        # print('pred: ' + str(pred[mask].shape))

        loss = F.binary_cross_entropy_with_logits(pred, y_true)

        return loss
    
    def forward(self, data: HeteroData, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = {
          "patient": self.pat_emb(data["patient"].node_id),
          "visit": self.vis_emb(data["visit"].node_id),
          "symptom": self.symp_emb(data["symptom"].node_id),
          "procedure": self.proc_emb(data["procedure"].node_id),
          "disease": self.dis_emb(data["disease"].node_id),
          "drug": self.drug_emb(data["drug"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["visit"],
            x_dict["drug"],
            data["visit", "has_received", "drug"].edge_label_index,
        )

        loss = self.calculate_loss(pred, data["visit", "has_received", "drug"].edge_label, mask)

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
        hidden_channels: int = 128,
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
        self.label_tokenizer = self.get_label_tokenizer()

        # print('vocab_size: ' + str(self.label_tokenizer.vocabulary('A03A')))

        self.proc_df, self.symp_df, self.drug_df, self.diag_df = self.get_dataframe()

        self.atc_pre_dict = self.get_dict_drugs()

        self.edge_index_patient_to_visit, self.edge_index_visit_to_symptom, \
            self.edge_index_visit_to_disease, self.edge_index_visit_to_procedure, \
            self.edge_index_visit_to_drug = self.get_edge_index()

        self.graph = self.graph_definition()

        self.train_loader = self.get_batches()

        self.mask = self.generate_mask()

        self.layer = GNNLayer(self.train_loader, self.mask, self.hidden_channels, self.embedding_dim)

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

    def get_dict_drugs(self) -> Dict[str, int]:
        # Creazione di un vocabolario unico dai codici ATC
        atc_pre_vocab = self.drug_df['ATC_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ATC al suo indice nel vocabolario
        atc_pre_dict = {code: self.label_tokenizer.vocabulary(code) for code in atc_pre_vocab}

        return atc_pre_dict

    def get_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the graph.

        Returns:
            graph: a `torch_geometric.data.HeteroData` object.
        """
        # =============== MAPPING VISITS ===========================
        # print('Num. Patients: ' + str(len(self.proc_df['SUBJECT_ID'].unique())))
        # print('Num. Visits: ' + str(len(self.proc_df['HADM_ID'].unique())))

        # Creazione di un vocabolario unico dai codici HADM_ID
        hadm_vocab = self.symp_df['HADM_ID'].unique()        
        # Creazione di un vocabolario unico dai SUBJECT_ID
        subject_vocab = self.symp_df['SUBJECT_ID'].unique()

        # Creazione di un dizionario che mappa il HADM_ID al suo indice nel vocabolario
        hadm_dict = {code: i for i, code in enumerate(hadm_vocab)}
        # Creazione di un dizionario che mappa il SUBJECT_ID al suo indice nel vocabolario
        subject_dict = {code: i for i, code in enumerate(subject_vocab)}

        # Sostituzione dei valori nella colonna 'HADM_ID' con i corrispondenti indici nel vocabolario
        self.symp_df['HADM_ID'] = self.symp_df['HADM_ID'].map(hadm_dict)
        # Sostituzione dei valori nella colonna 'SUBJECT_ID' con i corrispondenti indici nel vocabolario
        self.symp_df['SUBJECT_ID'] = self.symp_df['SUBJECT_ID'].map(subject_dict)

        has_patient_id = torch.from_numpy(self.symp_df['SUBJECT_ID'].values)
        has_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)

        edge_index_patient_to_visit = torch.stack([has_patient_id, has_visit_id], dim=0)

        # print('Dimension of edge index: ' + str(edge_index_patient_to_visit.shape))
        # print("Final edge indices pointing from patients to visits:")
        # print("=================================================")
        # print(edge_index_patient_to_visit)
        # print("=================================================")

        # =============== MAPPING SYMPTOMS ===========================
        # print('Num. Visits: ' + str(len(self.symp_df['HADM_ID'].unique())))
        # print('Num. Symptoms: ' + str(len(self.symp_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_symp_vocab = self.symp_df['ICD9_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_symp_dict = {code: i for i, code in enumerate(icd9_symp_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.symp_df['ICD9_CODE'] = self.symp_df['ICD9_CODE'].map(icd9_symp_dict)

        presents_visit_id = torch.from_numpy(self.symp_df['HADM_ID'].values)
        presents_symptom_id = torch.from_numpy(self.symp_df['ICD9_CODE'].values)

        edge_index_visit_to_symptom = torch.stack([presents_visit_id, presents_symptom_id], dim=0)

        # print('Dimension of edge index: ' + str(edge_index_visit_to_symptom.shape))
        # print("Final edge indices pointing from visits to symptoms:")
        # print("=================================================")
        # print(edge_index_visit_to_symptom)
        # print("=================================================")

        # =============== MAPPING DIAGNOSES ===========================
        # print('Num. Visits: ' + str(len(self.diag_df['HADM_ID'].unique())))
        # print('Num. Diseases: ' + str(len(self.diag_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_diag_vocab = self.diag_df['ICD9_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_diag_dict = {code: i for i, code in enumerate(icd9_diag_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.diag_df['ICD9_CODE_DIAG'] = self.diag_df['ICD9_CODE'].map(icd9_diag_dict)
        # Sostituzione dei valori nella colonna 'HADM_ID' con i corrispondenti indici nel vocabolario
        self.diag_df['HADM_ID'] = self.diag_df['HADM_ID'].map(hadm_dict)

        self.diag_df.drop('ICD9_CODE', axis=1, inplace=True)

        hasdisease_visit_id = torch.from_numpy(self.diag_df['HADM_ID'].values)
        hasdisease_disease_id = torch.from_numpy(self.diag_df['ICD9_CODE_DIAG'].values)

        edge_index_visit_to_disease = torch.stack([hasdisease_visit_id, hasdisease_disease_id], dim=0)

        # print('Dimension of edge index: ' + str(edge_index_visit_to_disease.shape))
        # print("Final edge indices pointing from visits to diseases:")
        # print("=================================================")
        # print(edge_index_visit_to_disease)
        # print("=================================================")

        # =============== MAPPING PROCEDURES ===========================
        # print('Num. Visits: ' + str(len(self.proc_df['HADM_ID'].unique())))
        # print('Num. Procedures: ' + str(len(self.proc_df['ICD9_CODE'].unique())))

        # Creazione di un vocabolario unico dai codici ICD9_CODE
        icd9_proc_vocab = self.proc_df['ICD9_CODE'].unique()

        # Creazione di un dizionario che mappa il codice ICD9_CODE al suo indice nel vocabolario
        icd9_proc_dict = {code: i for i, code in enumerate(icd9_proc_vocab)}

        # Sostituzione dei valori nella colonna 'ICD9_CODE' con i corrispondenti indici nel vocabolario
        self.proc_df['ICD9_CODE_PROC'] = self.proc_df['ICD9_CODE'].map(icd9_proc_dict)
        # Sostituzione dei valori nella colonna 'HADM_ID' con i corrispondenti indici nel vocabolario
        self.proc_df['HADM_ID'] = self.proc_df['HADM_ID'].map(hadm_dict)

        self.proc_df.drop('ICD9_CODE', axis=1, inplace=True)

        hastreat_visit_id = torch.from_numpy(self.proc_df['HADM_ID'].values)
        hastreat_procedure_id = torch.from_numpy(self.proc_df['ICD9_CODE_PROC'].values)

        edge_index_visit_to_procedure = torch.stack([hastreat_visit_id, hastreat_procedure_id], dim=0)

        # print('Dimension of edge index: ' + str(edge_index_visit_to_procedure.shape))
        # print("Final edge indices pointing from visits to procedures:")
        # print("=================================================")
        # print(edge_index_visit_to_procedure)
        # print("=================================================")

        # =============== MAPPING DRUGS ===========================
        # print('Num. Visits: ' + str(len(self.drug_df['HADM_ID'].unique())))
        # print('Num. Drugs: ' + str(len(self.drug_df['ATC_CODE'].unique())))

        # Sostituzione dei valori nella colonna 'ATC' con i corrispondenti indici nel vocabolario
        self.drug_df['ATC_CODE_PRE'] = self.drug_df['ATC_CODE'].map(self.atc_pre_dict)
        # Sostituzione dei valori nella colonna 'HADM_ID' con i corrispondenti indici nel vocabolario
        self.drug_df['HADM_ID'] = self.drug_df['HADM_ID'].map(hadm_dict)

        self.drug_df.drop('ATC_CODE', axis=1, inplace=True)
        # self.drug_df.drop(['SUBJECT_ID','ATC_CODE_PRE'], axis=1, inplace=True)

        hasreceived_visit_id = torch.from_numpy(self.drug_df['HADM_ID'].values)
        hasreceived_drug_id = torch.from_numpy(self.drug_df['ATC_CODE_PRE'].values)

        edge_index_visit_to_drug = torch.stack([hasreceived_visit_id, hasreceived_drug_id], dim=0)

        # print('Dimension of edge index: ' + str(edge_index_visit_to_drug.shape))
        # print("Final edge indices pointing from visits to drugs:")
        # print("=================================================")
        # print(edge_index_visit_to_drug)

        return edge_index_patient_to_visit, edge_index_visit_to_symptom, edge_index_visit_to_disease, \
               edge_index_visit_to_procedure, edge_index_visit_to_drug
    
    def graph_definition(self) -> HeteroData:
        # Creazione del grafo
        graph = HeteroData()

        # Save node indices:
        graph["patient"].node_id = torch.arange(len(self.symp_df['SUBJECT_ID'].unique()))
        graph["visit"].node_id = torch.arange(len(self.symp_df['HADM_ID'].unique()))
        graph["symptom"].node_id = torch.arange(len(self.symp_df['ICD9_CODE'].unique()))
        graph["disease"].node_id = torch.arange(len(self.diag_df['ICD9_CODE_DIAG'].unique()))
        graph["procedure"].node_id = torch.arange(len(self.proc_df['ICD9_CODE_PROC'].unique()))
        graph["drug"].node_id = torch.arange(len(self.atc_pre_dict))

        # Add the edge indices:
        graph["patient", "has", "visit"].edge_index = self.edge_index_patient_to_visit
        graph["visit", "presents", "symptom"].edge_index = self.edge_index_visit_to_symptom
        graph["visit", "has", "disease"].edge_index = self.edge_index_visit_to_disease
        graph["visit", "has_treat", "procedure"].edge_index = self.edge_index_visit_to_procedure
        graph["visit", "has_received", "drug"].edge_index = self.edge_index_visit_to_drug

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        graph = T.ToUndirected()(graph)

        # print("=================================================")
        # print("Final graph:")
        # print(graph)

        return graph

    def get_batches(self) -> HeteroData:
        neg_edges = negative_sampling(self.graph['visit', 'has_received', 'drug'].edge_index, num_nodes=(self.graph['visit'].num_nodes,self.graph['drug'].num_nodes))
        
        self.graph['visit', 'has_received', 'drug'].edge_label_index = self.graph['visit', 'has_received', 'drug'].edge_index
        self.graph['visit', 'has_received', 'drug'].edge_label = torch.ones(self.graph['visit', 'has_received', 'drug'].edge_label_index.shape[1], dtype=torch.float)
        self.graph['visit', 'has_received', 'drug'].edge_label_index = torch.cat((self.graph['visit', 'has_received', 'drug'].edge_label_index, neg_edges), dim=1)
        self.graph['visit', 'has_received', 'drug'].edge_label = torch.cat((self.graph['visit', 'has_received', 'drug'].edge_label, torch.zeros(neg_edges.shape[1], dtype=torch.float)), dim=0)
        
        # print(self.graph['visit', 'has_received', 'drug'].edge_label_index.shape)
        # print(self.graph['visit', 'has_received', 'drug'].edge_label)

        return self.graph

    def generate_mask(self) -> torch.Tensor:
        mask = torch.ones_like(self.graph['visit', 'has_received', 'drug'].edge_label, dtype=torch.bool)
        # print(mask)

        # Ottieni tutti i possibili archi nel grafo
        all_possible_edges = torch.cartesian_prod(torch.arange(self.graph['visit'].num_nodes), torch.arange(self.graph['drug'].num_nodes))

        # Filtra gli archi esistenti nel grafo attuale
        existing_edges = self.graph['visit', 'has_received', 'drug'].edge_label_index.t().contiguous()

        # Trova gli archi mancanti nel grafo attuale
        missing_edges = torch.tensor(list(set(map(tuple, all_possible_edges.tolist())) - set(map(tuple, existing_edges.tolist())))).t().contiguous()

        self.graph['visit', 'has_received', 'drug'].edge_label_index = torch.cat([self.graph['visit', 'has_received', 'drug'].edge_label_index, missing_edges], dim=1)
        self.graph['visit', 'has_received', 'drug'].edge_label = torch.cat([self.graph['visit', 'has_received', 'drug'].edge_label, torch.zeros(missing_edges.size(1), dtype=torch.float)], dim=0)

        # Stampa gli archi mancanti
        # print("Edges mancanti nel grafo attuale:")
        # print(missing_edges.shape)

        # Estensione della maschera con False per gli archi mancanti
        mask = torch.cat([mask, torch.zeros(missing_edges.size(1), dtype=torch.bool)], dim=0)

        # Stampa la maschera
        # print("Maschera estesa:")
        # print(mask.shape)

        # Stampa il numero di True e False nella maschera
        # print("Numero di True in mask:", torch.sum(mask).item())
        # print("Numero di False in mask:", mask.size(0) - torch.sum(mask).item())

        return mask

    def convert_batches(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        symp_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.symptoms,
        })
        symp_df = symp_df.explode('ICD9_CODE')
        symp_df = symp_df.explode('ICD9_CODE')

        proc_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.procedures,
        })
        proc_df = proc_df.explode('ICD9_CODE')
        proc_df = proc_df.explode('ICD9_CODE')

        drug_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ATC_CODE': self.drugs,
        })
        drug_df = drug_df.explode('ATC_CODE')
        drug_df = drug_df.explode('ATC_CODE')

        diag_df = pd.DataFrame({
            'SUBJECT_ID': self.patient_id,
            'HADM_ID': self.visit_id,
            'ICD9_CODE': self.conditions,
        })
        diag_df = diag_df.explode('ICD9_CODE')
        diag_df = diag_df.explode('ICD9_CODE')

        return proc_df, symp_df, drug_df, diag_df

    def create_y_prob_mat(self) -> torch.Tensor:
        data = {
            'Visit_ID': [], 
            'Drug_ID': [], 
            'Prob': []
        }

        edge_label_full = (self.train_loader["visit", "has_received", "drug"].edge_label_index).numpy()
        prob_full = self.y_prob.detach().numpy()
        vis = edge_label_full[0]
        dru = edge_label_full[1]
        for i in range(0,len(vis)):
            data['Visit_ID'].append(vis[i])
            data['Drug_ID'].append(dru[i])
            data['Prob'].append(prob_full[i])

        df = pd.DataFrame(data)
        df.sort_values(by=['Drug_ID'], ascending=[True], inplace=True)
        
        # Creare una lista di visite uniche e di drug uniche
        unique_visits = df['Visit_ID'].unique()
        unique_drugs = df['Drug_ID'].unique()

        # Creare una matrice di zeri con dimensioni (num. visite, num. drug distinte)
        y_prob_mat = torch.zeros(len(unique_visits), self.label_tokenizer.get_vocabulary_size())

        # Riempire la matrice con i valori di probabilità
        for i, visit in enumerate(unique_visits):
            # print('visit: ' + str(visit))
            visit_data = df[df['Visit_ID'] == visit]
            for _, row in visit_data.iterrows():
                drug_index = torch.tensor([np.where(unique_drugs == row['Drug_ID'])[0][0]], dtype=torch.long)
                # print('drug_index: ' + str(drug_index))
                y_prob_mat[i, drug_index] = torch.tensor([row['Prob']], dtype=torch.float)
                # print('prob: ' + str(row['Prob']))

        return y_prob_mat

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

        self.patient_id = patient_id
        self.visit_id = visit_id
        self.conditions = conditions
        self.procedures = procedures
        self.symptoms = symptoms
        self.drugs = drugs

        # print('visit_id: ' + str(visit_id))
        # print('drugs: ' + str(drugs))
        # print('----------------------------------')

        # print('ATC_PRE_DICT: ' + str(self.atc_pre_dict))
        # print()

        # labels_index = self.label_tokenizer.batch_encode_2d(
        #         self.drugs, padding=False, truncation=False
        #     )
        # print('labels_index: ' + str(labels_index))
        # print()

        # tokens = self.label_tokenizer.batch_decode_2d(
        #         labels_index, padding=False
        #     )
        # print('tokens: ' + str(tokens))
        # print()

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(self.drugs, self.label_tokenizer)

        # print('y_true: ' + str(y_true))
        # print()

        self.proc_df, self.symp_df, self.drug_df, self.diag_df = self.convert_batches()

        self.edge_index_patient_to_visit, self.edge_index_visit_to_symptom, \
            self.edge_index_visit_to_disease, self.edge_index_visit_to_procedure, \
            self.edge_index_visit_to_drug = self.get_edge_index()

        self.graph = self.graph_definition()

        self.train_loader = self.get_batches()

        self.mask = self.generate_mask()

        # print(self.train_loader['visit', 'has_received', 'drug'].edge_label)
        loss, pred = self.layer(self.train_loader, self.mask)

        self.y_prob = self.prepare_y_prob(pred)

        y_prob_mat = self.create_y_prob_mat()

        # print('y_prob_mat: ' + str(y_prob_mat))
        # print()

        return {
            "loss": loss,
            "y_prob": y_prob_mat,
            "y_true": y_true
        }