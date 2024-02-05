from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class MICRONLayer(nn.Module):
    """MICRON layer.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    This layer is used in the MICRON model. But it can also be used as a
    standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        num_drugs: total number of medications to recommend.
        lam: regularization parameter for the reconstruction loss. Default is 0.1.

    Examples:
        >>> from pyhealth.models import MICRONLayer
        >>> patient_emb = torch.randn(3, 5, 32) # [patient, visit, input_size]
        >>> medications = torch.randint(0, 2, (3, 50)).float()
        >>> layer = MICRONLayer(32, 64, 50)
        >>> loss, y_prob = layer(patient_emb, medications)
        >>> loss.shape
        torch.Size([])
        >>> y_prob.shape
        torch.Size([3, 50])
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_drugs: int, lam: float = 0.1
    ):
        super(MICRONLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_drugs
        self.lam = lam

        self.health_net = nn.Linear(input_size, hidden_size)
        self.prescription_net = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_drugs)

        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_reconstruction_loss(
        logits: torch.tensor, logits_residual: torch.tensor, mask: torch.tensor
    ) -> torch.tensor:
        rec_loss = torch.mean(
            torch.square(
                torch.sigmoid(logits[:, 1:, :])
                - torch.sigmoid(logits[:, :-1, :] + logits_residual)
            )
            * mask[:, 1:].unsqueeze(2)
        )
        return rec_loss

    def forward(
        self,
        patient_emb: torch.tensor,
        medications: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            patient_emb: a tensor of shape [patient, visit, input_size].
            medications: a multihot tensor of shape [patient, num_labels].
            mask: an optional tensor of shape [patient, visit] where
                1 indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        # (patient, visit, hidden_size)
        health_rep = self.health_net(patient_emb)
        drug_rep = self.prescription_net(health_rep)
        logits = self.fc(drug_rep)
        logits_last_visit = get_last_visit(logits, mask)
        bce_loss = self.bce_loss_fn(logits_last_visit, medications)

        # (batch, visit-1, input_size)
        health_rep_last = health_rep[:, :-1, :]
        # (batch, visit-1, input_size)
        health_rep_cur = health_rep[:, 1:, :]
        # (batch, visit-1, input_size)
        health_rep_residual = health_rep_cur - health_rep_last
        drug_rep_residual = self.prescription_net(health_rep_residual)
        logits_residual = self.fc(drug_rep_residual)
        rec_loss = self.compute_reconstruction_loss(logits, logits_residual, mask)

        loss = bce_loss + self.lam * rec_loss
        y_prob = torch.sigmoid(logits_last_visit)

        return loss, y_prob


class MICRON(BaseModel):
    """MICRON model.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    Note:
        This model is only for medication prediction which takes diagnosis
        and procedures as feature_keys, and medications as label_key. It only operates
        on the visit level.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the MICRON layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MICRON, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        self.data_enrich = False
        for feature in feature_keys:
            if feature == "symptoms":
                self.data_enrich = True

        # validate kwargs for MICRON layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "num_drugs" in kwargs:
            raise ValueError("num_drugs is determined by the dataset")
        if self.data_enrich:
            self.micron = MICRONLayer(
                input_size=embedding_dim * 3,
                hidden_size=hidden_dim,
                num_drugs=self.label_tokenizer.get_vocabulary_size(),
                **kwargs
            )
        else:
            self.micron = MICRONLayer(
                input_size=embedding_dim * 2,
                hidden_size=hidden_dim,
                num_drugs=self.label_tokenizer.get_vocabulary_size(),
                **kwargs
            )

    def forward(
        self,
        diagnosis: List[List[List[str]]],
        procedures: List[List[List[str]]],
        symptoms: List[List[List[str]]],
        medications: List[List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            diagnosis: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
            symptoms: a nested list in three levels [patient, visit, symptom].
            medications: a nested list in two levels [patient, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels] representing
                    the ground truth of each drug.
        """
        diagnosis = self.feat_tokenizers["diagnosis"].batch_encode_3d(diagnosis)
        # (patient, visit, code)
        diagnosis = torch.tensor(diagnosis, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        diagnosis = self.embeddings["diagnosis"](diagnosis)
        # (patient, visit, embedding_dim)
        diagnosis = torch.sum(diagnosis, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)

        if self.data_enrich:
            symptoms = self.feat_tokenizers["symptoms"].batch_encode_3d(symptoms)
            # (patient, visit, code)
            symptoms = torch.tensor(symptoms, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            symptoms = self.embeddings["symptoms"](symptoms)
            # (patient, visit, embedding_dim)
            symptoms = torch.sum(symptoms, dim=2)

        # (patient, visit, embedding_dim * 2)
        if self.data_enrich:
            patient_emb = torch.cat([diagnosis, procedures, symptoms], dim=2)
        else:
            patient_emb = torch.cat([diagnosis, procedures], dim=2)
        # (patient, visit)
        mask = torch.sum(patient_emb, dim=2) != 0
        # (patient, num_labels)
        medications = self.prepare_labels(medications, self.label_tokenizer)

        loss, y_prob = self.micron(patient_emb, medications, mask)

        return {
            "loss": loss, 
            "y_prob": y_prob, 
            "y_true": medications
        }
