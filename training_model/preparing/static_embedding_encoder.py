import torch
import torch.nn.functional as F
import torch.nn as nn


class StaticEmbedderEncoder(nn.Module):
    def __init__(self, static_dim, unique_drugs_size, unique_comorbities_size, emb_dim, hidden_dim):
        super().__init__()

        self.drug_embedding = nn.Embedding(
            num_embeddings=unique_drugs_size + 1,
            embedding_dim=emb_dim,
            padding_idx=0)

        self.comorb_embedding = nn.Embedding(
            num_embeddings=unique_comorbities_size + 1,
            embedding_dim=emb_dim,
            padding_idx=0)

        self.static_fc = nn.Linear(static_dim, hidden_dim)
        self.drug_fc = nn.Linear(emb_dim, hidden_dim)
        self.comorb_fc = nn.Linear(emb_dim, hidden_dim)

        self.final_fc = nn.Linear(hidden_dim * 3, hidden_dim)

    @staticmethod
    def _embed_and_pool(indices, embedding_layer):
        """Embed padded index tensor and mean-pool over non-padding positions.

        :param indices: (B, max_len) LongTensor with 0 as padding
        :param embedding_layer: nn.Embedding with padding_idx=0
        :return: (B, emb_dim) mean-pooled embedding
        """
        embedded = embedding_layer(indices)
        mask = (indices != 0).unsqueeze(-1).float()
        summed = (embedded * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count

    def forward(self, static_tensor, drug_indices, comorb_indices):
        """
        :param static_tensor: (B, static_dim) normalized static features
        :param drug_indices: (B, max_drugs) padded LongTensor
        :param comorb_indices: (B, max_comorbs) padded LongTensor
        """
        static_encoded = F.relu(self.static_fc(static_tensor))
        drug_encoded = F.relu(self.drug_fc(self._embed_and_pool(drug_indices, self.drug_embedding)))
        comorb_encoded = F.relu(self.comorb_fc(self._embed_and_pool(comorb_indices, self.comorb_embedding)))

        combined = torch.cat([static_encoded, drug_encoded, comorb_encoded], dim=1)
        return F.relu(self.final_fc(combined))
