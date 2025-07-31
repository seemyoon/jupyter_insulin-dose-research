import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.embed_and_pool import embed_and_pool


class StaticEmbedderEncoder(nn.Module):
    # creation of a neural network module
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

        # Index:     Vector:
        # 0 →        [0.0, 0.0, 0.0]      ← padding_idx (not trained)
        # 1 →        [0.25, -0.1, 0.8]    ← embedding of medicine 1
        # 2 →        [-0.3, 0.4, 0.6]     ← embedding medicine 2
        # 3 →        [0.7, -0.2, -0.5]    ← embedding medicine 3
        # 4 →        [0.1, 0.3, -0.9]     ← embedding medicine 4

        # When there is no drug, it is necessary to indicate that there is none, therefore padding_idx=0 is used. After we added the number of unique drugs, it increased, i.e., by one. Index 0 is reserved padding.
        # padding_idx=0 is a parameter that tells PyTorch not to update the vector for index 0.

        # liner layers
        self.static_fc = nn.Linear(static_dim, hidden_dim)
        self.drug_fc = nn.Linear(emb_dim, hidden_dim)
        self.comorb_fc = nn.Linear(emb_dim, hidden_dim)

        self.final_fc = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, static_tensor, drug_indices, comorb_indices):  # Convert lists of lists into tensors with padding
        static_encoded = F.relu(self.static_fc(static_tensor))
        drug_encoded = F.relu(self.drug_fc(embed_and_pool(drug_indices, self.drug_embedding)))
        comorb_encoded = F.relu(self.comorb_fc(embed_and_pool(comorb_indices, self.comorb_embedding)))

        combined = torch.cat([static_encoded, drug_encoded, comorb_encoded], dim=1)
        return F.relu(self.final_fc(combined))
