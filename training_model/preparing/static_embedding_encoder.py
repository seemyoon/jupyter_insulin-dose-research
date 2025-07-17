import torch
import torch as nn


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
        self.static_fc = nn.Linear(static_dim, hidden_dim)

    # +1 — because 0 will be used as an “empty value” that you manually insert when there is no data for the patient.
    # padding_idx=0 — PyTorch will fill the embedding vector with zeros itself and will not train this index (the gradient is not calculated for padding_idx).

    # With nn.Module, PyTorch automatically adds all these layers to the model's parameter list.
    def forward(self, static_tensor, drug_indices, comorb_indices):  # Convert lists of lists into tensors with padding

        drug_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(i, dtype=torch.long) for i in drug_indices], batch_first=True
        )

        # [torch.tensor(i, dtype=torch.long) for i in drug_indices]
        # → convert:
        # [[3, 7, 1], [5, 2], [4]]
        # ↓
        # [tensor([3, 7, 1]), tensor([5, 2]), tensor([4])]
        # dtype=torch.long — required because nn.Embedding requires integer indices (int64).

        # nn.utils.rnn.pad_sequence([...], batch_first=True)
        # Problem: the lengths of the lists are different (for example, 3, 2, 1).
        # The neural network requires that all tensors in the batch be of the same shape.
        # Solution: pad short lists with zeros on the right:
        # [
        # [3, 7, 1],   → length 3
        # [5, 2, 0],   → added 0 (padding)
        # [4, 0, 0]    → added two 0s (padding)
        # ]
        # pad_sequence(..., batch_first=True)

        # will return a tensor:
        # drug_padded = tensor([
        #     [3, 7, 1],
        #     [5, 2, 0],
        #     [4, 0, 0]
        # ], dtype=torch.long)

        comorb_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(i, dtype=torch.long) for i in comorb_indices], batch_first=True
        )

        drug_emb = self.drug_embedding(drug_padded)
        comorb_emb = self.comorb_embedding(comorb_padded)

        drug_emb_mean = drug_emb.mean(dim=1)
        comorb_emb_mean = comorb_emb.mean(dim=1)

        # [
        #     [1., 2., 3., 4.],
        #     [5., 6., 7., 8.],
        #     [9.,10.,11.,12.]
        # ]
        # drug_emb[0].mean(dim=0) (среднее по строкам):

        # What we get in drug_padded
        # drug_padded = tensor([
        #     [3, 7, 1],
        #     [5, 2, 0],
        #     [4, 0, 0]
        # ], dtype=torch.long)

        # What if we don't do mean?
        # Then we would get a tensor of shape (batch_size, max_drugs, emb_dim) → it would:
        # - take up more memory
        # - could not be simply concatenated with other features
        # - require additional layers (e.g., RNN or attention) to “roll up” the sequence into a fixed dimension

        static_repr = self.static_fc(static_tensor)

        return torch.cat([static_repr, drug_emb_mean, comorb_emb_mean], dim=1)
