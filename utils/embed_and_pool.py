import torch


def embed_and_pool(indices_lists, embedding_layer):
    """

    :param indices_lists:  list of lists with ID numbers.
    for example:
    [[3, 7, 1], # Patient 1 has 3 drugs with IDs 3, 7, and 1.
    [5, 2],     # Patient 2 has 2 drugs with IDs 5 and 2.
    [4]         # Patient 3 has 1 drug with ID 4.]
    :param embedding_layer: nn.Embedding layer that outputs a vector (e.g., length 8) for each ID.
    """

    max_len = max(len(lst) for lst in indices_lists)

    padded = torch.zeros(
        (len(indices_lists), max_len),
        dtype=torch.long)
    # You create a template for the batch, where you will later insert real values, and the zeros will remain as padding for alignment.

    # tensor([
    #     [0, 0, 0],  # for patient 1
    #     [0, 0, 0],  # for patient 2
    #     [0, 0, 0]   # for patient 3
    # ])

    for i, lst in enumerate(indices_lists):
        padded[i, :len(lst)] = torch.tensor(lst, dtype=torch.long)
        # insert the tensor lst (i.e., the list of drug indices) into the i-th row of the padded matrix, starting from position 0 to len(lst). The remainder will be filled with zeros.

        # Pads all lists with zeros on the right so that they have length 3
        # [3, 7, 1]  →  [3, 7, 1]
        # [5, 2]     →  [5, 2, 0]
        # [4]        →  [4, 0, 0]

        # padded = tensor([
        #   [3, 7, 1],
        #   [5, 2, 0],
        #   [4, 0, 0]
        # ])

    embedded = embedding_layer(padded)  # It replaces each index of the drug with a vector of size embedding_dim.

    # example embedding_layer - random numbers
    # embedding_table = {
    #     1: [0.1, 0.3],
    #     2: [0.5, 0.4],
    #     3: [0.2, 0.7],
    #     4: [0.9, 0.2],
    #     5: [0.8, 0.1],
    #     7: [0.3, 0.5],
    #     0: [0.0, 0.0]  # padding
    # }

    # Convert padded to embedded:
    # embedded = [
    #     [[0.2, 0.7], [0.3, 0.5], [0.1, 0.3]],       # patient 0
    #     [[0.8, 0.1], [0.5, 0.4], [0.0, 0.0]],       # patient 1
    #     [[0.9, 0.2], [0.0, 0.0], [0.0, 0.0]]        # patient 2
    # ]

    mask = (padded != 0).unsqueeze(-1).float()

    # padded != 0 → tensor([
    # [1, 1, 1],
    # [1, 1, 0],
    # [1, 0, 0]
    # ])
    # .unsqueeze(-1). example:
    #
    # output:
    # tensor([[[1],
    #          [2],
    #          [3]],
    #
    #         [[4],
    #          [5],
    #          [6]]])
    # ! Convert to 3D. If the mask remained in 2D, multiplying it with 3D bedding would result in an error due to a mismatch in sizes.

    # Why .float()?
    # By default, Boolean values are True/False.
    # But we need to multiply by numbers, so we convert Boolean True/False to the numbers 1.0 and 0.0.

    summed = (embedded * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)

    # What is dim=1?
    # dim=1 is the index of the axis along which we sum.
    # If the tensor has dimensions (3, 4, 5), then:
    #
    # dim=0 is the first axis (3),
    # dim=1 is the second axis (4),
    # dim=2 is the third axis (5).
    #
    # In our case, the size is (batch_size, max_len, emb_dim) → dim=1 is precisely the axis with the number of tokens (preparations/comorbidities).

    # mask.sum(dim=1)
    # [
    #     [1, 1, 1],
    #     [1, 1, 0], =>[3, 2, 1]
    # [1, 0, 0]
    # ]

    # Why clamp(min=1)?
    # To avoid division by 0, if suddenly the list of drugs is empty (all zeros). Then the sum will be 0, and you can't divide by 0. clamp(min=1) will replace 0 with 1.
    return summed / count
    # If we simply add up the embeddings of all drugs, those with many drugs will have a large vector, and those with few drugs will have a small vector. This is incorrect.

    # To ensure that all patients receive vectors of equal length, we take the average (i.e., divide the sum by the number).
    # Example:
    # Patient 1: vectors [1, 2, 3], [2, 3, 4], [3, 4, 5]
    # Sum: [6, 9, 12]
    # Average: [6/3, 9/3, 12/3] = [2, 3, 4]
    # Patient 2: vector [5, 6, 7]
    #
    # Average: [5, 6, 7] (only one vector)
