from training_model.preparing.static_embedding_encoder import StaticEmbedderEncoder
from training_model.preparing.static_preprocessing import StaticProcessing
from training_model.repository import Repository

repo = Repository()

patients = repo.get_patients_td()

patient_to_drugs = repo.get_patient_drugs_map()
patient_to_comorbities = repo.get_patient_comorbities_map()

unique_drugs = StaticProcessing.get_unique_entities(list(patient_to_drugs.values()))
unique_comorbities = StaticProcessing.get_unique_entities(list(patient_to_comorbities.values()))

static_tensor, drug_indices, comorb_indices = StaticProcessing.get_static_tensor_with_embeddings(
    patients,
    unique_drugs,
    unique_comorbities,
    patient_to_drugs,
    patient_to_comorbities)

# creating a static data encoder model
static_dim = static_tensor.shape[1]  # static_tensor.shape = (e.g. 128, 9) → 128 patients, each with 9 features
unique_drugs_size = len(unique_drugs)
unique_comorbities_size = len(unique_comorbities)
emb_dim = 32  # 32 - is default
hidden_dim = 64

encoder = StaticEmbedderEncoder(static_dim, unique_drugs_size, unique_comorbities_size, emb_dim, hidden_dim)

static_data = encoder(static_tensor, drug_indices, comorb_indices)
