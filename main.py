from training_model.preparing.static_embedding_encoder import StaticEmbedderEncoder
from training_model.preparing.static_preprocessing import StaticProcessing
from training_model.repository import Repository


class FullModel:
    def __init__(self):
        self.repo = Repository()

    def get_static_data(self):
        patients = self.repo.get_patients_td()

        patient_to_drugs = self.repo.get_patient_drugs_map()
        patient_to_comorbities = self.repo.get_patient_comorbities_map()

        unique_drugs = StaticProcessing.get_unique_entities(list(patient_to_drugs.values()))
        unique_comorbities = StaticProcessing.get_unique_entities(list(patient_to_comorbities.values()))

        static_tensor, drug_indices, comorb_indices = StaticProcessing.get_static_tensor_with_embeddings(
            patients,
            unique_drugs,
            unique_comorbities,
            patient_to_drugs,
            patient_to_comorbities)

        encoder = StaticEmbedderEncoder(
            static_dim=static_tensor.shape[1],
            # static_tensor.shape = (e.g. 128, 9) → 128 patients, each with 9 features
            unique_drugs_size=len(unique_drugs),
            unique_comorbities_size=len(unique_comorbities),
            emb_dim=32,  # 32 - is default
            hidden_dim=64

        )
        output = encoder(static_tensor, drug_indices, comorb_indices)

        return {p.id: output[i] for i, p in enumerate(patients)}

        # output:
        # tensor([[0.2, 0.5, 0.1],   # for the first patient
        #         [0.8, 0.3, 0.0],   # for the second
        #         [0.4, 0.9, 0.6]])  # for the third

    def get_dynamic_data(self):
        pass

    def main(self):
        pass


if __name__ == '__main__':
    model = FullModel()
    model.main()
