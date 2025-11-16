from torch.utils.data import random_split, DataLoader

from training_model.glucose_dataset import GlucoseDataset
from training_model.models.hybrid.hybrid import HybridRealization
from training_model.models.lstm_linear.lstm_linear_realization import LSTMLinearRealization
from training_model.models.xgboost.xgboost_model import XGBoostRealization
from training_model.preparing.static_embedding_encoder import StaticEmbedderEncoder
from training_model.preparing.static_preprocessing import StaticProcessing
from training_model.repository import Repository
from utils.make_windows import MakeWindows


class FullModel:
    def __init__(self, model_type="hybrid"):
        self.repo = Repository()
        self.model_type = model_type.lower()  # 'lstm' | 'xgboost' | 'hybrid'

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
            patient_to_comorbities
        )

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
        insulin_val = self.repo.get_taking_insulin()
        tablets_val = self.repo.get_taking_tablets()
        meas = self.repo.get_measurements()
        food_intake = self.repo.get_dietary()

        return MakeWindows().build_feature_windows(insulin_val, tablets_val, meas, food_intake)

    def main(self, batch_size=32):
        static_dict = self.get_static_data()
        windows = self.get_dynamic_data()

        num_ins = len(self.repo.get_insulin_list())
        num_drug_types = len(self.repo.get_tablets_list())

        full_dataset = GlucoseDataset(windows, static_dict, num_ins, num_drug_types)

        # split: 80% train, 10% val, 10% test
        n_total = len(full_dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [n_train, n_val, n_test])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=GlucoseDataset.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GlucoseDataset.collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GlucoseDataset.collate_fn
        )

        match self.model_type:
            case "lstm":
                lstm_linear = LSTMLinearRealization(num_drug_types, num_ins, train_loader, val_loader, test_loader)
                lstm_linear.train()
                lstm_linear.validate()
                lstm_linear.test()

            case "xgboost":
                XGBoostRealization(num_drug_types, num_ins, train_loader, val_loader, test_loader)

            case "hybrid":
                trainer = HybridRealization(num_drug_types, num_ins, train_loader, val_loader, test_loader)
                trainer.hybrid_train()


if __name__ == '__main__':
    model = FullModel()
    model.main()
