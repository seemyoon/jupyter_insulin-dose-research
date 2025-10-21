from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

from training_model.diabetes_model import DiabetesModel
from training_model.glucose_dataset import GlucoseDataset
from training_model.preparing.static_embedding_encoder import StaticEmbedderEncoder
from training_model.preparing.static_preprocessing import StaticProcessing
from training_model.repository import Repository
from utils.make_windows import MakeWindows


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

    def main(self, epochs=20, batch_size=32, lr=1e-3):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=GlucoseDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=GlucoseDataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=GlucoseDataset.collate_fn)

        model = DiabetesModel(static_dim=64, hidden_size=64, num_drug_types=num_drug_types, num_insulin_types=num_ins)
        optimizer = Adam(model.parameters(), lr=lr)
        mse_loss_fn = nn.MSELoss()
        bce_loss_fn = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(epochs):
            # TRAIN
            model.train()
            train_mse_loss = 0
            train_bce_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                measurements = batch['measurements'].to(device)
                static = batch['static'].to(device)
                food_intake = batch['food_intake'].to(device)

                y_insulin = batch['y_insulin'].to(device)
                y_tab = batch['y_diabetes_tablet'].to(device)
                y_ins_mask = batch['y_insulin_mask'].to(device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = model(measurements, static, food_intake)

                mse_loss = mse_loss_fn(pred_ins_dose, y_insulin) + mse_loss_fn(pred_tab_dose, y_tab)
                bce_loss = bce_loss_fn(pred_ins_mask, y_ins_mask) + bce_loss_fn(pred_tab_mask, y_tab_mask)
                loss = mse_loss + bce_loss

                loss.backward(retain_graph=True)
                optimizer.step()

                train_mse_loss += mse_loss.item()
                train_bce_loss += bce_loss.item()

            train_mse_loss /= len(train_loader)
            train_bce_loss /= len(train_loader)

            # VALIDATION
            model.eval()
            val_mse_loss = 0
            val_bce_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    measurements = batch['measurements'].to(device)
                    static = batch['static'].to(device)
                    food_intake = batch['food_intake'].to(device)

                    y_insulin = batch['y_insulin'].to(device)
                    y_tab = batch['y_diabetes_tablet'].to(device)
                    y_ins_mask = batch['y_insulin_mask'].to(device)
                    y_tab_mask = batch['y_diabetes_tablet_mask'].to(device)

                    pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = model(measurements, static,
                                                                                       food_intake)

                    val_mse_loss += mse_loss_fn(pred_ins_dose, y_insulin).item() + mse_loss_fn(pred_tab_dose,
                                                                                               y_tab).item()
                    val_bce_loss += bce_loss_fn(pred_ins_mask, y_ins_mask).item() + bce_loss_fn(pred_tab_mask,
                                                                                                y_tab_mask).item()

            val_mse_loss /= len(val_loader)
            val_bce_loss /= len(val_loader)
            val_total_loss = val_mse_loss + val_bce_loss

            # save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save(model.state_dict(), 'best_model_gru.pth')

            print(f"Epoch {epoch:2d} | Train MSE: {train_mse_loss:.4f}, Train BCE: {train_bce_loss:.4f} | "
                  f"Val MSE: {val_mse_loss:.4f}, Val BCE: {val_bce_loss:.4f}")

        # TEST
        model.load_state_dict(torch.load('best_model_gru.pth'))
        model.eval()
        test_mse_loss = 0
        test_bce_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                measurements = batch['measurements'].to(device)
                static = batch['static'].to(device)
                food_intake = batch['food_intake'].to(device)

                y_insulin = batch['y_insulin'].to(device)
                y_tab = batch['y_diabetes_tablet'].to(device)
                y_ins_mask = batch['y_insulin_mask'].to(device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = model(measurements, static, food_intake)

                test_mse_loss += mse_loss_fn(pred_ins_dose, y_insulin).item() + mse_loss_fn(pred_tab_dose, y_tab).item()
                test_bce_loss += bce_loss_fn(pred_ins_mask, y_ins_mask).item() + bce_loss_fn(pred_tab_mask,
                                                                                             y_tab_mask).item()

        test_mse_loss /= len(test_loader)
        test_bce_loss /= len(test_loader)
        print(f"Test MSE: {test_mse_loss:.4f}, Test BCE: {test_bce_loss:.4f}")


if __name__ == '__main__':
    model = FullModel()
    model.main()
