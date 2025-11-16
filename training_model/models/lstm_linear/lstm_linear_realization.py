from torch.optim import Adam
import torch.nn as nn
import torch

from training_model.models.lstm_linear.lstm_linear import LSTMLinearModel


class LSTMLinearRealization:
    def __init__(
            self,
            num_drug_types,
            num_ins,
            train_loader,
            val_loader,
            test_loader,
            static_dim=64,
            hidden_size=64,
            epochs=300,
            batch_size=32,
            lr=1e-3,
            patience=10,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.num_drug_types = num_drug_types
        self.num_ins = num_ins
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMLinearModel(
            static_dim=static_dim,
            hidden_size=hidden_size,
            num_drug_types=num_drug_types,
            num_insulin_types=num_ins
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss_fn = nn.MSELoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_path = 'best_model.pth'

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_mse_loss, train_bce_loss = 0, 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food_intake = batch['food_intake'].to(self.device)

                y_insulin = batch['y_insulin'].to(self.device)
                y_tab = batch['y_diabetes_tablet'].to(self.device)
                y_ins_mask = batch['y_insulin_mask'].to(self.device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(self.device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = self.model(
                    measurements,
                    static,
                    food_intake
                )

                mse_loss = self.mse_loss_fn(pred_ins_dose, y_insulin) + self.mse_loss_fn(pred_tab_dose, y_tab)
                bce_loss = self.bce_loss_fn(pred_ins_mask, y_ins_mask) + self.bce_loss_fn(pred_tab_mask, y_tab_mask)
                loss = mse_loss + bce_loss

                loss.backward(retain_graph=True)
                self.optimizer.step()

                train_mse_loss += mse_loss.item()
                train_bce_loss += bce_loss.item()

            train_mse_loss /= len(self.train_loader)
            train_bce_loss /= len(self.train_loader)

            val_mse_loss, val_bce_loss = self.validate()

            print(f"Epoch {epoch:3d} | Train MSE: {train_mse_loss:.4f}, BCE: {train_bce_loss:.4f} | "
                  f"Val MSE: {val_mse_loss:.4f}, BCE: {val_bce_loss:.4f}")

            val_total_loss = val_mse_loss + val_bce_loss
            if val_total_loss < self.best_val_loss:
                self.best_val_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print("Training finished.")
        self.test()

    def validate(self):
        self.model.eval()
        val_mse_loss, val_bce_loss = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food_intake = batch['food_intake'].to(self.device)

                y_insulin = batch['y_insulin'].to(self.device)
                y_tab = batch['y_diabetes_tablet'].to(self.device)
                y_ins_mask = batch['y_insulin_mask'].to(self.device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(self.device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = self.model(measurements, static,
                                                                                        food_intake)

                val_mse_loss += self.mse_loss_fn(pred_ins_dose, y_insulin).item() + self.mse_loss_fn(pred_tab_dose,
                                                                                                     y_tab).item()
                val_bce_loss += self.bce_loss_fn(pred_ins_mask, y_ins_mask).item() + self.bce_loss_fn(pred_tab_mask,
                                                                                                      y_tab_mask).item()

        return val_mse_loss / len(self.val_loader), val_bce_loss / len(self.val_loader)

    def test(self):
        print("\nTesting best model...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()

        test_mse_loss, test_bce_loss = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food_intake = batch['food_intake'].to(self.device)

                y_insulin = batch['y_insulin'].to(self.device)
                y_tab = batch['y_diabetes_tablet'].to(self.device)
                y_ins_mask = batch['y_insulin_mask'].to(self.device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(self.device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = self.model(measurements, static,
                                                                                        food_intake)

                test_mse_loss += self.mse_loss_fn(pred_ins_dose, y_insulin).item() + self.mse_loss_fn(pred_tab_dose,
                                                                                                      y_tab).item()
                test_bce_loss += self.bce_loss_fn(pred_ins_mask, y_ins_mask).item() + self.bce_loss_fn(pred_tab_mask,
                                                                                                       y_tab_mask).item()

        test_mse_loss /= len(self.test_loader)
        test_bce_loss /= len(self.test_loader)
        print(f"Test MSE: {test_mse_loss:.4f}, Test BCE: {test_bce_loss:.4f}")
