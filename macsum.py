import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import trange
from dataclasses import dataclass
from typing import Callable, Type, Dict, List, Optional

from model import MacsumCore


@dataclass
class Prediction:
    """Data Transfer Object for model predictions."""
    lower: np.ndarray
    upper: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Returns the midpoint of the interval."""
        return (self.lower + self.upper) / 2.0

    @property
    def mpiw(self) -> np.ndarray:
        """Returns the width of the interval for each sample."""
        return self.upper - self.lower


class Macsum:
    """
    Main interface for the Macsum model.
    Provides a simplified workflow for training and inference.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        # Internal dictionary to store training progress
        self._history = {
            "train_loss": [], "val_loss": [],
            "train_picp": [], "val_picp": [],
            "train_mpiw": [], "val_mpiw": [],
            "lr": []
        }

    def history(self) -> Dict[str, List[float]]:
        """Returns the logged metrics from the training session."""
        return self._history

    def _compute_metrics(self, y_true, y_low, y_up):
        """Computes PICP (Coverage) and MPIW (Mean Width) for the current batch."""
        with torch.no_grad():
            # PICP: Prediction Interval Coverage Probability
            within_bounds = (y_true >= y_low) & (y_true <= y_up)
            picp = within_bounds.float().mean().item()

            # MPIW: Mean Prediction Interval Width
            mpiw = (y_up - y_low).mean().item()
        return picp, mpiw

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              loss_fn: Callable,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              lr: float = 0.001,
              batch_size: int = 32,
              optimizer_cls: Type[torch.optim.Optimizer] = optim.Adam):
        """
        Trains the model and logs performance metrics.
        - loss_fn: Required. A function (y_true, y_low, y_up) returning a scalar Tensor.
        """
        # Prepare Training Data
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_t, Y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare Validation Data (Optional)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(
                X_val, dtype=torch.float32, device=self.device)
            Y_val_t = torch.tensor(
                y_val, dtype=torch.float32, device=self.device)

        # Model and Optimizer initialization
        N = X.shape[1] if X.ndim > 1 else 1
        self.model = MacsumCore(N=N, device=self.device).to(self.device)
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)

        for epoch in trange(1, epochs + 1, desc="Macsum Training"):
            self.model.train()
            e_loss, e_picp, e_mpiw = 0.0, 0.0, 0.0

            for X_batch, Y_batch in dataloader:
                optimizer.zero_grad()
                y_low, y_up = self.model(X_batch)

                loss = loss_fn(Y_batch, y_low, y_up)
                loss.backward()
                optimizer.step()

                # Log Training Metrics
                p, m = self._compute_metrics(Y_batch, y_low, y_up)
                e_loss += loss.item()
                e_picp += p
                e_mpiw += m

            # Store Training History
            num_batches = len(dataloader)
            self._history["train_loss"].append(e_loss / num_batches)
            self._history["train_picp"].append(e_picp / num_batches)
            self._history["train_mpiw"].append(e_mpiw / num_batches)
            self._history["lr"].append(optimizer.param_groups[0]['lr'])

            # Perform Validation Step
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_low, v_up = self.model(X_val_t)
                    v_loss = loss_fn(Y_val_t, v_low, v_up).item()
                    v_picp, v_mpiw = self._compute_metrics(
                        Y_val_t, v_low, v_up)

                    self._history["val_loss"].append(v_loss)
                    self._history["val_picp"].append(v_picp)
                    self._history["val_mpiw"].append(v_mpiw)

        print("\nTraining Complete.")

    def predict(self, X: np.ndarray) -> Prediction:
        """Generates interval-valued predictions for the given input array."""
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_low, y_up = self.model(X_t)

        return Prediction(lower=y_low.cpu().numpy(), upper=y_up.cpu().numpy())
