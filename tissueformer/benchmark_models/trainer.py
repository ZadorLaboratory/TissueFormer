"""
Lightweight PyTorch training loop for benchmark models.

Supports:
- Standard train/eval with early stopping
- Per-method loss functions (CE for CellCnn/scAGG, BCE for ScRAT)
- L1 regularization hook for CellCnn
- Cosine LR schedule with warmup for ScRAT
- Logging to wandb
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb


class BenchmarkTrainer:
    """Training loop for benchmark DL models.

    Args:
        model: nn.Module to train.
        optimizer: PyTorch optimizer.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: torch device.
        n_epochs: Maximum number of epochs.
        loss_fn: Loss function (default: CrossEntropyLoss).
        loss_reduction: 'mean' or 'sum' (scAGG uses 'sum').
        early_stopping_patience: Stop after this many epochs without improvement.
            Set to 0 or None to disable.
        early_stopping_start_epoch: Only start early stopping after this epoch.
        scheduler: LR scheduler (optional).
        l1_loss_fn: Callable returning L1 loss (for CellCnn).
        model_name: Name prefix for logging.
        use_sigmoid: If True, apply sigmoid + BCE loss (ScRAT).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        n_epochs: int = 20,
        loss_fn: nn.Module | None = None,
        early_stopping_patience: int | None = 5,
        early_stopping_start_epoch: int = 0,
        scheduler=None,
        l1_loss_fn=None,
        model_name: str = "benchmark",
        use_sigmoid: bool = False,
        n_classes: int = 2,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.scheduler = scheduler
        self.l1_loss_fn = l1_loss_fn
        self.model_name = model_name
        self.use_sigmoid = use_sigmoid
        self.n_classes = n_classes

        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif use_sigmoid:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_start_epoch = early_stopping_start_epoch
        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0

    def _to_target(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels to the format expected by the loss function."""
        if self.use_sigmoid:
            # BCE expects one-hot float targets
            if labels.ndim == 1:
                targets = torch.zeros(
                    labels.size(0), self.n_classes, device=self.device
                )
                # Support soft labels (float)
                if labels.dtype in (torch.float32, torch.float64):
                    # Check if they're actually integers stored as float
                    if torch.all(labels == labels.long().float()):
                        targets.scatter_(1, labels.long().unsqueeze(1), 1.0)
                    else:
                        # Soft labels -- interpret as interpolation between classes
                        # For 2-class: label is probability of class 1
                        targets[:, 0] = 1.0 - labels
                        targets[:, 1] = labels
                else:
                    targets.scatter_(1, labels.unsqueeze(1), 1.0)
                return targets
            return labels.float()
        else:
            return labels.long()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for cells, labels, mask in self.train_loader:
            cells = cells.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(cells, mask=mask)
            targets = self._to_target(labels)
            loss = self.loss_fn(logits, targets)

            if self.l1_loss_fn is not None:
                loss = loss + self.l1_loss_fn()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate on a data loader.

        Returns:
            loss: Average loss.
            predictions: (n_samples,) integer predictions.
            labels: (n_samples,) true labels.
            probs: (n_samples, n_classes) probability estimates.
        """
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for cells, labels, mask in loader:
            cells = cells.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(cells, mask=mask)
            targets = self._to_target(labels)
            loss = self.loss_fn(logits, targets)
            total_loss += loss.item()
            n_batches += 1

            if self.use_sigmoid:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(np.int64))
            all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        predictions = np.concatenate(all_preds)
        labels_out = np.concatenate(all_labels)
        probs_out = np.concatenate(all_probs)
        return avg_loss, predictions, labels_out, probs_out

    def train(self) -> dict:
        """Full training loop with optional early stopping.

        Returns:
            Dict with best validation loss and final metrics.
        """
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels, val_probs = self.eval_epoch()

            val_acc = (val_preds == val_labels).mean()

            wandb.log(
                {
                    f"{self.model_name}/train_loss": train_loss,
                    f"{self.model_name}/val_loss": val_loss,
                    f"{self.model_name}/val_accuracy": val_acc,
                    f"{self.model_name}/epoch": epoch,
                    f"{self.model_name}/lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            print(
                f"  [{self.model_name}] Epoch {epoch+1}/{self.n_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.3f}"
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.patience_counter = 0
            else:
                if (
                    self.early_stopping_patience
                    and epoch >= self.early_stopping_start_epoch
                ):
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(
                            f"  [{self.model_name}] Early stopping at epoch {epoch+1}"
                        )
                        break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return {"best_val_loss": self.best_val_loss}
