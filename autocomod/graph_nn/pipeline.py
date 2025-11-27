from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .contrastive_loss import supervised_contrastive_loss
from .metrics import MetricsComputing


class GnnPipeline:
    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module,
        root_classifier: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        self.model = model
        self.projector = projector
        self.root_classifier = root_classifier
        self.optimizer = optimizer
        self.device = device
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    def inference(self, batch: Data, training: bool = True):
        batch = batch.to(self.device)

        # Forward pass
        z = self.model(batch.x, batch.edge_index)

        # Contrastive objective
        h = self.projector(z)
        loss_contrastive = supervised_contrastive_loss(h, batch.y)

        # Root classification objective
        root_logits = self.root_classifier(z)
        root_labels = (batch.y == 0).float().unsqueeze(1)
        loss_bce = self.bce_loss_fn(root_logits, root_labels)

        # Total loss
        loss = loss_contrastive + loss_bce

        if training:
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return (
            z,
            h,
            root_logits,
            {
                "bce": loss_bce.item(),
                "contrastive": loss_contrastive.item(),
            },
        )

    def _inference_loader(self, loader: DataLoader, training: bool = True):
        metrics = defaultdict(float)

        for batch in loader:
            _, h, _, losses = self.inference(batch, training=training)
            for key, value in losses.items():
                metrics[key] += value

            true_labels = batch.y
            for key, value in MetricsComputing.compute_all(
                h, true_labels, batch.edge_index
            ).items():
                metrics[key] += value

        for key, value in metrics.items():
            metrics[key] /= len(loader)

        return metrics

    def train(self, loader: DataLoader):
        self.model.train()
        self.projector.train()
        return self._inference_loader(loader, training=True)

    def validate(self, loader: DataLoader):
        self.model.eval()
        self.projector.eval()
        return self._inference_loader(loader, training=False)

    def save(self, filepath: str) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "root_classifier_state_dict": self.root_classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "device": str(self.device),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        model: nn.Module,
        projector: nn.Module,
        root_classifier: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device | None = None,
    ) -> "GnnPipeline":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        projector.load_state_dict(checkpoint["projector_state_dict"])
        root_classifier.load_state_dict(checkpoint["root_classifier_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if device is None:
            device = torch.device(checkpoint["device"])

        return cls(
            model=model,
            projector=projector,
            root_classifier=root_classifier,
            optimizer=optimizer,
            device=device,
        )
