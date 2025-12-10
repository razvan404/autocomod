from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from .contrastive_loss import supervised_contrastive_loss


class GnnModuleClassificationPipeline:
    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        self.model = model
        self.projector = projector
        self.optimizer = optimizer
        self.device = device

    @classmethod
    def create_subgraph_with_holdout(cls, data: Data) -> tuple[Data, torch.Tensor]:
        """
        Create a subgraph by holding out one node per module.

        Returns:
            subgraph: Data object with held-out nodes removed
            held_out_mask: Boolean tensor indicating which nodes were held out
        """
        labels = data.y.numpy()
        unique_labels = np.unique(labels)
        num_nodes = data.x.shape[0]

        held_out_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 1:
                held_out_idx = np.random.choice(label_indices)
                held_out_indices.append(held_out_idx)

        held_out_mask = torch.zeros(num_nodes, dtype=torch.bool)
        held_out_mask[held_out_indices] = True

        keep_mask = ~held_out_mask
        keep_indices = torch.where(keep_mask)[0]

        old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
        old_to_new[keep_indices] = torch.arange(len(keep_indices))

        new_x = data.x[keep_mask]
        new_y = data.y[keep_mask]

        src, dst = data.edge_index
        edge_mask = keep_mask[src] & keep_mask[dst]
        new_edge_index = old_to_new[data.edge_index[:, edge_mask]]

        new_edge_attr = None
        if data.edge_attr is not None:
            new_edge_attr = data.edge_attr[edge_mask]

        subgraph = Data(
            x=new_x,
            y=new_y,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
        )

        return subgraph, held_out_mask

    def train_step(self, data: Data):
        """Train on a single graph with one node per module held out."""
        self.model.train()
        self.projector.train()

        subgraph, _ = self.create_subgraph_with_holdout(data)
        subgraph = subgraph.to(self.device)

        z = self.model(subgraph.x, subgraph.edge_index)
        h = self.projector(z)
        loss = supervised_contrastive_loss(h, subgraph.y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"contrastive": loss.item()}

    def validate_step(self, data: Data) -> dict[str, float]:
        """
        Validate on a full graph.

        Creates a subgraph, gets embeddings for full graph,
        clusters them, and checks if held-out nodes are in correct clusters.
        """
        self.model.eval()
        self.projector.eval()

        _, held_out_mask = self.create_subgraph_with_holdout(data)
        data = data.to(self.device)

        with torch.no_grad():
            z = self.model(data.x, data.edge_index)
            h = self.projector(z)

        h_np = h.cpu().numpy()
        labels = data.y.cpu().numpy()
        held_out_mask_np = held_out_mask.numpy()

        k = len(np.unique(labels))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_assignments = km.fit_predict(h_np)

        held_out_indices = np.where(held_out_mask_np)[0]
        correct_50 = 0
        correct_75 = 0
        correct_90 = 0
        total = len(held_out_indices)

        for idx in held_out_indices:
            true_label = labels[idx]
            predicted_cluster = cluster_assignments[idx]

            # Find non-held-out nodes from the same module
            same_module_mask = (labels == true_label) & (~held_out_mask_np)
            same_module_count = same_module_mask.sum()

            if same_module_count == 0:
                continue

            # Count how many module-mates are in the same cluster as held-out node
            same_cluster_mask = cluster_assignments == predicted_cluster
            module_mates_in_cluster = (same_module_mask & same_cluster_mask).sum()
            ratio = module_mates_in_cluster / same_module_count

            if ratio >= 0.50:
                correct_50 += 1
            if ratio >= 0.75:
                correct_75 += 1
            if ratio >= 0.9:
                correct_90 += 1

        return {
            "held_out_accuracy_50": correct_50 / total if total > 0 else 0.0,
            "held_out_accuracy_75": correct_75 / total if total > 0 else 0.0,
            "held_out_accuracy_90": correct_90 / total if total > 0 else 0.0,
            "held_out_correct_50": correct_50,
            "held_out_correct_75": correct_75,
            "held_out_correct_90": correct_90,
            "held_out_total": total,
        }

    def train_epoch(self, dataset) -> dict[str, float]:
        """Train on all graphs in the dataset."""
        metrics = defaultdict(float)
        for data in dataset:
            step_metrics = self.train_step(data)
            for key, value in step_metrics.items():
                metrics[key] += value
        for key in metrics:
            metrics[key] /= len(dataset)
        return dict(metrics)

    def validate_epoch(self, dataset) -> dict[str, float]:
        """Validate on all graphs in the dataset."""
        totals = defaultdict(float)

        for data in dataset:
            step_metrics = self.validate_step(data)
            for key, value in step_metrics.items():
                totals[key] += value

        result = dict(totals)
        total_held_out = totals.get("held_out_total", 0)

        for key in list(totals.keys()):
            if key.startswith("held_out_correct_"):
                suffix = key.replace("held_out_correct_", "")
                accuracy_key = f"held_out_accuracy_{suffix}"
                result[accuracy_key] = (
                    totals[key] / total_held_out if total_held_out > 0 else 0.0
                )

        return result

    def save(self, filepath: str) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
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
        optimizer: optim.Optimizer,
        device: torch.device | None = None,
    ) -> "GnnModuleClassificationPipeline":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        projector.load_state_dict(checkpoint["projector_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if device is None:
            device = torch.device(checkpoint["device"])

        return cls(
            model=model,
            projector=projector,
            optimizer=optimizer,
            device=device,
        )
