from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from autocomod.graph_nn import (
    GnnEncoder,
    ProjectionHead,
    ClassificationHead,
    GnnModularisationPipeline,
)
from autocomod.graphs_dataset import GraphsDataset
from autocomod.logger import logger
from autocomod.settings import Settings


class GnnTrainerSettings(Settings):
    experiment_name: str = "base_gnn"
    repos_file: str = "data/repos.csv"
    pyg_dir: str = "data/pyg_graphs"
    checkpoints_dir: str = "checkpoints"
    continue_training: bool = False
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 1
    hidden_dim: int = 128
    out_dim: int = 64
    validation_frequency: int = 10


def train_gnn(settings: GnnTrainerSettings):
    # Setup checkpoint directory
    experiment_dir = Path(settings.checkpoints_dir) / settings.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_dir / "model.pt"
    train_metrics_path = experiment_dir / "train.json"
    validation_metrics_path = experiment_dir / "validation.json"

    # Load datasets
    repos = pd.read_csv(settings.repos_file)

    train_repos = repos[repos.split == "train"]
    train_files = [f"{repo.id}.pt" for _, repo in train_repos.iterrows()]

    validation_repos = repos[repos.split == "validation"]
    validation_files = [f"{repo.id}.pt" for _, repo in validation_repos.iterrows()]

    train_dataset = GraphsDataset(settings.pyg_dir, train_files)
    validation_dataset = GraphsDataset(settings.pyg_dir, validation_files)

    train_loader = DataLoader(
        train_dataset, batch_size=settings.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=settings.batch_size, shuffle=False
    )

    # Get input dimension from first sample
    in_dim = train_dataset[0].x.shape[1]

    # Initialize models
    model = GnnEncoder(
        in_channels=in_dim, hidden=settings.hidden_dim, out_channels=settings.out_dim
    )
    projector = ProjectionHead(in_dim=settings.out_dim, proj_dim=settings.out_dim)
    root_classifier = ClassificationHead(in_channels=settings.out_dim)

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(projector.parameters())
        + list(root_classifier.parameters()),
        lr=settings.learning_rate,
    )

    # Load checkpoint if continuing training
    start_epoch = 1
    train_history = defaultdict(list)
    validation_history = defaultdict(list)

    if settings.continue_training and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        pipeline = GnnModularisationPipeline.load(
            str(checkpoint_path), model, projector, root_classifier, optimizer
        )

        # Load existing metrics
        if train_metrics_path.exists():
            with open(train_metrics_path) as f:
                train_history = json.load(f)
                start_epoch = len(train_history["epoch"]) + 1

        if validation_metrics_path.exists():
            with open(validation_metrics_path) as f:
                validation_history = json.load(f)

        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        pipeline = GnnModularisationPipeline(
            model, projector, root_classifier, optimizer
        )
        logger.info("Starting training from scratch")

    # Training loop
    for epoch in range(start_epoch, settings.num_epochs + 1):
        train_metrics = pipeline.train(train_loader)

        # Append train metrics
        train_history["epoch"].append(epoch)
        for key, value in train_metrics.items():
            train_history[key].append(value)

        # Validation
        if epoch % settings.validation_frequency == 0:
            logger.info(f"Training metrics at epoch {epoch}: {dict(train_metrics)}")

            val_metrics = pipeline.validate(validation_loader)

            # Append validation metrics
            validation_history["epoch"].append(epoch)
            for key, value in val_metrics.items():
                validation_history[key].append(value)

            logger.info(f"Validation metrics at epoch {epoch}: {dict(val_metrics)}")

        # Save checkpoint and metrics
        pipeline.save(str(checkpoint_path))
        with open(train_metrics_path, "w") as f:
            json.dump(train_history, f, indent=2)
        with open(validation_metrics_path, "w") as f:
            json.dump(validation_history, f, indent=2)

    logger.info(f"Training completed. Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    train_gnn(settings=GnnTrainerSettings())
