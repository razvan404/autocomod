from typing import Any, Dict, List
import torch
from torch_geometric.data import Data


class PygParser:
    NODE_FEATURE_KEYS = [
        "in_degree",
        "out_degree",
        "total_degree",
        "num_imports_internal",
        "num_imports_external",
        "num_calls_out",
        "num_instantiate_out",
        "num_attribute_out",
    ]

    EDGE_FEATURE_KEYS = [
        "shared_internal",
        "shared_external",
        "num_call",
        "num_instantiate",
        "num_attribute",
    ]

    def __init__(self, graph_dict: Dict[str, Any]) -> None:
        self.graph_dict = graph_dict

        self.node_names: List[str] = graph_dict["node_names"]
        self.x_raw: List[Dict[str, Any]] = graph_dict["x_raw"]
        self.labels: List[str] = graph_dict["labels"]
        self.edge_index_raw = graph_dict["edge_index"]
        self.edge_attr_raw = graph_dict["edge_attr_raw"]

    def to_pyg_data(self, **kwargs) -> Data:
        """Convert this graph dictionary to a PyTorch Geometric Data object."""

        x = self._build_node_feature_tensor()
        y, labels = self._encode_labels()
        edge_index = self._build_edge_index_tensor()
        edge_attr = self._build_edge_attr_tensor()

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )

        # Save metadata inside the Data object for debugging / visualization
        data._metadata = {
            "node_names": self.node_names,
            "labels": labels,
            **kwargs,
        }

        return data

    def _build_node_feature_tensor(self) -> torch.Tensor:
        features = []
        for node_feat in self.x_raw:
            row = [float(node_feat[key]) for key in self.NODE_FEATURE_KEYS]
            features.append(row)

        return torch.tensor(features, dtype=torch.float)

    def _encode_labels(self):
        unique_labels = sorted(set(self.labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

        y = torch.tensor([label_to_idx[lab] for lab in self.labels], dtype=torch.long)
        return y, unique_labels

    def _build_edge_index_tensor(self) -> torch.Tensor:
        edge_index = (
            torch.tensor(self.edge_index_raw, dtype=torch.long).t().contiguous()
        )
        return edge_index

    def _build_edge_attr_tensor(self) -> torch.Tensor:
        features = []
        for edge_feat in self.edge_attr_raw:
            row = [float(edge_feat[key]) for key in self.EDGE_FEATURE_KEYS]
            features.append(row)

        return torch.tensor(features, dtype=torch.float)
