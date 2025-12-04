import warnings

import networkx as nx
from networkx.algorithms.community import modularity
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    normalized_mutual_info_score,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class MetricsComputing:
    @classmethod
    def compute_all(
        cls, z: torch.Tensor, true_labels: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, float]:
        z_np = z.detach().cpu().numpy()
        true = true_labels.detach().cpu().numpy()
        edges = edge_index.t().detach().cpu().numpy()

        k = len(np.unique(true_labels))
        km = KMeans(n_clusters=k, n_init="auto")
        pred = km.fit_predict(z_np)

        ari_score = adjusted_rand_score(true, pred)
        nmi_score = normalized_mutual_info_score(true, pred)
        sil_score = cls.compute_silhouette(z_np, pred)
        mod_q = cls.compute_modularity_nx(edges, pred)
        se_metrics = cls.compute_se_metrics(edges, pred)

        return {
            "ari": ari_score,
            "nmi": nmi_score,
            "silhouette": sil_score,
            "modularity": mod_q,
            **se_metrics,
        }

    @classmethod
    def compute_ground_truth(
        cls, true_labels: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, float]:
        true = true_labels.detach().cpu().numpy()
        edges = edge_index.t().detach().cpu().numpy()

        mod_q = cls.compute_modularity_nx(edges, true)
        se_metrics = cls.compute_se_metrics(edges, true)

        return {
            "ari": 1.0,
            "nmi": 1.0,
            "modularity": mod_q,
            **se_metrics,
        }

    @classmethod
    def compute_ari(cls, pred: np.ndarray, true_labels: np.ndarray):
        ari = adjusted_rand_score(true_labels, pred)

        return ari, pred

    @classmethod
    def compute_silhouette(cls, z: np.ndarray, cluster_labels: np.ndarray):
        if len(np.unique(cluster_labels)) < 2:
            sil = -1.0
        else:
            sil = silhouette_score(z, cluster_labels)

        return sil

    @classmethod
    def compute_modularity_nx(cls, edges: np.ndarray, pred: np.ndarray):
        G = nx.DiGraph()
        G.add_nodes_from(range(pred.shape[0]))
        G.add_edges_from(edges)

        communities = [set(np.where(pred == c)[0]) for c in np.unique(pred)]

        return modularity(G, communities)

    @classmethod
    def compute_se_metrics(cls, edges: np.ndarray, pred: np.ndarray):
        """
        Compute cohesion and instability per module,
        based on a predicted clustering assignment.
        """

        src, dst = edges.T
        modules = np.unique(pred)
        module_nodes = {m: np.where(pred == m)[0] for m in modules}

        # Storage
        cohesion = {}
        instability = {}

        # Convert edges to module-level edges
        for m in modules:
            nodes = module_nodes[m]

            # Mask edges where src or dst is in this module
            mask_src = np.isin(src, nodes)
            mask_dst = np.isin(dst, nodes)

            edges_in_module = mask_src & mask_dst
            edges_outgoing = mask_src & ~mask_dst
            edges_incoming = ~mask_src & mask_dst

            internal_edges = edges_in_module.sum()
            outgoing_edges = edges_outgoing.sum()
            incoming_edges = edges_incoming.sum()

            # Cohesion = internal edges / (total nodes in cluster)^2
            size = len(nodes)
            cohesion[m] = internal_edges / (size * size + 1e-9)

            # Instability: I = outgoing / (incoming + outgoing)
            denom = outgoing_edges + incoming_edges
            instability[m] = outgoing_edges / (denom + 1e-9)

        return {
            "avg_cohesion": float(np.mean(list(cohesion.values()))),
            "avg_instability": float(np.mean(list(instability.values()))),
        }
