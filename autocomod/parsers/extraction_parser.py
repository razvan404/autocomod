from typing import Any


class ExtractionParser:
    def __init__(self, repo_data: dict[str, Any]) -> None:
        self._nodes_json = repo_data["nodes"]
        self._edges_json = repo_data["edges"]

        # Filled during processing:
        self.node_names: list[str] = []
        self.node_to_idx: dict[str, int] = {}
        self.internal_sets = {}
        self.external_sets = {}

    def compute_processed_data(self) -> dict[str, Any]:
        """
        Main entry point: produce a unified graph object
        containing raw features + processed edge data.
        """

        self._collect_nodes()

        # augment imports using edge connectivity
        self._augment_internal_imports_with_edges()

        self._build_node_feature_sets()

        labels = self._build_node_labels()
        edge_index, edge_attr_raw = self._build_edges_with_features()

        node_struct = self._build_node_structural_features(edge_index, edge_attr_raw)
        x_raw = self._build_raw_node_features(node_struct)

        return {
            "node_names": self.node_names,
            "node_to_idx": self.node_to_idx,
            "x_raw": x_raw,
            "labels": labels,
            "edge_index": edge_index,
            "edge_attr_raw": edge_attr_raw,
        }

    def _collect_nodes(self):
        """Assign deterministic node ordering."""
        self.node_names = sorted(self._nodes_json.keys())
        self.node_to_idx = {name: i for i, name in enumerate(self.node_names)}

    def _augment_internal_imports_with_edges(self):
        """
        Some internal imports are only visible in the edges (Jedi inference),
        not in the AST-collected imports.

        For each edge (src -> dst), ensure dst is listed in src.imports_internal.
        """

        for src_name, dst_dict in self._edges_json.items():
            if src_name not in self._nodes_json:
                continue

            for dst_name in dst_dict.keys():
                # Only augment if this is a real node
                if dst_name not in self._nodes_json:
                    continue

                internal = self._nodes_json[src_name]["imports_internal"]

                # Add only if it's not already present
                if dst_name not in internal:
                    internal.append(dst_name)

    def _build_node_feature_sets(self):
        """Precompute sets of internal & external imports per node."""
        self.internal_sets = {
            name: set(self._nodes_json[name]["imports_internal"])
            for name in self.node_names
        }
        self.external_sets = {
            name: set(self._nodes_json[name]["imports_external"])
            for name in self.node_names
        }

    def _build_node_labels(self) -> list[str]:
        """Module name per node (raw)."""
        labels = []
        for name in self.node_names:
            info = self._nodes_json[name]
            module = info["module"]
            if module == ".":
                labels.append(module)
                continue
            labels.append(module.split(".", 1)[0])
        return labels

    def _compute_shared_imports(self, src: str, dst: str) -> tuple[int, int]:
        """Return (#shared_internal, #shared_external)."""
        shared_internal = len(
            self.internal_sets[src].intersection(self.internal_sets[dst])
        )
        shared_external = len(
            self.external_sets[src].intersection(self.external_sets[dst])
        )
        return shared_internal, shared_external

    def _build_edges_with_features(self):
        """
        Build unique aggregated edges.
        For each (src -> dst):
            - compute shared imports
            - count call / instantiate / attribute usages
            - check direct_import based on AST imports from nodes data
        """

        edge_index = []
        edge_attr_raw = []

        for src_name, dst_dict in self._edges_json.items():
            if src_name not in self.node_to_idx:
                continue

            src_idx = self.node_to_idx[src_name]

            for dst_name, usages in dst_dict.items():
                if dst_name not in self.node_to_idx:
                    continue

                dst_idx = self.node_to_idx[dst_name]

                shared_internal, shared_external = self._compute_shared_imports(
                    src_name, dst_name
                )

                # aggregate inferred usage types
                num_call = 0
                num_instantiate = 0
                num_attribute = 0

                for usage in usages:
                    t = usage["type"]
                    if t == "call":
                        num_call += 1
                    elif t == "instantiate":
                        num_instantiate += 1
                    elif t == "attribute":
                        num_attribute += 1

                # append unique edge
                edge_index.append((src_idx, dst_idx))
                edge_attr_raw.append(
                    {
                        "shared_internal": shared_internal,
                        "shared_external": shared_external,
                        "num_call": num_call,
                        "num_instantiate": num_instantiate,
                        "num_attribute": num_attribute,
                    }
                )

        return edge_index, edge_attr_raw

    def _build_node_structural_features(
        self, edge_index: list[tuple[int, int]], edge_attr_raw: list[dict]
    ):
        """Compute node-level structural statistics from edges."""

        # initialize
        struct = {
            name: {
                "in_degree": 0,
                "out_degree": 0,
                "num_calls_out": 0,
                "num_instantiate_out": 0,
                "num_attribute_out": 0,
                "num_imports_internal": len(self._nodes_json[name]["imports_internal"]),
                "num_imports_external": len(self._nodes_json[name]["imports_external"]),
            }
            for name in self.node_names
        }

        # accumulate edge statistics
        for (src_idx, dst_idx), attr in zip(edge_index, edge_attr_raw):
            src = self.node_names[src_idx]
            dst = self.node_names[dst_idx]

            # degrees
            struct[src]["out_degree"] += 1
            struct[dst]["in_degree"] += 1

            # usage-level counts
            struct[src]["num_calls_out"] += attr["num_call"]
            struct[src]["num_instantiate_out"] += attr["num_instantiate"]
            struct[src]["num_attribute_out"] += attr["num_attribute"]

        # compute total degree
        for name in self.node_names:
            s = struct[name]
            s["total_degree"] = s["in_degree"] + s["out_degree"]

        return struct

    def _build_raw_node_features(self, node_struct: dict[str, int]):
        """Combine import info + structural info."""
        x_raw = []
        for name in self.node_names:
            info = self._nodes_json[name]
            struct = node_struct[name]

            x_raw.append(
                {
                    "imports_internal": info["imports_internal"],
                    "imports_external": info["imports_external"],
                    **struct,
                }
            )
        return x_raw
