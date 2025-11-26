import ast
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

import jedi

from autocomod.logger import logger


class RepoParser:
    def __init__(self, repo_path: Path) -> None:
        self._repo_path = repo_path
        self._absolute_repo_path = repo_path.absolute()
        self._repo_name = repo_path.name
        self._files = self._collect_files()

    def compute_graph_data(self):
        return self._build_nodes(), self._build_edges()

    def _collect_files(self) -> List[Path]:
        return [path for path in self._repo_path.rglob("*.py")]

    def _path_to_module(self, file_path: Path) -> str:
        """Converts path/to/file.py to path.to.file module name"""

        if file_path.is_absolute():
            rel_path = file_path.relative_to(self._absolute_repo_path)
        else:
            rel_path = file_path.relative_to(self._repo_path)

        if str(rel_path) == ".":
            return str(rel_path)

        rel = rel_path.with_suffix("")
        return ".".join(rel.parts)

    @classmethod
    def _extract_imports(cls, path: Path):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except Exception:
            return [], []

        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_imports.append(node.module)

        return imports + from_imports

    def _build_nodes(self):
        """Create node data for each file."""

        nodes = {}
        for file_path in self._files:
            module = self._path_to_module(file_path.parent)
            imports = self._extract_imports(file_path)

            # Classify imports into internal/external
            internal = set()
            external = set()
            for imp in imports:
                if not imp:
                    continue
                elif imp.startswith(self._repo_name):
                    internal.add(imp)
                else:
                    external.add(imp.split(".")[0])

            nodes[self._path_to_module(file_path)] = {
                "module": module,
                "imports_internal": list(internal),
                "imports_external": list(external),
            }

        return nodes

    def _build_edges(self):
        """Use Jedi to infer call and instantiation relationships."""

        edges = {}
        for file_path in self._files:
            try:
                script = jedi.Script(path=str(file_path))
            except Exception:
                logger.warning(f"Failed to parse {file_path}")
                continue

            source = self._path_to_module(file_path)
            edges[source] = defaultdict(list)

            # Iterate over all names/expressions
            for name in script.get_names(all_scopes=True):
                if name.type == "statement":
                    continue

                # Infer the symbol referenced at this location
                try:
                    inferred = name.infer()
                except Exception:
                    continue

                for target in inferred:
                    if not (
                        target.is_definition()
                        and target.module_path is not None
                        and target.full_name is not None
                        and target.full_name.startswith(self._repo_name)
                    ):
                        continue

                    dest = self._path_to_module(Path(target.module_path))
                    if dest == source:
                        continue

                    # internal reference -> create edge
                    edges[source][dest].append(
                        {
                            "type": self._infer_edge_type(name.type),
                            "symbol": target.name,
                        }
                    )
        return edges

    @classmethod
    def _infer_edge_type(
        cls, jedi_type: str
    ) -> Literal["call", "instantiate", "attribute"]:
        if jedi_type == "function" or jedi_type == "call":
            return "call"
        if jedi_type == "class":
            return "instantiate"
        return "attribute"
