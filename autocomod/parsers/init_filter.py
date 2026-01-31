import ast
from pathlib import Path


class RepoHasInitDependencyError(Exception):
    def __init__(self, repo_path: Path, init_file: Path, imported_name: str):
        self.repo_path = repo_path
        self.init_file = init_file
        self.imported_name = imported_name
        super().__init__(
            f"Repo '{repo_path}': {init_file} defines '{imported_name}' which is imported elsewhere"
        )


class InitFileAnalyzer:
    """Analyzes __init__.py files to determine if they contain imported code."""

    @classmethod
    def get_defined_names(cls, file_path: Path) -> set[str]:
        """Returns names of functions and classes defined at the top level."""
        try:
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
        except SyntaxError:
            return set()

        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return names

    @classmethod
    def get_imports_from_module(cls, file_path: Path, module_name: str) -> set[str]:
        """
        Returns names imported from a specific module.

        For `from package import foo, bar` where module_name matches package,
        returns {'foo', 'bar'}.
        """
        try:
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
        except SyntaxError:
            return set()

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == module_name
                    or node.module.startswith(module_name + ".")
                    or module_name.endswith("." + (node.module or ""))
                ):
                    for alias in node.names:
                        if alias.name != "*":
                            imported_names.add(alias.name)
                        # For `from x import *`, we can't easily track what's imported
        return imported_names

    @classmethod
    def analyze_repo(cls, repo_path: Path) -> tuple[bool, list[Path], str | None]:
        """
        Analyzes __init__.py files and their dependencies.
        Returns: (should_skip, init_files_to_filter, reason)
        """
        init_files = list(repo_path.rglob("__init__.py"))
        all_py_files = list(repo_path.rglob("*.py"))

        if not init_files:
            return False, [], None

        # For each init file, get names it defines
        init_defined_names: dict[Path, set[str]] = {}
        for init_file in init_files:
            defined = cls.get_defined_names(init_file)
            if defined:
                init_defined_names[init_file] = defined

        # If no init files define any functions/classes, filter them all out
        if not init_defined_names:
            return False, init_files, None

        # Check if any defined name is imported by other files
        for init_file, defined_names in init_defined_names.items():
            # Get the module path for this init file (e.g., "package" or "package.sub")
            rel_path = init_file.parent.relative_to(repo_path)
            if str(rel_path) == ".":
                module_name = repo_path.name
            else:
                module_name = repo_path.name + "." + ".".join(rel_path.parts)

            # Check all other files for imports from this module
            for py_file in all_py_files:
                if py_file == init_file:
                    continue

                imported_names = cls.get_imports_from_module(py_file, module_name)

                # Check if any defined name is imported
                overlap = defined_names & imported_names
                if overlap:
                    return (
                        True,
                        [],
                        f"{init_file}: '{next(iter(overlap))}' imported by {py_file}",
                    )

        # No init-defined code is imported elsewhere - filter out all init files
        return False, init_files, None
