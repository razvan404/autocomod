import importlib.util
import tempfile
from pathlib import Path

import pytest

_filter_path = Path(__file__).parent.parent / "autocomod" / "parsers" / "init_filter.py"
_filter_spec = importlib.util.spec_from_file_location("init_filter", _filter_path)
_init_filter = importlib.util.module_from_spec(_filter_spec)
_filter_spec.loader.exec_module(_init_filter)
RepoHasInitDependencyError = _init_filter.RepoHasInitDependencyError

_parser_path = Path(__file__).parent.parent / "autocomod" / "parsers" / "repo_parser.py"
_parser_spec = importlib.util.spec_from_file_location("repo_parser", _parser_path)
_repo_parser = importlib.util.module_from_spec(_parser_spec)
_parser_spec.loader.exec_module(_repo_parser)
RepoParser = _repo_parser.RepoParser


class TestRepoParserInitFiltering:
    """Tests for RepoParser's __init__.py filtering behavior."""

    def test_init_files_excluded_from_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "myrepo"
            repo.mkdir()

            # Create an imports-only init
            (repo / "__init__.py").write_text("from .module import Foo")
            # Create a regular module
            (repo / "module.py").write_text("class Foo: pass")

            parser = RepoParser(repo)

            file_names = [f.name for f in parser._files]
            assert "__init__.py" not in file_names
            assert "module.py" in file_names

    def test_repo_with_init_code_not_imported_filters_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "myrepo"
            repo.mkdir()

            # Init with unused function
            (repo / "__init__.py").write_text("def unused(): pass")
            (repo / "module.py").write_text("x = 1")

            parser = RepoParser(repo)

            file_names = [f.name for f in parser._files]
            assert "__init__.py" not in file_names
            assert "module.py" in file_names

    def test_repo_with_init_code_imported_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "mypackage"
            repo.mkdir()

            # Init with function that gets imported
            (repo / "__init__.py").write_text("def helper(): pass")
            (repo / "module.py").write_text("from mypackage import helper")

            with pytest.raises(Exception) as exc_info:
                RepoParser(repo)

            # Check exception type by name (importlib creates separate class)
            assert "RepoHasInitDependencyError" in type(exc_info.value).__name__

    def test_repo_without_init_files_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "myrepo"
            repo.mkdir()

            (repo / "module.py").write_text("x = 1")
            (repo / "other.py").write_text("y = 2")

            parser = RepoParser(repo)

            file_names = [f.name for f in parser._files]
            assert "module.py" in file_names
            assert "other.py" in file_names

    def test_nodes_exclude_filtered_init_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "myrepo"
            repo.mkdir()

            # Create an imports-only init
            (repo / "__init__.py").write_text("from .module import Foo")
            # Create regular modules
            (repo / "module.py").write_text("class Foo: pass")
            (repo / "other.py").write_text("from .module import Foo")

            parser = RepoParser(repo)
            nodes, _ = parser.compute_graph_data()

            node_names = list(nodes.keys())
            assert not any("__init__" in name for name in node_names)
            assert any("module" in name for name in node_names)
            assert any("other" in name for name in node_names)

    def test_edges_exclude_filtered_init_files(self):
        """Built edges should not have filtered __init__ files as destinations.

        When code references a package as a namespace (e.g., `myrepo.pkg`),
        Jedi resolves this to the __init__.py file. These should be filtered out.

        This is an integration test that requires the zipline repo to be present.
        """
        repo = Path("repos/zipline/zipline")
        if not repo.exists():
            pytest.skip("zipline repo not available")

        parser = RepoParser(repo)
        _, edges = parser.compute_graph_data()

        # Collect all edge destinations
        all_destinations = set()
        for source_edges in edges.values():
            all_destinations.update(source_edges.keys())

        # No destination should contain __init__
        init_destinations = [d for d in all_destinations if "__init__" in d]
        assert (
            init_destinations == []
        ), f"Found __init__ in edge destinations: {init_destinations}"
