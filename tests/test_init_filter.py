import importlib.util
import tempfile
from pathlib import Path

_module_path = Path(__file__).parent.parent / "autocomod" / "parsers" / "init_filter.py"
_spec = importlib.util.spec_from_file_location("init_filter", _module_path)
_init_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_init_filter)
InitFileAnalyzer = _init_filter.InitFileAnalyzer


class TestGetDefinedNames:
    """Tests for InitFileAnalyzer.get_defined_names method."""

    def test_empty_file_has_no_defined_names(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == set()

    def test_imports_only_has_no_defined_names(self):
        content = """
import os
from pathlib import Path
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == set()

    def test_function_def_returns_name(self):
        content = """
def helper():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == {"helper"}

    def test_async_function_def_returns_name(self):
        """File with async function definition returns the function name."""
        content = """
async def fetch_data():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == {"fetch_data"}

    def test_class_def_returns_name(self):
        content = """
class Config:
    DEBUG = True
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == {"Config"}

    def test_multiple_definitions_returns_all_names(self):
        content = """
def foo():
    pass

class Bar:
    pass

async def baz():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == {
                "foo",
                "Bar",
                "baz",
            }

    def test_syntax_error_returns_empty_set(self):
        content = "def broken("
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            assert InitFileAnalyzer.get_defined_names(Path(f.name)) == set()


class TestAnalyzeRepo:
    """Tests for InitFileAnalyzer.analyze_repo method."""

    def test_repo_with_no_init_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "module.py").write_text("x = 1")

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is False
            assert init_files == []
            assert reason is None

    def test_repo_with_imports_only_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            init_file = repo / "__init__.py"
            init_file.write_text("from .module import Foo")
            (repo / "module.py").write_text("class Foo: pass")

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is False
            assert init_file in init_files
            assert reason is None

    def test_repo_with_init_code_not_imported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            init_file = repo / "__init__.py"
            init_file.write_text("def unused(): pass")
            (repo / "module.py").write_text("x = 1")  # Does not import 'unused'

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is False
            assert init_file in init_files
            assert reason is None

    def test_repo_with_init_code_imported_elsewhere_skips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "mypackage"
            repo.mkdir()

            init_file = repo / "__init__.py"
            init_file.write_text("def helper(): pass")

            # This file imports 'helper' from the package
            (repo / "module.py").write_text("from mypackage import helper")

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is True
            assert init_files == []
            assert reason is not None
            assert "helper" in reason

    def test_repo_with_nested_init_code_imported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "mypackage"
            repo.mkdir()

            # Root init - no code
            (repo / "__init__.py").write_text("from .sub import x")

            # Subpackage with code
            sub = repo / "sub"
            sub.mkdir()
            (sub / "__init__.py").write_text("def process(): pass")
            (sub / "module.py").write_text("x = 1")

            # Another file imports from sub
            (repo / "other.py").write_text("from mypackage.sub import process")

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is True
            assert init_files == []
            assert "process" in reason

    def test_repo_with_class_in_init_imported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "mypackage"
            repo.mkdir()

            init_file = repo / "__init__.py"
            init_file.write_text("class Config: pass")

            (repo / "module.py").write_text("from mypackage import Config")

            should_skip, init_files, reason = InitFileAnalyzer.analyze_repo(repo)

            assert should_skip is True
            assert init_files == []
            assert "Config" in reason
