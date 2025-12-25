"""
Tests for visibility_atlas.paths module.

Tests cover:
- Project root detection via .project-root marker
- Path resolution relative to project root
- Paths singleton canonical path properties
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from visibility_atlas.paths import get_project_root, get_path, paths, ensure_dir


class TestGetProjectRoot:
    """Tests for get_project_root() function."""
    
    def test_finds_project_root(self):
        """Should find the project root containing .project-root."""
        root = get_project_root()
        assert root.exists()
        assert (root / ".project-root").exists()
    
    def test_returns_path_object(self):
        """Should return a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
    
    def test_is_absolute_path(self):
        """Should return an absolute path."""
        root = get_project_root()
        assert root.is_absolute()
    
    def test_result_is_cached(self):
        """Should return the same object on repeated calls (caching)."""
        root1 = get_project_root()
        root2 = get_project_root()
        # Should be the exact same object due to caching
        assert root1 is root2


class TestGetPath:
    """Tests for get_path() function."""
    
    def test_resolves_single_part(self):
        """Should resolve a single path component."""
        path = get_path("data")
        expected = get_project_root() / "data"
        assert path == expected
    
    def test_resolves_multiple_parts(self):
        """Should resolve multiple path components."""
        path = get_path("data", "processed", "geo")
        expected = get_project_root() / "data" / "processed" / "geo"
        assert path == expected
    
    def test_returns_absolute_path(self):
        """Should return an absolute path."""
        path = get_path("data")
        assert path.is_absolute()
    
    def test_returns_path_object(self):
        """Should return a Path object."""
        path = get_path("configs")
        assert isinstance(path, Path)


class TestPathsSingleton:
    """Tests for the Paths singleton instance."""
    
    def test_root_property(self):
        """paths.root should return project root."""
        assert paths.root == get_project_root()
    
    def test_configs_path(self):
        """paths.configs should point to configs directory."""
        assert paths.configs == get_project_root() / "configs"
    
    def test_data_raw_path(self):
        """paths.data_raw should point to data/raw directory."""
        assert paths.data_raw == get_project_root() / "data" / "raw"
    
    def test_data_processed_path(self):
        """paths.data_processed should point to data/processed directory."""
        assert paths.data_processed == get_project_root() / "data" / "processed"
    
    def test_processed_visibility_path(self):
        """paths.processed_visibility should point to visibility output directory."""
        assert paths.processed_visibility == get_project_root() / "data" / "processed" / "visibility"
    
    def test_logs_path(self):
        """paths.logs should point to logs directory."""
        assert paths.logs == get_project_root() / "logs"
    
    def test_reports_path(self):
        """paths.reports should point to reports directory."""
        assert paths.reports == get_project_root() / "reports"
    
    def test_config_file_paths(self):
        """Config file paths should point to correct YAML files."""
        assert paths.params_yml == get_project_root() / "configs" / "params.yml"
        assert paths.sources_yml == get_project_root() / "configs" / "sources.yml"
        assert paths.strata_yml == get_project_root() / "configs" / "strata.yml"


class TestEnsureDir:
    """Tests for ensure_dir() function."""
    
    def test_creates_directory(self, tmp_path):
        """Should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        result = ensure_dir(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_creates_nested_directories(self, tmp_path):
        """Should create nested directories."""
        nested = tmp_path / "level1" / "level2" / "level3"
        assert not nested.exists()
        
        result = ensure_dir(nested)
        
        assert nested.exists()
        assert result == nested
    
    def test_returns_existing_directory(self, tmp_path):
        """Should return existing directory without error."""
        existing = tmp_path / "existing"
        existing.mkdir()
        
        result = ensure_dir(existing)
        
        assert result == existing
        assert existing.exists()
    
    def test_accepts_string_path(self, tmp_path):
        """Should accept string paths."""
        new_dir = str(tmp_path / "string_path")
        
        result = ensure_dir(new_dir)
        
        assert isinstance(result, Path)
        assert result.exists()
    
    def test_returns_path_object(self, tmp_path):
        """Should always return a Path object."""
        result = ensure_dir(tmp_path / "test")
        assert isinstance(result, Path)


class TestPathsExist:
    """Tests that canonical paths exist in the project."""
    
    def test_project_root_has_marker(self):
        """Project root should contain .project-root marker."""
        assert (paths.root / ".project-root").exists()
    
    def test_configs_directory_exists(self):
        """configs/ directory should exist."""
        assert paths.configs.exists()
    
    def test_params_yml_exists(self):
        """configs/params.yml should exist."""
        assert paths.params_yml.exists()
    
    def test_sources_yml_exists(self):
        """configs/sources.yml should exist."""
        assert paths.sources_yml.exists()
    
    def test_data_directory_structure_exists(self):
        """Basic data directory structure should exist."""
        assert paths.data_raw.exists()
        assert paths.data_processed.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

