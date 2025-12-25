"""
Pytest configuration and shared fixtures.

This module provides common fixtures used across test modules.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    from visibility_atlas.paths import get_project_root
    return get_project_root()


@pytest.fixture(scope="session")
def params_config():
    """Load the params.yml configuration."""
    from visibility_atlas.io_utils import read_yaml
    from visibility_atlas.paths import paths
    return read_yaml(paths.params_yml)


@pytest.fixture(scope="session")
def sources_config():
    """Load the sources.yml configuration."""
    from visibility_atlas.io_utils import read_yaml
    from visibility_atlas.paths import paths
    return read_yaml(paths.sources_yml)


@pytest.fixture
def sample_visibility_df():
    """Create a sample visibility DataFrame for testing."""
    return pd.DataFrame({
        "geo_id": ["BK0101", "BK0102", "MN0101", "QN0101", "SI0101"],
        "stratum_id": ["total"] * 5,
        "time_window_id": ["2019"] * 5,
        "source_id": ["chs"] * 5,
        "observed_count": [50.0, 25.0, 100.0, 10.0, 5.0],
        "reference_pop": [10000.0, 8000.0, 20000.0, 5000.0, 3000.0],
        "visibility": [5.0, 3.125, 5.0, 2.0, 1.67],
        "reliability_flag": ["high", "high", "high", "low", "suppressed"],
        "numerator_type": ["respondents"] * 5,
    })


@pytest.fixture
def sample_crosswalk_df():
    """Create a sample crosswalk DataFrame for testing."""
    return pd.DataFrame({
        "source_geo_id": ["101", "101", "102", "102", "103"],
        "target_geo_id": ["BK0101", "BK0102", "MN0101", "MN0102", "QN0101"],
        "weight": [0.6, 0.4, 0.7, 0.3, 1.0],
    })


@pytest.fixture
def sample_acs_df():
    """Create a sample ACS denominators DataFrame for testing."""
    return pd.DataFrame({
        "geo_id": ["BK0101", "BK0101", "BK0102", "MN0101"],
        "stratum_id": ["total", "age_18_24", "total", "total"],
        "time_window_id": ["2018_2022"] * 4,
        "reference_pop": [50000.0, 5000.0, 60000.0, 100000.0],
        "reference_pop_moe": [1000.0, 500.0, 1200.0, 2000.0],
        "reference_pop_se": [607.3, 303.6, 729.0, 1215.8],
    })


@pytest.fixture
def temp_parquet_file(tmp_path):
    """Create a temporary parquet file path."""
    return tmp_path / "test_output.parquet"


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file path."""
    return tmp_path / "test_output.json"


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test (quick sanity checks)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take > 10 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires full pipeline data)"
    )

