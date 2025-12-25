"""
Smoke tests for pipeline outputs.

These tests verify that:
- Expected output files exist
- Output files have expected structure
- Key metrics match documented values

Run with: pytest tests/ -v -m smoke
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd

from visibility_atlas.paths import paths


pytestmark = pytest.mark.smoke


class TestOutputFilesExist:
    """Verify expected output files exist."""
    
    def test_nta_canonical_exists(self):
        """NTA canonical boundaries should exist."""
        path = paths.processed_geo / "nta_canonical.parquet"
        assert path.exists(), f"Missing: {path}"
    
    def test_acs_denominators_exists(self):
        """ACS denominators should exist."""
        path = paths.processed_denominators / "acs_denominators.parquet"
        assert path.exists(), f"Missing: {path}"
    
    def test_visibility_long_exists(self):
        """Unified visibility table should exist."""
        path = paths.processed_visibility / "visibility_long.parquet"
        assert path.exists(), f"Missing: {path}"
    
    def test_source_correlations_exists(self):
        """Source correlations should exist."""
        path = paths.processed_matrix / "source_correlations.parquet"
        assert path.exists(), f"Missing: {path}"
    
    def test_typology_assignments_exists(self):
        """Typology assignments should exist."""
        path = paths.processed_typologies / "typology_assignments.parquet"
        assert path.exists(), f"Missing: {path}"
    
    def test_crosswalks_exist(self):
        """Population-weighted crosswalks should exist."""
        assert (paths.processed_xwalk / "tract_to_nta_pop_weights.parquet").exists()
        assert (paths.processed_xwalk / "zcta_to_nta_pop_weights.parquet").exists()
    
    def test_uhf42_crosswalk_exists(self):
        """Proper UHF42 spatial crosswalk should exist."""
        path = paths.processed_xwalk / "uhf42_to_nta_pop_weights.parquet"
        assert path.exists(), f"Missing: {path}"


class TestVisibilityLongStructure:
    """Verify visibility_long.parquet has expected structure."""
    
    @pytest.fixture
    def visibility_long(self):
        """Load visibility_long.parquet."""
        return pd.read_parquet(paths.processed_visibility / "visibility_long.parquet")
    
    def test_row_count(self, visibility_long):
        """Should have 753 rows (documented value)."""
        assert len(visibility_long) == 753, f"Expected 753 rows, got {len(visibility_long)}"
    
    def test_has_required_columns(self, visibility_long):
        """Should have all required schema columns."""
        required = [
            "geo_id", "stratum_id", "source_id", "time_window_id",
            "observed_count", "reference_pop", "visibility",
            "reliability_flag", "numerator_type"
        ]
        for col in required:
            assert col in visibility_long.columns, f"Missing column: {col}"
    
    def test_has_three_sources(self, visibility_long):
        """Should have exactly 3 sources: chs, sparcs, vital."""
        sources = set(visibility_long["source_id"].unique())
        expected = {"chs", "sparcs", "vital"}
        assert sources == expected, f"Expected {expected}, got {sources}"
    
    def test_minimal_missing_visibility(self, visibility_long):
        """Visibility column should have minimal missing values (< 1%)."""
        missing = visibility_long["visibility"].isna().sum()
        missing_pct = missing / len(visibility_long) * 100
        # Allow up to 1% missing (some cells may lack data due to crosswalk gaps)
        assert missing_pct < 1.0, f"Found {missing} missing visibility values ({missing_pct:.1f}%)"


class TestCorrelations:
    """Verify source correlations match documented values."""
    
    @pytest.fixture
    def correlations(self):
        """Load source correlations."""
        return pd.read_parquet(paths.processed_matrix / "source_correlations.parquet")
    
    def test_chs_sparcs_correlation(self, correlations):
        """CHS-SPARCS correlation should be r ≈ 0.122."""
        row = correlations[
            (correlations["source_1"] == "chs") & 
            (correlations["source_2"] == "sparcs")
        ]
        if len(row) == 0:
            row = correlations[
                (correlations["source_1"] == "sparcs") & 
                (correlations["source_2"] == "chs")
            ]
        
        assert len(row) == 1, "Should find exactly one CHS-SPARCS pair"
        r = row.iloc[0]["pearson_r"]
        assert abs(r - 0.122) < 0.01, f"Expected r ≈ 0.122, got {r}"
    
    def test_chs_vital_correlation(self, correlations):
        """CHS-Vital correlation should be r ≈ 0.237."""
        row = correlations[
            (correlations["source_1"] == "chs") & 
            (correlations["source_2"] == "vital")
        ]
        if len(row) == 0:
            row = correlations[
                (correlations["source_1"] == "vital") & 
                (correlations["source_2"] == "chs")
            ]
        
        assert len(row) == 1, "Should find exactly one CHS-Vital pair"
        r = row.iloc[0]["pearson_r"]
        assert abs(r - 0.237) < 0.01, f"Expected r ≈ 0.237, got {r}"
    
    def test_sparcs_vital_correlation(self, correlations):
        """SPARCS-Vital correlation should be r ≈ 0.817."""
        row = correlations[
            (correlations["source_1"] == "sparcs") & 
            (correlations["source_2"] == "vital")
        ]
        if len(row) == 0:
            row = correlations[
                (correlations["source_1"] == "vital") & 
                (correlations["source_2"] == "sparcs")
            ]
        
        assert len(row) == 1, "Should find exactly one SPARCS-Vital pair"
        r = row.iloc[0]["pearson_r"]
        assert abs(r - 0.817) < 0.01, f"Expected r ≈ 0.817, got {r}"
    
    def test_n_pairs_chs_sparcs(self, correlations):
        """CHS-SPARCS should have n=232 pairs."""
        row = correlations[
            ((correlations["source_1"] == "chs") & (correlations["source_2"] == "sparcs")) |
            ((correlations["source_1"] == "sparcs") & (correlations["source_2"] == "chs"))
        ]
        n = row.iloc[0]["n_pairs"]
        assert n == 232, f"Expected n=232, got {n}"


class TestTypologies:
    """Verify typology clustering outputs."""
    
    @pytest.fixture
    def typology_assignments(self):
        """Load typology assignments."""
        return pd.read_parquet(paths.processed_typologies / "typology_assignments.parquet")
    
    def test_has_six_clusters(self, typology_assignments):
        """Should have 6 typology clusters."""
        n_clusters = typology_assignments["cluster_id"].nunique()
        assert n_clusters == 6, f"Expected 6 clusters, got {n_clusters}"
    
    def test_stability_score_high(self, typology_assignments):
        """Mean stability should be ≈ 98.4%."""
        mean_stability = typology_assignments["stability_score"].mean()
        assert mean_stability > 0.95, f"Expected stability > 0.95, got {mean_stability}"
        assert abs(mean_stability - 0.984) < 0.02, f"Expected ≈ 0.984, got {mean_stability}"
    
    def test_has_typology_labels(self, typology_assignments):
        """All rows should have typology labels."""
        missing = typology_assignments["typology_label"].isna().sum()
        assert missing == 0, f"Found {missing} rows without typology labels"


class TestCrosswalks:
    """Verify crosswalk quality."""
    
    @pytest.fixture
    def uhf_crosswalk(self):
        """Load UHF42 crosswalk."""
        return pd.read_parquet(paths.processed_xwalk / "uhf42_to_nta_pop_weights.parquet")
    
    def test_weights_sum_to_one(self, uhf_crosswalk):
        """Weights should sum to ~1.0 per source geography."""
        weight_sums = uhf_crosswalk.groupby("source_geo_id")["weight"].sum()
        
        # Allow small tolerance for floating point
        assert (weight_sums > 0.99).all(), "Some UHF weights sum to < 0.99"
        assert (weight_sums < 1.01).all(), "Some UHF weights sum to > 1.01"
    
    def test_has_42_uhf_neighborhoods(self, uhf_crosswalk):
        """Should have approximately 42 UHF source neighborhoods."""
        n_uhf = uhf_crosswalk["source_geo_id"].nunique()
        # Allow for some exclusions (parks, etc.)
        assert 35 <= n_uhf <= 45, f"Expected ~42 UHF neighborhoods, got {n_uhf}"
    
    def test_weights_are_positive(self, uhf_crosswalk):
        """All weights should be positive."""
        assert (uhf_crosswalk["weight"] >= 0).all()
        assert (uhf_crosswalk["weight"] <= 1).all()


class TestACSDenominators:
    """Verify ACS denominators."""
    
    @pytest.fixture
    def acs_denominators(self):
        """Load ACS denominators."""
        return pd.read_parquet(paths.processed_denominators / "acs_denominators.parquet")
    
    def test_total_population_plausible(self, acs_denominators):
        """Total NYC population should be ~8.6 million."""
        total_df = acs_denominators[acs_denominators["stratum_id"] == "total"]
        total_pop = total_df["reference_pop"].sum()
        
        # NYC population should be 8-9 million
        assert 8_000_000 <= total_pop <= 9_500_000, f"Total pop {total_pop} outside expected range"
    
    def test_has_required_strata(self, acs_denominators):
        """Should have total and age strata."""
        strata = set(acs_denominators["stratum_id"].unique())
        assert "total" in strata
        # Should have at least some age strata
        age_strata = [s for s in strata if s.startswith("age_")]
        assert len(age_strata) >= 3, f"Expected at least 3 age strata, got {age_strata}"


class TestMetadataSidecars:
    """Verify metadata sidecars exist for key outputs."""
    
    def test_visibility_long_has_metadata(self):
        """visibility_long should have metadata sidecar."""
        metadata_path = paths.processed_visibility / "visibility_long_metadata.json"
        assert metadata_path.exists(), f"Missing metadata: {metadata_path}"
    
    def test_acs_denominators_has_metadata(self):
        """ACS denominators should have metadata sidecar."""
        metadata_path = paths.processed_denominators / "acs_denominators_metadata.json"
        assert metadata_path.exists(), f"Missing metadata: {metadata_path}"
    
    def test_metadata_has_run_id(self):
        """Metadata should contain run_id."""
        from visibility_atlas.io_utils import read_json
        metadata = read_json(paths.processed_visibility / "visibility_long_metadata.json")
        assert "run_id" in metadata, "Metadata missing run_id"
    
    def test_metadata_has_row_count(self):
        """Metadata should contain row_count."""
        from visibility_atlas.io_utils import read_json
        metadata = read_json(paths.processed_visibility / "visibility_long_metadata.json")
        assert "row_count" in metadata, "Metadata missing row_count"
        assert metadata["row_count"] == 753


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

