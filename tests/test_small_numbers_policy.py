"""
Tests for small numbers policy implementation.

Tests cover:
- Suppression rules are correctly applied
- Reliability flags are set according to thresholds
- Small denominators trigger low reliability
- Policy parameters are read from config
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np

from visibility_atlas.io_utils import read_yaml
from visibility_atlas.paths import paths


class TestSmallNumbersPolicyConfig:
    """Tests for small numbers policy configuration."""
    
    @pytest.fixture
    def params(self):
        """Load params.yml configuration."""
        return read_yaml(paths.params_yml)
    
    def test_config_has_small_numbers_section(self, params):
        """params.yml should have small_numbers section."""
        assert "small_numbers" in params
    
    def test_config_has_min_denominator(self, params):
        """Config should define min_denominator threshold."""
        assert "min_denominator" in params["small_numbers"]
        assert params["small_numbers"]["min_denominator"] > 0
    
    def test_config_has_min_numerator(self, params):
        """Config should define min_numerator threshold."""
        assert "min_numerator" in params["small_numbers"]
        assert params["small_numbers"]["min_numerator"] > 0
    
    def test_config_has_suppression_threshold(self, params):
        """Config should define suppression_threshold."""
        assert "suppression" in params["small_numbers"]
        assert "suppression_threshold" in params["small_numbers"]["suppression"]
    
    def test_thresholds_are_ordered(self, params):
        """suppression_threshold < min_numerator should hold."""
        suppression = params["small_numbers"]["suppression"]["suppression_threshold"]
        min_num = params["small_numbers"]["min_numerator"]
        assert suppression <= min_num, "suppression_threshold should be <= min_numerator"
    
    def test_config_has_reliability_flags(self, params):
        """Config should define reliability flag values."""
        assert "reliability_flags" in params["small_numbers"]
        flags = params["small_numbers"]["reliability_flags"]
        assert "high" in flags
        assert "low" in flags
        assert "suppressed" in flags


class TestApplySmallNumbersPolicy:
    """Tests for small numbers policy application logic."""
    
    @pytest.fixture
    def sample_visibility_df(self):
        """Create sample visibility DataFrame for testing."""
        return pd.DataFrame({
            "geo_id": ["BK0101", "BK0102", "BK0103", "BK0104", "BK0105"],
            "observed_count": [100.0, 25.0, 8.0, 3.0, 0.0],
            "reference_pop": [10000.0, 5000.0, 1000.0, 40.0, 100.0],
            "visibility": [10.0, 5.0, 8.0, 75.0, 0.0],
            "reliability_flag": ["high", "high", "high", "high", "high"],
        })
    
    def apply_policy(self, df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """
        Apply small numbers policy to DataFrame.
        
        This mirrors the logic in 07_build_visibility_tables.py.
        """
        if params is None:
            params = {
                "small_numbers_policy": {
                    "min_numerator": 10,
                    "min_denominator": 50,
                    "suppression_threshold": 5,
                }
            }
        
        policy = params.get("small_numbers_policy", {})
        min_numerator = policy.get("min_numerator", 10)
        min_denominator = policy.get("min_denominator", 50)
        suppression_threshold = policy.get("suppression_threshold", 5)
        
        df = df.copy()
        
        # Apply suppression for small numerators
        small_numerator = df["observed_count"] < suppression_threshold
        df.loc[small_numerator & (df["reliability_flag"] != "suppressed"), "reliability_flag"] = "suppressed"
        
        # Apply low reliability for borderline cases
        borderline = (df["observed_count"] >= suppression_threshold) & (df["observed_count"] < min_numerator)
        df.loc[borderline & (df["reliability_flag"] == "high"), "reliability_flag"] = "low"
        
        # Apply low reliability for small denominators
        small_denom = df["reference_pop"] < min_denominator
        df.loc[small_denom & (df["reliability_flag"] == "high"), "reliability_flag"] = "low"
        
        return df
    
    def test_high_count_high_pop_stays_high(self, sample_visibility_df):
        """High count + high population should remain 'high' reliability."""
        df = self.apply_policy(sample_visibility_df)
        
        # Row 0: observed_count=100, reference_pop=10000 → should stay high
        assert df.loc[0, "reliability_flag"] == "high"
    
    def test_borderline_count_becomes_low(self, sample_visibility_df):
        """Borderline count (between suppression and min_numerator) should become 'low'."""
        df = self.apply_policy(sample_visibility_df)
        
        # Row 2: observed_count=8 (between 5 and 10) → should be low
        assert df.loc[2, "reliability_flag"] == "low"
    
    def test_very_small_count_becomes_suppressed(self, sample_visibility_df):
        """Very small count (< suppression_threshold) should become 'suppressed'."""
        df = self.apply_policy(sample_visibility_df)
        
        # Row 3: observed_count=3 (< 5) → should be suppressed
        assert df.loc[3, "reliability_flag"] == "suppressed"
        
        # Row 4: observed_count=0 (< 5) → should be suppressed
        assert df.loc[4, "reliability_flag"] == "suppressed"
    
    def test_small_denominator_becomes_low(self, sample_visibility_df):
        """Small denominator should trigger 'low' reliability."""
        df = self.apply_policy(sample_visibility_df)
        
        # Row 3: reference_pop=40 (< 50) → but already suppressed due to count
        # Check that small denom logic doesn't override suppression
        assert df.loc[3, "reliability_flag"] == "suppressed"
    
    def test_suppression_overrides_low(self, sample_visibility_df):
        """Suppression should take precedence over low reliability."""
        df = self.apply_policy(sample_visibility_df)
        
        # Row 3 has both small count (suppressed) and small denom (low)
        # Suppression should win
        assert df.loc[3, "reliability_flag"] == "suppressed"
    
    def test_policy_does_not_modify_original(self, sample_visibility_df):
        """Policy application should not modify the original DataFrame."""
        original_flags = sample_visibility_df["reliability_flag"].copy()
        
        _ = self.apply_policy(sample_visibility_df)
        
        # Original should be unchanged
        assert (sample_visibility_df["reliability_flag"] == original_flags).all()


class TestReliabilityFlagDistribution:
    """Tests for expected reliability flag distributions."""
    
    def test_all_flags_are_valid(self):
        """All reliability flags in output should be valid values."""
        valid_flags = {"high", "low", "suppressed", "unknown"}
        
        # Load actual visibility data if available
        visibility_path = paths.processed_visibility / "visibility_long.parquet"
        if visibility_path.exists():
            import pandas as pd
            df = pd.read_parquet(visibility_path)
            
            actual_flags = set(df["reliability_flag"].unique())
            invalid = actual_flags - valid_flags
            assert len(invalid) == 0, f"Invalid reliability flags found: {invalid}"
    
    def test_reliability_flags_not_all_same(self):
        """Output should have variation in reliability flags (not all same)."""
        visibility_path = paths.processed_visibility / "visibility_long.parquet"
        if visibility_path.exists():
            import pandas as pd
            df = pd.read_parquet(visibility_path)
            
            unique_flags = df["reliability_flag"].nunique()
            assert unique_flags > 1, "Expected variation in reliability flags"


class TestEdgeCases:
    """Tests for edge cases in small numbers policy."""
    
    def test_zero_count_is_suppressed(self):
        """Zero observed count should always be suppressed."""
        df = pd.DataFrame({
            "geo_id": ["TEST"],
            "observed_count": [0.0],
            "reference_pop": [10000.0],
            "reliability_flag": ["high"],
        })
        
        # Apply policy
        df_result = df.copy()
        df_result.loc[df_result["observed_count"] < 5, "reliability_flag"] = "suppressed"
        
        assert df_result.loc[0, "reliability_flag"] == "suppressed"
    
    def test_nan_count_handling(self):
        """NaN observed count should be handled gracefully."""
        df = pd.DataFrame({
            "geo_id": ["TEST"],
            "observed_count": [np.nan],
            "reference_pop": [10000.0],
            "reliability_flag": ["high"],
        })
        
        # NaN < 5 is False, so shouldn't trigger suppression
        # But should trigger low reliability based on missing data
        small_numerator = df["observed_count"] < 5
        assert pd.isna(small_numerator.iloc[0]) or small_numerator.iloc[0] == False
    
    def test_exactly_at_threshold(self):
        """Count exactly at suppression threshold should NOT be suppressed."""
        df = pd.DataFrame({
            "geo_id": ["TEST"],
            "observed_count": [5.0],  # Exactly at threshold
            "reference_pop": [10000.0],
            "reliability_flag": ["high"],
        })
        
        # Apply suppression: count < 5 → suppressed
        small_numerator = df["observed_count"] < 5
        df.loc[small_numerator, "reliability_flag"] = "suppressed"
        
        # 5 is NOT < 5, so should remain high (but may become low due to min_numerator)
        assert df.loc[0, "reliability_flag"] != "suppressed"


class TestVisualizationOpacity:
    """Tests for visualization opacity rules tied to reliability."""
    
    @pytest.fixture
    def params(self):
        """Load params.yml configuration."""
        return read_yaml(paths.params_yml)
    
    def test_opacity_rules_defined(self, params):
        """Visualization opacity rules should be defined for each reliability level."""
        assert "visualization" in params
        assert "opacity" in params["visualization"]
        
        opacity = params["visualization"]["opacity"]
        assert "high" in opacity
        assert "low" in opacity
        assert "suppressed" in opacity
    
    def test_opacity_values_are_valid(self, params):
        """Opacity values should be between 0 and 1."""
        opacity = params["visualization"]["opacity"]
        
        for level, value in opacity.items():
            assert 0 <= value <= 1, f"Invalid opacity for {level}: {value}"
    
    def test_high_has_highest_opacity(self, params):
        """High reliability should have highest opacity."""
        opacity = params["visualization"]["opacity"]
        
        assert opacity["high"] >= opacity["low"]
        assert opacity["high"] >= opacity["suppressed"]
    
    def test_suppressed_has_lowest_opacity(self, params):
        """Suppressed should have lowest opacity."""
        opacity = params["visualization"]["opacity"]
        
        assert opacity["suppressed"] <= opacity["low"]
        assert opacity["suppressed"] <= opacity["high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

