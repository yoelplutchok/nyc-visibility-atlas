"""
Tests for visibility_atlas.schemas module.

Tests cover:
- Schema definitions are complete
- Schema validation catches missing columns
- Schema validation catches type mismatches
- Schema validation catches null values in non-nullable columns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np

from visibility_atlas.schemas import (
    SCHEMA_REGISTRY,
    SCHEMA_VISIBILITY,
    SCHEMA_CROSSWALK,
    SCHEMA_ACS_DENOMINATORS,
    SCHEMA_TYPOLOGY,
    TableSchema,
    ColumnSpec,
    validate_schema,
    get_schema,
    SchemaValidationError,
)


class TestSchemaRegistry:
    """Tests for schema registry completeness."""
    
    def test_registry_has_required_schemas(self):
        """Registry should contain all required schema definitions."""
        required_schemas = [
            "canonical_geo",
            "crosswalk",
            "acs_denominators",
            "visibility",
            "visibility_matrix",
            "typology_assignments",
        ]
        for schema_name in required_schemas:
            assert schema_name in SCHEMA_REGISTRY, f"Missing schema: {schema_name}"
    
    def test_get_schema_returns_schema(self):
        """get_schema() should return TableSchema objects."""
        schema = get_schema("visibility")
        assert isinstance(schema, TableSchema)
    
    def test_get_schema_raises_on_unknown(self):
        """get_schema() should raise ValueError for unknown schemas."""
        with pytest.raises(ValueError, match="Unknown schema"):
            get_schema("nonexistent_schema")


class TestVisibilitySchema:
    """Tests for the visibility schema definition."""
    
    def test_visibility_schema_has_required_columns(self):
        """Visibility schema should define all required columns."""
        required_cols = [
            "geo_id",
            "stratum_id",
            "time_window_id",
            "source_id",
            "observed_count",
            "reference_pop",
            "visibility",
            "reliability_flag",
            "numerator_type",
        ]
        schema_cols = SCHEMA_VISIBILITY.all_columns()
        for col in required_cols:
            assert col in schema_cols, f"Missing column in schema: {col}"
    
    def test_visibility_schema_required_columns(self):
        """Visibility schema should mark key columns as required."""
        required = SCHEMA_VISIBILITY.required_columns()
        assert "geo_id" in required
        assert "source_id" in required
        assert "visibility" in required


class TestValidateSchema:
    """Tests for validate_schema() function."""
    
    @pytest.fixture
    def valid_visibility_df(self):
        """Create a valid DataFrame matching visibility schema."""
        return pd.DataFrame({
            "geo_id": ["BK0101", "BK0102", "MN0101"],
            "stratum_id": ["total", "total", "total"],
            "time_window_id": ["2019", "2019", "2019"],
            "source_id": ["chs", "chs", "chs"],
            "observed_count": [10.0, 15.0, 20.0],
            "reference_pop": [1000.0, 1500.0, 2000.0],
            "visibility": [10.0, 10.0, 10.0],
            "reliability_flag": ["high", "high", "low"],
            "numerator_type": ["respondents", "respondents", "respondents"],
        })
    
    def test_valid_dataframe_passes(self, valid_visibility_df):
        """Valid DataFrame should pass validation without error."""
        # Should not raise
        errors = validate_schema(valid_visibility_df, SCHEMA_VISIBILITY)
        assert errors == []
    
    def test_missing_required_column_fails(self, valid_visibility_df):
        """Missing required column should raise SchemaValidationError."""
        df = valid_visibility_df.drop(columns=["geo_id"])
        
        with pytest.raises(SchemaValidationError, match="Missing required column.*geo_id"):
            validate_schema(df, SCHEMA_VISIBILITY)
    
    def test_accepts_schema_by_name(self, valid_visibility_df):
        """Should accept schema name string instead of TableSchema object."""
        # Should not raise
        errors = validate_schema(valid_visibility_df, "visibility")
        assert errors == []
    
    def test_nullable_column_with_nulls_passes(self, valid_visibility_df):
        """Nullable column with null values should pass."""
        # observed_count is nullable=True
        df = valid_visibility_df.copy()
        df.loc[0, "observed_count"] = np.nan
        
        # Should not raise
        errors = validate_schema(df, SCHEMA_VISIBILITY)
        assert errors == []
    
    def test_non_nullable_column_with_nulls_fails(self, valid_visibility_df):
        """Non-nullable column with null values should fail."""
        # geo_id is nullable=False
        df = valid_visibility_df.copy()
        df.loc[0, "geo_id"] = None
        
        with pytest.raises(SchemaValidationError, match="null values"):
            validate_schema(df, SCHEMA_VISIBILITY)


class TestCrosswalkSchema:
    """Tests for the crosswalk schema."""
    
    @pytest.fixture
    def valid_crosswalk_df(self):
        """Create a valid crosswalk DataFrame."""
        return pd.DataFrame({
            "source_geo_id": ["101", "101", "102"],
            "target_geo_id": ["BK0101", "BK0102", "MN0101"],
            "weight": [0.6, 0.4, 1.0],
        })
    
    def test_valid_crosswalk_passes(self, valid_crosswalk_df):
        """Valid crosswalk DataFrame should pass validation."""
        errors = validate_schema(valid_crosswalk_df, SCHEMA_CROSSWALK)
        assert errors == []
    
    def test_crosswalk_requires_weight(self, valid_crosswalk_df):
        """Crosswalk schema should require weight column."""
        df = valid_crosswalk_df.drop(columns=["weight"])
        
        with pytest.raises(SchemaValidationError, match="Missing required column.*weight"):
            validate_schema(df, SCHEMA_CROSSWALK)
    
    def test_crosswalk_weight_must_be_float(self, valid_crosswalk_df):
        """Crosswalk weight column should be float64."""
        df = valid_crosswalk_df.copy()
        df["weight"] = df["weight"].astype(str)  # Wrong type
        
        with pytest.raises(SchemaValidationError, match="dtype"):
            validate_schema(df, SCHEMA_CROSSWALK)


class TestACSDenominatorsSchema:
    """Tests for the ACS denominators schema."""
    
    @pytest.fixture
    def valid_acs_df(self):
        """Create a valid ACS denominators DataFrame."""
        return pd.DataFrame({
            "geo_id": ["BK0101", "BK0101", "BK0102"],
            "stratum_id": ["total", "age_18_24", "total"],
            "time_window_id": ["2018_2022", "2018_2022", "2018_2022"],
            "reference_pop": [50000.0, 5000.0, 60000.0],
            "reference_pop_moe": [1000.0, 500.0, 1200.0],
        })
    
    def test_valid_acs_passes(self, valid_acs_df):
        """Valid ACS denominators DataFrame should pass validation."""
        errors = validate_schema(valid_acs_df, SCHEMA_ACS_DENOMINATORS)
        assert errors == []
    
    def test_acs_requires_reference_pop(self, valid_acs_df):
        """ACS schema should require reference_pop column."""
        df = valid_acs_df.drop(columns=["reference_pop"])
        
        with pytest.raises(SchemaValidationError, match="Missing required column.*reference_pop"):
            validate_schema(df, SCHEMA_ACS_DENOMINATORS)


class TestTypologySchema:
    """Tests for the typology assignments schema."""
    
    @pytest.fixture
    def valid_typology_df(self):
        """Create a valid typology DataFrame."""
        return pd.DataFrame({
            "geo_id": ["BK0101", "BK0102", "MN0101"],
            "typology_id": ["1", "2", "1"],
            "typology_label": ["High visibility", "Low visibility", "High visibility"],
            "stability_score": [0.95, 0.88, 0.92],
        })
    
    def test_valid_typology_passes(self, valid_typology_df):
        """Valid typology DataFrame should pass validation."""
        errors = validate_schema(valid_typology_df, SCHEMA_TYPOLOGY)
        assert errors == []
    
    def test_typology_requires_label(self, valid_typology_df):
        """Typology schema should require typology_label column."""
        df = valid_typology_df.drop(columns=["typology_label"])
        
        with pytest.raises(SchemaValidationError, match="Missing required column.*typology_label"):
            validate_schema(df, SCHEMA_TYPOLOGY)


class TestColumnSpec:
    """Tests for ColumnSpec dataclass."""
    
    def test_column_spec_creation(self):
        """Should create ColumnSpec with all attributes."""
        spec = ColumnSpec(
            name="test_col",
            dtype="float64",
            required=True,
            nullable=False,
            description="Test column"
        )
        assert spec.name == "test_col"
        assert spec.dtype == "float64"
        assert spec.required is True
        assert spec.nullable is False
        assert spec.description == "Test column"
    
    def test_column_spec_defaults(self):
        """ColumnSpec should have sensible defaults."""
        spec = ColumnSpec(name="test", dtype="object")
        assert spec.required is True
        assert spec.nullable is False
        assert spec.description == ""


class TestTableSchema:
    """Tests for TableSchema dataclass."""
    
    def test_required_columns(self):
        """required_columns() should return only required columns."""
        schema = TableSchema(
            name="test",
            description="Test schema",
            columns=[
                ColumnSpec("required_col", "object", required=True),
                ColumnSpec("optional_col", "object", required=False),
            ]
        )
        required = schema.required_columns()
        assert "required_col" in required
        assert "optional_col" not in required
    
    def test_all_columns(self):
        """all_columns() should return all column names."""
        schema = TableSchema(
            name="test",
            description="Test schema",
            columns=[
                ColumnSpec("col1", "object"),
                ColumnSpec("col2", "float64"),
            ]
        )
        all_cols = schema.all_columns()
        assert all_cols == ["col1", "col2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

