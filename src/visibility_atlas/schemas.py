"""
Schema validation for canonical outputs.

This module implements guardrail R3: Schema validation is mandatory.
Validate all canonical outputs on write and read. Schema drift is a hard failure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from visibility_atlas.paths import paths


class SchemaValidationError(Exception):
    """Raised when data does not conform to expected schema."""
    pass


@dataclass
class ColumnSpec:
    """Specification for a single column."""
    name: str
    dtype: str  # pandas dtype string (e.g., "int64", "float64", "object", "geometry")
    required: bool = True
    nullable: bool = False
    description: str = ""


@dataclass
class TableSchema:
    """Schema definition for a table."""
    name: str
    description: str
    columns: list[ColumnSpec]
    
    def required_columns(self) -> list[str]:
        """Get list of required column names."""
        return [c.name for c in self.columns if c.required]
    
    def all_columns(self) -> list[str]:
        """Get list of all column names."""
        return [c.name for c in self.columns]


# =============================================================================
# Schema definitions for canonical outputs
# =============================================================================

SCHEMA_CANONICAL_GEO = TableSchema(
    name="canonical_geo",
    description="Canonical geography boundaries",
    columns=[
        ColumnSpec("geo_id", "object", required=True, nullable=False,
                   description="Unique geography identifier"),
        ColumnSpec("geo_name", "object", required=True, nullable=True,
                   description="Human-readable geography name"),
        ColumnSpec("geo_type", "object", required=True, nullable=False,
                   description="Geography type (e.g., 'nta', 'zcta')"),
        ColumnSpec("borough", "object", required=False, nullable=True,
                   description="Borough name if applicable"),
        ColumnSpec("geometry", "geometry", required=True, nullable=False,
                   description="Geometry column"),
    ]
)

SCHEMA_CROSSWALK = TableSchema(
    name="crosswalk",
    description="Population-weighted crosswalk between geographies",
    columns=[
        ColumnSpec("source_geo_id", "object", required=True, nullable=False,
                   description="Source geography ID"),
        ColumnSpec("target_geo_id", "object", required=True, nullable=False,
                   description="Target geography ID"),
        ColumnSpec("weight", "float64", required=True, nullable=False,
                   description="Population weight (0-1)"),
    ]
)

SCHEMA_ACS_DENOMINATORS = TableSchema(
    name="acs_denominators",
    description="ACS-derived population denominators by geography and stratum",
    columns=[
        ColumnSpec("geo_id", "object", required=True, nullable=False,
                   description="Geography identifier"),
        ColumnSpec("stratum_id", "object", required=True, nullable=False,
                   description="Stratum identifier (e.g., 'age_18_24', 'total')"),
        ColumnSpec("time_window_id", "object", required=True, nullable=False,
                   description="Time window identifier (e.g., '2018_2022')"),
        ColumnSpec("reference_pop", "float64", required=True, nullable=False,
                   description="Reference population estimate"),
        ColumnSpec("reference_pop_moe", "float64", required=True, nullable=True,
                   description="Margin of error for reference population"),
        ColumnSpec("reference_pop_se", "float64", required=False, nullable=True,
                   description="Standard error (derived from MOE)"),
    ]
)

SCHEMA_VISIBILITY = TableSchema(
    name="visibility",
    description="Visibility index by source, geography, stratum, and time",
    columns=[
        ColumnSpec("geo_id", "object", required=True, nullable=False,
                   description="Geography identifier"),
        ColumnSpec("stratum_id", "object", required=True, nullable=False,
                   description="Stratum identifier"),
        ColumnSpec("time_window_id", "object", required=True, nullable=False,
                   description="Time window identifier"),
        ColumnSpec("source_id", "object", required=True, nullable=False,
                   description="Data source identifier"),
        ColumnSpec("observed_count", "float64", required=True, nullable=True,
                   description="Observed count (numerator)"),
        ColumnSpec("reference_pop", "float64", required=True, nullable=False,
                   description="Reference population (denominator)"),
        ColumnSpec("visibility", "float64", required=True, nullable=True,
                   description="Visibility index (per 1,000 residents)"),
        ColumnSpec("reliability_flag", "object", required=True, nullable=False,
                   description="Reliability indicator (e.g., 'high', 'low', 'suppressed')"),
        ColumnSpec("numerator_type", "object", required=True, nullable=False,
                   description="Type of numerator (respondents|unique_persons|encounters|enrollees|events)"),
    ]
)

SCHEMA_VISIBILITY_MATRIX = TableSchema(
    name="visibility_matrix",
    description="Cross-source visibility matrix (wide format)",
    columns=[
        ColumnSpec("geo_id", "object", required=True, nullable=False,
                   description="Geography identifier"),
        ColumnSpec("stratum_id", "object", required=True, nullable=False,
                   description="Stratum identifier"),
        ColumnSpec("time_window_id", "object", required=True, nullable=False,
                   description="Time window identifier"),
        # Additional columns are source-specific visibility values
    ]
)

SCHEMA_TYPOLOGY = TableSchema(
    name="typology_assignments",
    description="Neighborhood typology cluster assignments",
    columns=[
        ColumnSpec("geo_id", "object", required=True, nullable=False,
                   description="Geography identifier"),
        ColumnSpec("typology_id", "object", required=True, nullable=False,
                   description="Assigned typology/cluster ID"),
        ColumnSpec("typology_label", "object", required=True, nullable=False,
                   description="Human-readable typology label"),
        ColumnSpec("stability_score", "float64", required=False, nullable=True,
                   description="Bootstrap stability score (0-1)"),
    ]
)

# Registry of all schemas
SCHEMA_REGISTRY: dict[str, TableSchema] = {
    "canonical_geo": SCHEMA_CANONICAL_GEO,
    "crosswalk": SCHEMA_CROSSWALK,
    "acs_denominators": SCHEMA_ACS_DENOMINATORS,
    "visibility": SCHEMA_VISIBILITY,
    "visibility_matrix": SCHEMA_VISIBILITY_MATRIX,
    "typology_assignments": SCHEMA_TYPOLOGY,
}


# =============================================================================
# Validation functions
# =============================================================================

def validate_schema(
    df: pd.DataFrame,
    schema: TableSchema | str,
    strict: bool = False,
) -> list[str]:
    """
    Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate.
        schema: TableSchema object or schema name from registry.
        strict: If True, fail on extra columns not in schema.
        
    Returns:
        List of validation error messages (empty if valid).
        
    Raises:
        SchemaValidationError: If validation fails.
    """
    if isinstance(schema, str):
        if schema not in SCHEMA_REGISTRY:
            raise ValueError(f"Unknown schema: {schema}. Available: {list(SCHEMA_REGISTRY.keys())}")
        schema = SCHEMA_REGISTRY[schema]
    
    errors = []
    
    # Check required columns exist
    for col in schema.columns:
        if col.required and col.name not in df.columns:
            errors.append(f"Missing required column: {col.name}")
    
    # Check for extra columns in strict mode
    if strict:
        schema_cols = set(schema.all_columns())
        extra_cols = set(df.columns) - schema_cols
        if extra_cols:
            errors.append(f"Unexpected columns: {sorted(extra_cols)}")
    
    # Check column types and nullability
    for col in schema.columns:
        if col.name not in df.columns:
            continue
        
        series = df[col.name]
        
        # Check nullability
        if not col.nullable and series.isna().any():
            null_count = series.isna().sum()
            errors.append(f"Column '{col.name}' has {null_count} null values but is not nullable")
        
        # Check dtype (skip for geometry which needs special handling)
        if col.dtype != "geometry":
            # Flexible type checking
            actual_dtype = str(series.dtype)
            expected_dtype = col.dtype
            
            # Allow some type flexibility
            compatible = False
            if expected_dtype == "object" and actual_dtype in ("object", "string", "category"):
                compatible = True
            elif expected_dtype == "int64" and actual_dtype in ("int64", "int32", "Int64", "Int32"):
                compatible = True
            elif expected_dtype == "float64" and actual_dtype in ("float64", "float32", "Float64"):
                compatible = True
            elif expected_dtype == actual_dtype:
                compatible = True
            
            if not compatible:
                errors.append(f"Column '{col.name}' has dtype '{actual_dtype}', expected '{expected_dtype}'")
    
    if errors:
        error_msg = f"Schema validation failed for '{schema.name}':\n" + "\n".join(f"  - {e}" for e in errors)
        raise SchemaValidationError(error_msg)
    
    return errors


def validate_geodataframe(
    gdf: pd.DataFrame,
    schema: TableSchema | str,
    check_crs: bool = True,
    expected_crs: str = "EPSG:4326",
) -> list[str]:
    """
    Validate a GeoDataFrame against a schema.
    
    Args:
        gdf: GeoDataFrame to validate.
        schema: TableSchema object or schema name.
        check_crs: Whether to validate CRS.
        expected_crs: Expected CRS string.
        
    Returns:
        List of validation error messages.
        
    Raises:
        SchemaValidationError: If validation fails.
    """
    import geopandas as gpd
    
    errors = []
    
    # Check it's actually a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise SchemaValidationError("Expected GeoDataFrame but got DataFrame")
    
    # Check CRS
    if check_crs:
        if gdf.crs is None:
            errors.append("GeoDataFrame has no CRS defined")
        elif str(gdf.crs) != expected_crs and gdf.crs.to_string() != expected_crs:
            # Try to match by authority code
            try:
                if gdf.crs.to_epsg() != int(expected_crs.split(":")[1]):
                    errors.append(f"CRS mismatch: got {gdf.crs}, expected {expected_crs}")
            except (ValueError, AttributeError):
                errors.append(f"CRS mismatch: got {gdf.crs}, expected {expected_crs}")
    
    # Check for empty geometries
    if gdf.geometry.is_empty.any():
        empty_count = gdf.geometry.is_empty.sum()
        errors.append(f"GeoDataFrame has {empty_count} empty geometries")
    
    if errors:
        # Validate the rest of the schema too
        try:
            validate_schema(gdf, schema)
        except SchemaValidationError as e:
            errors.extend(str(e).split("\n")[1:])
        
        error_msg = f"GeoDataFrame validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise SchemaValidationError(error_msg)
    
    # If no geo-specific errors, validate the rest of the schema
    validate_schema(gdf, schema)
    
    return errors


def get_schema(name: str) -> TableSchema:
    """
    Get a schema by name from the registry.
    
    Args:
        name: Schema name.
        
    Returns:
        TableSchema object.
        
    Raises:
        ValueError: If schema not found.
    """
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[name]


def register_schema(schema: TableSchema) -> None:
    """
    Register a new schema in the registry.
    
    Args:
        schema: TableSchema to register.
    """
    SCHEMA_REGISTRY[schema.name] = schema

