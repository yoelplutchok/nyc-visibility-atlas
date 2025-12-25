"""
Quality assurance checks for data processing.

This module provides QA checks for:
- CRS validation
- Geographic bounds (NYC bounding box)
- Unique identifiers
- Empty geometries
- Data completeness
- Cross-reference integrity
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from visibility_atlas.paths import paths
from visibility_atlas.logging_utils import log_qa_check


# =============================================================================
# NYC Geographic constants
# =============================================================================

# NYC bounding box (WGS84 / EPSG:4326)
# Slightly expanded to catch edge cases
NYC_BOUNDS = {
    "min_lon": -74.30,  # West (Staten Island)
    "max_lon": -73.65,  # East (Far Rockaway / Nassau border)
    "min_lat": 40.47,   # South (Staten Island southern tip)
    "max_lat": 40.95,   # North (Bronx / Westchester border)
}

# Expected CRS for geographic data
EXPECTED_CRS_WGS84 = "EPSG:4326"
EXPECTED_CRS_NYC = "EPSG:2263"  # NY State Plane Long Island (feet)


@dataclass
class QAResult:
    """Result of a QA check."""
    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None
    
    def __bool__(self) -> bool:
        return self.passed


# =============================================================================
# CRS checks
# =============================================================================

def check_crs(
    gdf: pd.DataFrame,
    expected_crs: str | None = EXPECTED_CRS_WGS84,
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that a GeoDataFrame has a valid CRS.
    
    Args:
        gdf: GeoDataFrame to check.
        expected_crs: Expected CRS string (e.g., "EPSG:4326"). If None, just checks CRS exists.
        logger: Optional logger for structured logging.
        
    Returns:
        QAResult with check outcome.
    """
    import geopandas as gpd
    
    check_name = "crs_valid"
    
    if not isinstance(gdf, gpd.GeoDataFrame):
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="Input is not a GeoDataFrame",
            details={"type": type(gdf).__name__}
        )
    elif gdf.crs is None:
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="GeoDataFrame has no CRS defined",
            details={"crs": None}
        )
    elif expected_crs is not None:
        # Compare CRS
        try:
            expected_epsg = int(expected_crs.split(":")[1])
            actual_epsg = gdf.crs.to_epsg()
            if actual_epsg == expected_epsg:
                result = QAResult(
                    check_name=check_name,
                    passed=True,
                    message=f"CRS is {expected_crs}",
                    details={"crs": str(gdf.crs), "expected": expected_crs}
                )
            else:
                result = QAResult(
                    check_name=check_name,
                    passed=False,
                    message=f"CRS mismatch: got EPSG:{actual_epsg}, expected {expected_crs}",
                    details={"crs": str(gdf.crs), "expected": expected_crs}
                )
        except (ValueError, AttributeError):
            # Fall back to string comparison
            if str(gdf.crs) == expected_crs:
                result = QAResult(
                    check_name=check_name,
                    passed=True,
                    message=f"CRS is {expected_crs}",
                    details={"crs": str(gdf.crs)}
                )
            else:
                result = QAResult(
                    check_name=check_name,
                    passed=False,
                    message=f"CRS mismatch: got {gdf.crs}, expected {expected_crs}",
                    details={"crs": str(gdf.crs), "expected": expected_crs}
                )
    else:
        result = QAResult(
            check_name=check_name,
            passed=True,
            message=f"CRS is defined: {gdf.crs}",
            details={"crs": str(gdf.crs)}
        )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


# =============================================================================
# Bounds checks
# =============================================================================

def check_bounds_nyc(
    gdf: pd.DataFrame,
    bounds: dict | None = None,
    tolerance: float = 0.01,
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that geometries fall within NYC bounding box.
    
    Args:
        gdf: GeoDataFrame to check (must be in WGS84).
        bounds: Custom bounds dict with min_lon, max_lon, min_lat, max_lat.
        tolerance: Tolerance for bounds check (degrees).
        logger: Optional logger.
        
    Returns:
        QAResult with check outcome.
    """
    import geopandas as gpd
    
    check_name = "bounds_nyc"
    bounds = bounds or NYC_BOUNDS
    
    if not isinstance(gdf, gpd.GeoDataFrame):
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="Input is not a GeoDataFrame"
        )
    elif gdf.crs is None:
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="Cannot check bounds: no CRS defined"
        )
    else:
        # Reproject to WGS84 if needed
        gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs.to_epsg() != 4326 else gdf
        
        total_bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
        
        min_lon, min_lat, max_lon, max_lat = total_bounds
        
        within_bounds = (
            min_lon >= bounds["min_lon"] - tolerance and
            max_lon <= bounds["max_lon"] + tolerance and
            min_lat >= bounds["min_lat"] - tolerance and
            max_lat <= bounds["max_lat"] + tolerance
        )
        
        if within_bounds:
            result = QAResult(
                check_name=check_name,
                passed=True,
                message="All geometries within NYC bounds",
                details={
                    "data_bounds": {"min_lon": min_lon, "min_lat": min_lat, 
                                    "max_lon": max_lon, "max_lat": max_lat},
                    "expected_bounds": bounds
                }
            )
        else:
            result = QAResult(
                check_name=check_name,
                passed=False,
                message="Some geometries outside NYC bounds",
                details={
                    "data_bounds": {"min_lon": min_lon, "min_lat": min_lat,
                                    "max_lon": max_lon, "max_lat": max_lat},
                    "expected_bounds": bounds
                }
            )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


# =============================================================================
# ID uniqueness checks
# =============================================================================

def check_unique_ids(
    df: pd.DataFrame,
    id_column: str,
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that ID column has unique values.
    
    Args:
        df: DataFrame to check.
        id_column: Name of the ID column.
        logger: Optional logger.
        
    Returns:
        QAResult with check outcome.
    """
    check_name = "unique_ids"
    
    if id_column not in df.columns:
        result = QAResult(
            check_name=check_name,
            passed=False,
            message=f"ID column '{id_column}' not found",
            details={"columns": list(df.columns)}
        )
    else:
        total = len(df)
        unique = df[id_column].nunique()
        
        if total == unique:
            result = QAResult(
                check_name=check_name,
                passed=True,
                message=f"All {total} IDs are unique",
                details={"total": total, "unique": unique, "column": id_column}
            )
        else:
            duplicates = df[id_column].value_counts()
            dup_ids = duplicates[duplicates > 1].head(5).to_dict()
            result = QAResult(
                check_name=check_name,
                passed=False,
                message=f"Found {total - unique} duplicate IDs",
                details={
                    "total": total, 
                    "unique": unique, 
                    "duplicates": total - unique,
                    "sample_duplicates": dup_ids,
                    "column": id_column
                }
            )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


# =============================================================================
# Geometry checks
# =============================================================================

def check_no_empty_geoms(
    gdf: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that there are no empty geometries.
    
    Args:
        gdf: GeoDataFrame to check.
        logger: Optional logger.
        
    Returns:
        QAResult with check outcome.
    """
    import geopandas as gpd
    
    check_name = "no_empty_geoms"
    
    if not isinstance(gdf, gpd.GeoDataFrame):
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="Input is not a GeoDataFrame"
        )
    else:
        empty_count = gdf.geometry.is_empty.sum()
        null_count = gdf.geometry.isna().sum()
        
        if empty_count == 0 and null_count == 0:
            result = QAResult(
                check_name=check_name,
                passed=True,
                message=f"All {len(gdf)} geometries are valid",
                details={"total": len(gdf), "empty": 0, "null": 0}
            )
        else:
            result = QAResult(
                check_name=check_name,
                passed=False,
                message=f"Found {empty_count} empty and {null_count} null geometries",
                details={"total": len(gdf), "empty": empty_count, "null": null_count}
            )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


def check_valid_geoms(
    gdf: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that all geometries are topologically valid.
    
    Args:
        gdf: GeoDataFrame to check.
        logger: Optional logger.
        
    Returns:
        QAResult with check outcome.
    """
    import geopandas as gpd
    
    check_name = "valid_geoms"
    
    if not isinstance(gdf, gpd.GeoDataFrame):
        result = QAResult(
            check_name=check_name,
            passed=False,
            message="Input is not a GeoDataFrame"
        )
    else:
        invalid_mask = ~gdf.geometry.is_valid
        invalid_count = invalid_mask.sum()
        
        if invalid_count == 0:
            result = QAResult(
                check_name=check_name,
                passed=True,
                message=f"All {len(gdf)} geometries are topologically valid",
                details={"total": len(gdf), "invalid": 0}
            )
        else:
            result = QAResult(
                check_name=check_name,
                passed=False,
                message=f"Found {invalid_count} invalid geometries",
                details={"total": len(gdf), "invalid": invalid_count}
            )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


# =============================================================================
# Completeness checks
# =============================================================================

def check_no_nulls(
    df: pd.DataFrame,
    columns: list[str],
    logger: logging.Logger | None = None,
) -> QAResult:
    """
    Check that specified columns have no null values.
    
    Args:
        df: DataFrame to check.
        columns: Column names to check.
        logger: Optional logger.
        
    Returns:
        QAResult with check outcome.
    """
    check_name = "no_nulls"
    
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        result = QAResult(
            check_name=check_name,
            passed=False,
            message=f"Columns not found: {missing_cols}",
            details={"missing_columns": missing_cols}
        )
    else:
        null_counts = {c: df[c].isna().sum() for c in columns}
        total_nulls = sum(null_counts.values())
        
        if total_nulls == 0:
            result = QAResult(
                check_name=check_name,
                passed=True,
                message=f"No null values in {len(columns)} checked columns",
                details={"columns": columns, "null_counts": null_counts}
            )
        else:
            cols_with_nulls = {k: v for k, v in null_counts.items() if v > 0}
            result = QAResult(
                check_name=check_name,
                passed=False,
                message=f"Found {total_nulls} null values",
                details={"columns_with_nulls": cols_with_nulls}
            )
    
    if logger:
        log_qa_check(logger, check_name, result.passed, result.message, **(result.details or {}))
    
    return result


# =============================================================================
# Aggregate QA runner
# =============================================================================

def run_geo_qa_checks(
    gdf: pd.DataFrame,
    id_column: str = "geo_id",
    expected_crs: str = EXPECTED_CRS_WGS84,
    logger: logging.Logger | None = None,
    fail_on_error: bool = True,
) -> list[QAResult]:
    """
    Run standard QA checks for geographic data.
    
    Args:
        gdf: GeoDataFrame to check.
        id_column: Name of the ID column.
        expected_crs: Expected CRS string.
        logger: Optional logger.
        fail_on_error: If True, raise exception on first failure.
        
    Returns:
        List of QAResult objects.
        
    Raises:
        ValueError: If fail_on_error and any check fails.
    """
    results = []
    
    # CRS check
    results.append(check_crs(gdf, expected_crs, logger))
    
    # Bounds check
    results.append(check_bounds_nyc(gdf, logger=logger))
    
    # Unique IDs
    results.append(check_unique_ids(gdf, id_column, logger))
    
    # No empty geometries
    results.append(check_no_empty_geoms(gdf, logger))
    
    # Valid geometries
    results.append(check_valid_geoms(gdf, logger))
    
    if fail_on_error:
        failed = [r for r in results if not r.passed]
        if failed:
            messages = [f"{r.check_name}: {r.message}" for r in failed]
            raise ValueError(f"QA checks failed:\n" + "\n".join(messages))
    
    return results

