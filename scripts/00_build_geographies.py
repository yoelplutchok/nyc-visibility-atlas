#!/usr/bin/env python3
"""
00_build_geographies.py

Build canonical boundary files for the primary geography (NTA) and optional
sensitivity geography (ZCTA).

Pipeline Step: 00
Contract Reference: Section 11 - 00_build_geographies.py

Inputs:
    - Raw boundary files from NYC Open Data / Census TIGER
    - configs/params.yml (geography settings)
    - configs/data_inventory.yml (source URLs)

Outputs:
    - data/processed/geo/nta_canonical.parquet (GeoParquet)
    - data/processed/geo/nta_canonical.geojson (export only)
    - data/processed/metadata/nta_canonical_metadata.json

QA Checks:
    - CRS present and correct (EPSG:4326)
    - Bounds within NYC bounding box
    - Unique geo IDs
    - No empty geometries
    - No invalid geometries

Failure Modes:
    - Duplicate IDs
    - Missing CRS
    - Out-of-bounds geometry
    - Download failure (with manual fallback path)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import logging
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end, 
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_geoparquet, atomic_write_geojson, 
    atomic_write_json, read_yaml
)
from visibility_atlas.hashing import write_metadata_sidecar
from visibility_atlas.qa import (
    run_geo_qa_checks, check_crs, check_bounds_nyc,
    check_unique_ids, check_no_empty_geoms, check_valid_geoms,
    EXPECTED_CRS_WGS84
)
from visibility_atlas.schemas import validate_geodataframe, SCHEMA_CANONICAL_GEO


# =============================================================================
# CONSTANTS
# =============================================================================

SCRIPT_NAME = "00_build_geographies"

# NYC Open Data URLs for NTA boundaries
NTA_GEOJSON_URL = "https://data.cityofnewyork.us/api/geospatial/9nt8-h7nd?method=export&format=GeoJSON"

# Column mappings for NTA data
NTA_COLUMN_MAP = {
    "ntacode": "geo_id",
    "ntaname": "geo_name", 
    "nta2020": "geo_id",      # Alternative column name
    "ntaname2020": "geo_name", # Alternative column name
    "boroname": "borough",
    "boro_name": "borough",
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def download_nta_boundaries(logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    Download NTA boundaries from NYC Open Data.
    
    Falls back to local file if download fails.
    """
    log_step_start(logger, "download_nta_boundaries")
    
    raw_geo_dir = ensure_dir(paths.raw_geo)
    local_file = raw_geo_dir / "nta_2020.geojson"
    
    # Try to download
    try:
        logger.info(f"Downloading NTA boundaries from NYC Open Data...")
        gdf = gpd.read_file(NTA_GEOJSON_URL)
        logger.info(f"Downloaded {len(gdf)} NTA features")
        
        # Save raw file for provenance
        gdf.to_file(local_file, driver="GeoJSON")
        logger.info(f"Saved raw file to {local_file}")
        
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        
        # Check for local file
        if local_file.exists():
            logger.info(f"Using existing local file: {local_file}")
            gdf = gpd.read_file(local_file)
        else:
            # Provide manual download instructions
            logger.error(
                f"Cannot download NTA boundaries and no local file found.\n"
                f"Please manually download from:\n"
                f"  {NTA_GEOJSON_URL}\n"
                f"And save to:\n"
                f"  {local_file}"
            )
            raise FileNotFoundError(
                f"NTA boundaries not available. Download manually to {local_file}"
            )
    
    log_step_end(logger, "download_nta_boundaries", feature_count=len(gdf))
    return gdf


def normalize_nta_schema(
    gdf: gpd.GeoDataFrame, 
    logger: logging.Logger
) -> gpd.GeoDataFrame:
    """
    Normalize NTA GeoDataFrame to canonical schema.
    
    Canonical columns:
        - geo_id: Unique geography identifier
        - geo_name: Human-readable name
        - geo_type: Geography type ("nta")
        - borough: Borough name
        - geometry: Geometry column
    """
    log_step_start(logger, "normalize_nta_schema")
    
    logger.info(f"Input columns: {list(gdf.columns)}")
    
    # Create normalized dataframe
    normalized = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    
    # Map columns
    for src_col, tgt_col in NTA_COLUMN_MAP.items():
        src_col_lower = src_col.lower()
        # Check both original and lowercase
        for col in gdf.columns:
            if col.lower() == src_col_lower:
                if tgt_col not in normalized.columns:
                    normalized[tgt_col] = gdf[col]
                    logger.info(f"Mapped column: {col} -> {tgt_col}")
                break
    
    # Add geo_type
    normalized["geo_type"] = "nta"
    
    # Ensure required columns exist
    required = ["geo_id", "geo_name", "geo_type", "borough", "geometry"]
    missing = [c for c in required if c not in normalized.columns]
    
    if missing:
        logger.warning(f"Missing columns after mapping: {missing}")
        # Try to fill from available data
        if "geo_id" in missing:
            # Look for any ID-like column
            for col in gdf.columns:
                if "nta" in col.lower() and "code" in col.lower():
                    normalized["geo_id"] = gdf[col]
                    logger.info(f"Used {col} as geo_id")
                    break
        
        if "geo_name" in missing:
            for col in gdf.columns:
                if "nta" in col.lower() and "name" in col.lower():
                    normalized["geo_name"] = gdf[col]
                    logger.info(f"Used {col} as geo_name")
                    break
        
        if "borough" in missing:
            for col in gdf.columns:
                if "boro" in col.lower():
                    normalized["borough"] = gdf[col]
                    logger.info(f"Used {col} as borough")
                    break
    
    # Fill any still-missing columns with None
    for col in required:
        if col not in normalized.columns:
            normalized[col] = None
            logger.warning(f"Column {col} filled with None")
    
    # Reorder columns
    normalized = normalized[required]
    
    # Ensure CRS is WGS84
    if normalized.crs is None:
        logger.warning("No CRS detected, setting to EPSG:4326")
        normalized = normalized.set_crs("EPSG:4326")
    elif normalized.crs.to_epsg() != 4326:
        logger.info(f"Reprojecting from {normalized.crs} to EPSG:4326")
        normalized = normalized.to_crs("EPSG:4326")
    
    # Sort by geo_id for determinism
    normalized = normalized.sort_values("geo_id").reset_index(drop=True)
    
    log_step_end(logger, "normalize_nta_schema", 
                 columns=list(normalized.columns),
                 row_count=len(normalized))
    
    return normalized


def run_qa_checks(
    gdf: gpd.GeoDataFrame, 
    logger: logging.Logger,
    fail_on_error: bool = True
) -> list:
    """
    Run all QA checks on the canonical geography.
    """
    log_step_start(logger, "qa_checks")
    
    results = []
    
    # CRS check
    result = check_crs(gdf, EXPECTED_CRS_WGS84, logger)
    results.append(result)
    
    # Bounds check
    result = check_bounds_nyc(gdf, logger=logger)
    results.append(result)
    
    # Unique IDs
    result = check_unique_ids(gdf, "geo_id", logger)
    results.append(result)
    
    # No empty geometries
    result = check_no_empty_geoms(gdf, logger)
    results.append(result)
    
    # Valid geometries
    result = check_valid_geoms(gdf, logger)
    results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    logger.info(f"QA Summary: {passed} passed, {failed} failed")
    
    if fail_on_error and failed > 0:
        failed_checks = [r for r in results if not r.passed]
        messages = [f"  - {r.check_name}: {r.message}" for r in failed_checks]
        raise ValueError(f"QA checks failed:\n" + "\n".join(messages))
    
    log_step_end(logger, "qa_checks", passed=passed, failed=failed)
    
    return results


def write_outputs(
    gdf: gpd.GeoDataFrame,
    logger: logging.Logger,
    run_id: str,
    input_files: list = None,
    config_files: list = None,
) -> dict:
    """
    Write canonical outputs with metadata sidecars.
    """
    log_step_start(logger, "write_outputs")
    
    output_dir = ensure_dir(paths.processed_geo)
    
    outputs = {}
    
    # Write GeoParquet (primary format)
    parquet_path = output_dir / "nta_canonical.parquet"
    atomic_write_geoparquet(parquet_path, gdf)
    log_output_written(logger, parquet_path, row_count=len(gdf))
    outputs["parquet"] = parquet_path
    
    # Write GeoJSON (export format)
    geojson_path = output_dir / "nta_canonical.geojson"
    atomic_write_geojson(geojson_path, gdf)
    log_output_written(logger, geojson_path, row_count=len(gdf))
    outputs["geojson"] = geojson_path
    
    # Write metadata sidecar
    metadata_path = write_metadata_sidecar(
        output_path=parquet_path,
        run_id=run_id,
        input_files=input_files,
        config_files=config_files,
        parameters={
            "geo_type": "nta",
            "source": "NYC Open Data",
            "crs": str(gdf.crs),
        },
        row_count=len(gdf),
        extra={
            "boroughs": gdf["borough"].dropna().unique().tolist(),
            "bounds": list(gdf.total_bounds),
        }
    )
    outputs["metadata"] = metadata_path
    logger.info(f"Wrote metadata sidecar: {metadata_path}")
    
    log_step_end(logger, "write_outputs")
    
    return outputs


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for 00_build_geographies."""
    
    # Initialize logger
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load config
        params = read_yaml(paths.params_yml)
        primary_geo = params["geography"]["primary"]
        logger.info(f"Primary geography: {primary_geo}")
        
        if primary_geo != "nta":
            logger.warning(f"Primary geography is {primary_geo}, but this script builds NTA")
        
        # Download NTA boundaries
        raw_gdf = download_nta_boundaries(logger)
        
        # Normalize to canonical schema
        canonical_gdf = normalize_nta_schema(raw_gdf, logger)
        
        # Run QA checks
        qa_results = run_qa_checks(canonical_gdf, logger, fail_on_error=True)
        
        # Write outputs
        input_files = [paths.raw_geo / "nta_2020.geojson"]
        config_files = [paths.params_yml]
        
        outputs = write_outputs(
            canonical_gdf, 
            logger, 
            run_id,
            input_files=[f for f in input_files if f.exists()],
            config_files=config_files,
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   NTA count: {len(canonical_gdf)}")
        logger.info(f"   Boroughs: {canonical_gdf['borough'].dropna().unique().tolist()}")
        logger.info(f"   Output: {outputs['parquet']}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure required data files are available or download manually.")
        return 1
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

