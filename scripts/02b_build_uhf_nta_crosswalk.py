#!/usr/bin/env python3
"""
02b_build_uhf_nta_crosswalk.py

Build a proper UHF→NTA population-weighted crosswalk using REAL UHF boundaries.

This addresses the critique: "Borough-based UHF→NTA smoothing is avoidable
since real UHF boundaries exist."

Source: NYC DOHMH EHDP GitHub repository
https://github.com/nychealth/EHDP-data/tree/production/geography
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import get_logger, log_step_start, log_step_end, get_run_id
from visibility_atlas.io_utils import atomic_write_parquet
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "02b_build_uhf_nta_crosswalk"


def load_uhf_boundaries(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load UHF42 boundaries from downloaded GeoJSON."""
    uhf_path = paths.raw_geo / "uhf42_boundaries.geojson"
    
    if not uhf_path.exists():
        raise FileNotFoundError(f"UHF boundaries not found at {uhf_path}")
    
    uhf = gpd.read_file(uhf_path)
    
    # Rename for consistency
    uhf = uhf.rename(columns={
        'GEOCODE': 'uhf_code',
        'GEONAME': 'uhf_name',
        'BOROUGH': 'borough',
    })
    
    # Filter out N/A (parks/water)
    uhf = uhf[uhf['borough'] != 'N/A'].copy()
    
    logger.info(f"Loaded {len(uhf)} UHF neighborhoods")
    return uhf


def load_nta_boundaries(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load NTA boundaries."""
    nta_path = paths.processed_geo / "nta_canonical.geojson"
    
    if not nta_path.exists():
        raise FileNotFoundError(f"NTA boundaries not found at {nta_path}")
    
    nta = gpd.read_file(nta_path)
    
    logger.info(f"Loaded {len(nta)} NTA neighborhoods")
    return nta


def load_nta_populations(logger: logging.Logger) -> pd.DataFrame:
    """Load NTA population estimates."""
    pop_path = paths.processed_denominators / "acs_denominators.parquet"
    
    pop = pd.read_parquet(pop_path)
    
    # Filter to total stratum
    pop_total = pop[pop['stratum_id'] == 'total'][['geo_id', 'reference_pop']].copy()
    
    logger.info(f"Loaded populations for {len(pop_total)} NTAs")
    return pop_total


def compute_spatial_intersection(
    uhf: gpd.GeoDataFrame,
    nta: gpd.GeoDataFrame,
    nta_pop: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute UHF→NTA crosswalk using spatial intersection and population weighting.
    
    For each UHF-NTA pair that intersects:
    - Calculate the area of intersection
    - Weight by NTA population within that intersection
    """
    log_step_start(logger, "compute_spatial_intersection")
    
    # Ensure same CRS
    if uhf.crs != nta.crs:
        nta = nta.to_crs(uhf.crs)
    
    # Fix invalid geometries (buffer by 0)
    uhf['geometry'] = uhf['geometry'].buffer(0)
    nta['geometry'] = nta['geometry'].buffer(0)
    
    # Add NTA population
    nta = nta.merge(nta_pop, on='geo_id', how='left')
    
    # Compute intersection
    intersections = []
    
    for _, uhf_row in uhf.iterrows():
        uhf_code = uhf_row['uhf_code']
        uhf_geom = uhf_row['geometry']
        
        for _, nta_row in nta.iterrows():
            nta_id = nta_row['geo_id']
            nta_geom = nta_row['geometry']
            nta_area = nta_geom.area
            nta_pop_val = nta_row.get('reference_pop', 0)
            
            if uhf_geom.intersects(nta_geom):
                intersection = uhf_geom.intersection(nta_geom)
                intersection_area = intersection.area
                
                # Calculate fraction of NTA in this UHF
                area_fraction = intersection_area / nta_area if nta_area > 0 else 0
                
                # Weight by population in intersection
                pop_in_intersection = nta_pop_val * area_fraction if pd.notna(nta_pop_val) else 0
                
                if area_fraction > 0.001:  # Only keep meaningful overlaps
                    intersections.append({
                        'uhf_code': uhf_code,
                        'geo_id': nta_id,
                        'area_fraction': area_fraction,
                        'pop_in_intersection': pop_in_intersection,
                    })
    
    crosswalk = pd.DataFrame(intersections)
    
    # Compute weight: fraction of UHF population in each NTA
    # (sum of weights for each UHF should = 1)
    uhf_totals = crosswalk.groupby('uhf_code')['pop_in_intersection'].sum().reset_index()
    uhf_totals.columns = ['uhf_code', 'uhf_total_pop']
    
    crosswalk = crosswalk.merge(uhf_totals, on='uhf_code')
    crosswalk['weight'] = crosswalk['pop_in_intersection'] / crosswalk['uhf_total_pop']
    crosswalk['weight'] = crosswalk['weight'].fillna(0)
    
    # Normalize weights to sum to 1 per UHF
    weight_sums = crosswalk.groupby('uhf_code')['weight'].sum()
    logger.info(f"Weight sums per UHF (should be ~1): min={weight_sums.min():.3f}, max={weight_sums.max():.3f}")
    
    log_step_end(logger, "compute_spatial_intersection")
    return crosswalk


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info("=" * 60)
    logger.info("Building proper UHF→NTA crosswalk using real UHF boundaries")
    logger.info("=" * 60)
    
    try:
        # Load data
        uhf = load_uhf_boundaries(logger)
        nta = load_nta_boundaries(logger)
        nta_pop = load_nta_populations(logger)
        
        # Compute crosswalk
        crosswalk = compute_spatial_intersection(uhf, nta, nta_pop, logger)
        
        # Format output
        output = crosswalk[['uhf_code', 'geo_id', 'weight', 'area_fraction']].copy()
        output = output.rename(columns={
            'uhf_code': 'source_geo_id',
            'geo_id': 'target_geo_id',
        })
        
        # Write output
        output_path = paths.processed_xwalk / "uhf42_to_nta_pop_weights.parquet"
        atomic_write_parquet(output_path, output)
        logger.info(f"Wrote {output_path}")
        
        # Write metadata
        write_metadata_sidecar(
            output_path,
            run_id,
            parameters={
                'method': 'spatial_intersection_population_weighted',
                'uhf_source': 'nychealth/EHDP-data GitHub',
                'n_uhf': len(uhf),
                'n_nta': len(nta),
            },
            row_count=len(output),
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("UHF→NTA CROSSWALK SUMMARY:")
        logger.info(f"  UHF neighborhoods: {output['source_geo_id'].nunique()}")
        logger.info(f"  NTA neighborhoods: {output['target_geo_id'].nunique()}")
        logger.info(f"  Total mappings: {len(output)}")
        logger.info(f"  Mean NTAs per UHF: {len(output) / output['source_geo_id'].nunique():.1f}")
        logger.info("=" * 60)
        logger.info("This replaces the borough-based approximation!")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

