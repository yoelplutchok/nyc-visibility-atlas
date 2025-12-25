#!/usr/bin/env python3
"""
16_add_311_civic_layer.py

Add 311 complaints as a civic contrast layer.

Purpose: 311 is a NON-HEALTH data source that captures civic engagement.
If neighborhoods with low health surveillance visibility ALSO have high
311 engagement, it suggests they're not "invisible" — just invisible
to health systems specifically.

Note: 311 API only has data from 2020+, so we use 2020 for comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import get_logger, log_step_start, log_step_end, get_run_id
from visibility_atlas.io_utils import atomic_write_parquet


SCRIPT_NAME = "16_add_311_civic_layer"

# 311 API endpoint
API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Years available
YEARS = [2020, 2021, 2022]


def download_311_by_zip(year: int, logger: logging.Logger) -> pd.DataFrame:
    """Download 311 complaints aggregated by ZIP for a year."""
    
    log_step_start(logger, f"download_311_{year}")
    
    # Use server-side aggregation
    params = {
        "$select": "incident_zip, count(*) as complaint_count",
        "$where": f"created_date >= '{year}-01-01' AND created_date < '{year + 1}-01-01'",
        "$group": "incident_zip",
        "$limit": 50000,
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        df['year'] = year
        
        logger.info(f"Year {year}: Downloaded {len(df)} ZIP codes")
        log_step_end(logger, f"download_311_{year}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Error downloading year {year}: {e}")
        return pd.DataFrame()


def download_311_by_type(year: int, logger: logging.Logger) -> pd.DataFrame:
    """Download 311 complaints by type and ZIP."""
    
    # Top health-related complaint types
    health_types = [
        "Noise",
        "HEAT/HOT WATER",
        "Street Condition",
        "Sanitation",
        "Rodent",
        "Air Quality",
        "Asbestos",
        "Lead",
        "Mold",
    ]
    
    params = {
        "$select": "incident_zip, complaint_type, count(*) as n",
        "$where": f"created_date >= '{year}-01-01' AND created_date < '{year + 1}-01-01'",
        "$group": "incident_zip, complaint_type",
        "$order": "n DESC",
        "$limit": 100000,
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        df['year'] = year
        
        return df
        
    except Exception as e:
        logger.warning(f"Error downloading by type for {year}: {e}")
        return pd.DataFrame()


def load_zip_to_nta_crosswalk(logger: logging.Logger) -> pd.DataFrame:
    """Load ZIP to NTA crosswalk."""
    crosswalk_path = paths.processed_xwalk / "zcta_to_nta_pop_weights.parquet"
    
    if crosswalk_path.exists():
        df = pd.read_parquet(crosswalk_path)
        # Rename columns to expected format
        df = df.rename(columns={
            'source_geo_id': 'zip_code',
            'target_geo_id': 'geo_id',
        })
        logger.info(f"Loaded crosswalk with {len(df)} mappings")
        return df
    else:
        logger.warning("No ZIP-NTA crosswalk found")
        return pd.DataFrame()


def crosswalk_311_to_nta(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Crosswalk 311 data from ZIP to NTA."""
    
    if len(crosswalk) == 0:
        # Return ZIP-level data
        return df
    
    # Merge with crosswalk
    merged = df.merge(
        crosswalk[['zip_code', 'geo_id', 'weight']],
        left_on='incident_zip',
        right_on='zip_code',
        how='left'
    )
    
    # Apply weights
    merged['weighted_complaints'] = merged['complaint_count'].astype(float) * merged['weight'].fillna(1)
    
    # Aggregate to NTA
    nta_agg = merged.groupby(['year', 'geo_id']).agg({
        'weighted_complaints': 'sum',
    }).reset_index()
    
    nta_agg.columns = ['year', 'geo_id', 'complaints_311']
    
    return nta_agg


def compute_civic_visibility(df: pd.DataFrame, nta_pop: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Compute civic visibility (311 complaints per capita)."""
    
    if len(df) == 0 or 'geo_id' not in df.columns:
        return df
    
    # Merge with population
    if len(nta_pop) > 0 and 'reference_pop' in nta_pop.columns:
        # Filter to total stratum if present
        if 'stratum_id' in nta_pop.columns:
            nta_pop = nta_pop[nta_pop['stratum_id'] == 'total']
        
        merged = df.merge(nta_pop[['geo_id', 'reference_pop']], on='geo_id', how='left')
        merged['civic_visibility'] = merged['complaints_311'] / merged['reference_pop'] * 1000
    else:
        merged = df
        merged['civic_visibility'] = np.nan
    
    return merged


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Download 311 data
        all_years = []
        
        for year in YEARS:
            cache_file = paths.raw_311 / f"311_by_zip_{year}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached 311 data for {year}")
                df = pd.read_parquet(cache_file)
            else:
                df = download_311_by_zip(year, logger)
                if len(df) > 0:
                    ensure_dir(cache_file.parent)
                    atomic_write_parquet(cache_file, df)
            
            if len(df) > 0:
                all_years.append(df)
        
        if not all_years:
            logger.error("No 311 data downloaded")
            return 1
        
        combined = pd.concat(all_years, ignore_index=True)
        logger.info(f"Total 311 records: {len(combined)} ZIP×year combinations")
        
        # Load crosswalk
        crosswalk = load_zip_to_nta_crosswalk(logger)
        
        # Crosswalk to NTA
        nta_311 = crosswalk_311_to_nta(combined, crosswalk, logger)
        
        # Load NTA populations
        pop_path = paths.processed_denominators / "acs_denominators.parquet"
        if pop_path.exists():
            nta_pop = pd.read_parquet(pop_path)
        else:
            nta_pop = pd.DataFrame()
        
        # Compute civic visibility
        civic = compute_civic_visibility(nta_311, nta_pop, logger)
        
        # Write outputs
        output_path = paths.processed_visibility / "civic_311_visibility.parquet"
        atomic_write_parquet(output_path, civic if len(civic) > 0 else combined)
        logger.info(f"Wrote {output_path}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("311 CIVIC LAYER SUMMARY:")
        logger.info(f"  Years: {YEARS}")
        
        if len(combined) > 0:
            total_complaints = combined['complaint_count'].astype(int).sum()
            logger.info(f"  Total complaints: {total_complaints:,}")
            logger.info(f"  Unique ZIPs: {combined['incident_zip'].nunique()}")
        
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

