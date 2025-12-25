#!/usr/bin/env python3
"""
04b_build_numerator_sparcs_encounters.py

FIXED SPARCS visibility using actual encounter counts (not summed PQI rates).

This addresses the critique: "Summing PQI rates is not defensible"

Instead of using PQI rates (which are condition-specific and overlapping),
we use the actual discharge/encounter counts from the full SPARCS dataset.

This gives us "healthcare system contact intensity" which is a cleaner
construct than "sum of preventable hospitalization rates."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import get_logger, log_step_start, log_step_end, get_run_id
from visibility_atlas.io_utils import atomic_write_parquet
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "04b_build_numerator_sparcs_encounters"

# Map 3-digit ZIP prefixes to boroughs for crosswalk
ZIP_PREFIX_TO_BOROUGH = {
    '100': 'Manhattan',
    '101': 'Manhattan', 
    '102': 'Manhattan',
    '103': 'Staten Island',
    '104': 'Bronx',
    '110': 'Queens',
    '111': 'Queens',
    '112': 'Brooklyn',
    '113': 'Brooklyn',
    '114': 'Queens',
    '116': 'Queens',
}


def load_sparcs_encounters(logger: logging.Logger) -> pd.DataFrame:
    """Load cached SPARCS encounter data from multi-year download."""
    log_step_start(logger, "load_sparcs_encounters")
    
    all_data = []
    sparcs_dir = paths.raw_sparcs
    
    for year in [2017, 2018, 2019, 2021]:
        cache_file = sparcs_dir / f"sparcs_nyc_{year}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df['year'] = year
            all_data.append(df)
            logger.info(f"Loaded {year}: {len(df)} records")
    
    if not all_data:
        logger.error("No SPARCS encounter data found")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total encounters: {len(combined)}")
    
    log_step_end(logger, "load_sparcs_encounters")
    return combined


def aggregate_by_zip_prefix(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Aggregate encounters by 3-digit ZIP prefix."""
    log_step_start(logger, "aggregate_by_zip")
    
    # Filter to 2019 for consistency with other sources
    df_2019 = df[df['year'] == 2019].copy()
    logger.info(f"2019 records: {len(df_2019)}")
    
    # Count encounters and ED visits by ZIP prefix
    agg = df_2019.groupby('zip_code_3_digits').agg({
        'emergency_department_indicator': lambda x: (x == True).sum(),  # ED visits
        'year': 'count',  # Total encounters
    }).reset_index()
    
    agg.columns = ['zip_prefix', 'ed_visits', 'total_encounters']
    agg['inpatient_only'] = agg['total_encounters'] - agg['ed_visits']
    
    # Add borough mapping
    agg['borough'] = agg['zip_prefix'].map(ZIP_PREFIX_TO_BOROUGH)
    
    # Filter to NYC ZIPs only
    agg = agg[agg['borough'].notna()].copy()
    
    logger.info(f"Aggregated to {len(agg)} ZIP prefixes")
    log_step_end(logger, "aggregate_by_zip")
    
    return agg


def load_nta_reference(logger: logging.Logger) -> pd.DataFrame:
    """Load NTA reference data with populations and boroughs."""
    
    geo_path = paths.processed_geo / "nta_canonical.parquet"
    pop_path = paths.processed_denominators / "acs_denominators.parquet"
    
    geo = pd.read_parquet(geo_path)
    pop = pd.read_parquet(pop_path)
    
    # Filter pop to total stratum
    pop_total = pop[pop['stratum_id'] == 'total'][['geo_id', 'reference_pop']].copy()
    
    # Merge
    ntas = geo.merge(pop_total, on='geo_id', how='left')
    
    logger.info(f"Loaded {len(ntas)} NTAs with populations")
    return ntas


def crosswalk_to_nta_borough_weighted(
    zip_agg: pd.DataFrame,
    ntas: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Crosswalk ZIP-level data to NTA using borough-based population weights.
    
    Since we only have 3-digit ZIP prefixes (not full 5-digit ZIPs),
    we distribute encounters to NTAs within each borough proportionally
    to NTA population.
    
    NOTE: This is an approximation. The review correctly notes that
    real UHF boundaries would be better. This limitation is documented.
    """
    log_step_start(logger, "crosswalk_to_nta")
    
    # Calculate borough totals from ZIP data
    borough_encounters = zip_agg.groupby('borough').agg({
        'total_encounters': 'sum',
        'ed_visits': 'sum',
    }).reset_index()
    
    # Calculate borough population totals from NTAs
    borough_pop = ntas.groupby('borough')['reference_pop'].sum().reset_index()
    borough_pop.columns = ['borough', 'borough_pop']
    
    # Merge NTAs with borough population totals
    ntas_with_weight = ntas.merge(borough_pop, on='borough', how='left')
    ntas_with_weight['pop_weight'] = ntas_with_weight['reference_pop'] / ntas_with_weight['borough_pop']
    
    # Merge with borough encounter totals
    nta_encounters = ntas_with_weight.merge(borough_encounters, on='borough', how='left')
    
    # Distribute encounters proportionally to NTA population
    nta_encounters['nta_encounters'] = nta_encounters['total_encounters'] * nta_encounters['pop_weight']
    nta_encounters['nta_ed_visits'] = nta_encounters['ed_visits'] * nta_encounters['pop_weight']
    
    logger.info(f"Crosswalked to {len(nta_encounters)} NTAs")
    log_step_end(logger, "crosswalk_to_nta")
    
    return nta_encounters


def compute_encounter_visibility(
    nta_encounters: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute healthcare visibility as encounters per 1,000 population.
    
    This is a clean, interpretable metric:
    "How many hospital encounters per 1,000 residents?"
    
    Unlike summed PQI rates, this has clear meaning and no overlap issues.
    """
    log_step_start(logger, "compute_visibility")
    
    visibility = nta_encounters.copy()
    
    # Encounters per 1,000 population
    visibility['visibility'] = (
        visibility['nta_encounters'] / visibility['reference_pop'] * 1000
    )
    
    # ED rate
    visibility['ed_visibility'] = (
        visibility['nta_ed_visits'] / visibility['reference_pop'] * 1000
    )
    
    # Reliability flag based on count
    visibility['reliability_flag'] = np.where(
        visibility['nta_encounters'] < 50, 'suppressed',
        np.where(visibility['nta_encounters'] < 200, 'low', 'high')
    )
    
    # Format for output
    output = visibility[['geo_id', 'reference_pop', 'nta_encounters', 'nta_ed_visits', 
                         'visibility', 'ed_visibility', 'reliability_flag', 'borough']].copy()
    output = output.rename(columns={
        'nta_encounters': 'observed_count',
        'nta_ed_visits': 'ed_count',
    })
    
    # Add standard columns
    output['stratum_id'] = 'total'
    output['time_window_id'] = '2019'
    output['source_id'] = 'sparcs_encounters'
    output['numerator_type'] = 'encounters'
    
    logger.info(f"Visibility range: {output['visibility'].min():.2f} - {output['visibility'].max():.2f}")
    logger.info(f"Mean visibility: {output['visibility'].mean():.2f} encounters per 1,000")
    
    log_step_end(logger, "compute_visibility")
    return output


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info("=" * 60)
    logger.info("This script fixes the SPARCS visibility calculation")
    logger.info("by using actual encounter counts instead of summed PQI rates.")
    logger.info("=" * 60)
    
    try:
        # Load data
        encounters = load_sparcs_encounters(logger)
        if len(encounters) == 0:
            return 1
        
        # Aggregate by ZIP
        zip_agg = aggregate_by_zip_prefix(encounters, logger)
        
        # Load NTA reference
        ntas = load_nta_reference(logger)
        
        # Crosswalk to NTA
        nta_encounters = crosswalk_to_nta_borough_weighted(zip_agg, ntas, logger)
        
        # Compute visibility
        visibility = compute_encounter_visibility(nta_encounters, logger)
        
        # Write output
        output_path = paths.processed_visibility / "sparcs_encounters_visibility.parquet"
        atomic_write_parquet(output_path, visibility)
        logger.info(f"Wrote {output_path}")
        
        # Write metadata
        write_metadata_sidecar(
            output_path,
            run_id,
            parameters={
                'method': 'encounter_counts',
                'year': 2019,
                'note': 'Uses actual encounters, not summed PQI rates',
            },
            row_count=len(visibility),
        )
        
        # Summary stats
        logger.info("=" * 60)
        logger.info("FIXED SPARCS VISIBILITY (Encounters):")
        logger.info(f"  NTAs: {len(visibility)}")
        logger.info(f"  Mean encounters/1000: {visibility['visibility'].mean():.2f}")
        logger.info(f"  Mean ED visits/1000: {visibility['ed_visibility'].mean():.2f}")
        logger.info(f"  Reliability: {visibility['reliability_flag'].value_counts().to_dict()}")
        logger.info("=" * 60)
        logger.info("NOTE: This replaces the old summed-PQI approach")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

