#!/usr/bin/env python3
"""
04_build_numerator_sparcs.py

Ingest SPARCS (Statewide Planning and Research Cooperative System) hospital
discharge data and compute healthcare system visibility.

Pipeline Step: 04
Contract Reference: Section 11 - 04_build_numerator_sparcs.py

SPARCS provides hospital inpatient discharges by patient ZIP code of residence.
We use the publicly available Prevention Quality Indicators (PQI) dataset from
health.data.ny.gov which provides rates per 100,000 people by ZIP code.

Inputs:
    - SPARCS PQI data from NYS Health Data (ZIP code level)
    - data/processed/denominators/acs_denominators.parquet
    - data/processed/xwalk/zcta_to_nta_pop_weights.parquet

Outputs:
    - data/processed/numerators/sparcs.parquet
    - data/processed/visibility/sparcs_visibility.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet,
    read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "04_build_numerator_sparcs"

# NYS Health Data - SPARCS PQI by ZIP Code (public dataset)
# Dataset: Hospital Inpatient Discharges (SPARCS De-Identified): Adult PQI by Zip Code
SPARCS_PQI_DATASET_ID = "5q8c-d6xq"
SPARCS_API_BASE = "https://health.data.ny.gov/resource"

# NYC ZIP code prefixes (NYC ZIPs start with 100xx, 104xx, 110xx-114xx, 116xx)
NYC_ZIP_PREFIXES = ['100', '101', '102', '103', '104', '110', '111', '112', '113', '114', '116']


def get_nyc_zip_codes() -> set:
    """Get set of NYC ZIP codes from our ZCTA crosswalk."""
    try:
        zcta_xwalk = read_parquet(paths.processed_xwalk / "zcta_to_nta_pop_weights.parquet")
        return set(zcta_xwalk['source_geo_id'].unique())
    except FileNotFoundError:
        # Fallback: use ZIP prefix matching
        return set()


def download_sparcs_pqi_data(logger: logging.Logger, year: int) -> pd.DataFrame:
    """
    Download SPARCS PQI data from NYS Health Data API.
    
    This uses the Socrata Open Data API (SODA) to fetch Prevention Quality
    Indicator rates by ZIP code.
    """
    log_step_start(logger, "download_sparcs_pqi_data", year=year)
    
    raw_dir = ensure_dir(paths.raw_sparcs)
    cache_file = raw_dir / f"sparcs_pqi_{year}.parquet"
    
    if cache_file.exists():
        logger.info(f"Using cached SPARCS data: {cache_file}")
        df = read_parquet(cache_file)
        log_step_end(logger, "download_sparcs_pqi_data", rows=len(df), source="cache")
        return df
    
    # Query the API with filters
    api_url = f"{SPARCS_API_BASE}/{SPARCS_PQI_DATASET_ID}.json"
    
    # Socrata API parameters
    # Get data for the specified year, limit to 50k per request (paginate if needed)
    all_records = []
    offset = 0
    limit = 50000
    
    logger.info(f"Fetching SPARCS PQI data for year {year}...")
    
    while True:
        params = {
            "$where": f"year = {year}",
            "$limit": limit,
            "$offset": offset,
            "$order": "patient_zipcode,pqi_number"
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_records.extend(data)
            logger.info(f"  Fetched {len(all_records)} records so far...")
            
            if len(data) < limit:
                break
                
            offset += limit
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    if not all_records:
        logger.warning(f"No SPARCS data found for year {year}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    
    # Convert numeric columns
    numeric_cols = ['year', 'observed_rate_per_100_000_people', 'expected_rate_per_100_000_people']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter to NYC ZIP codes
    nyc_zips = get_nyc_zip_codes()
    if nyc_zips:
        df = df[df['patient_zipcode'].isin(nyc_zips)]
        logger.info(f"Filtered to {len(df)} NYC records using ZCTA crosswalk")
    else:
        # Fallback: filter by prefix
        df = df[df['patient_zipcode'].str[:3].isin(NYC_ZIP_PREFIXES)]
        logger.info(f"Filtered to {len(df)} NYC records using ZIP prefix matching")
    
    # Cache the data
    atomic_write_parquet(cache_file, df)
    logger.info(f"Cached SPARCS data to {cache_file}")
    
    log_step_end(logger, "download_sparcs_pqi_data", 
                 rows=len(df), 
                 unique_zips=df['patient_zipcode'].nunique() if len(df) > 0 else 0)
    
    return df


def aggregate_sparcs_to_total_discharges(
    df: pd.DataFrame, 
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Aggregate PQI data to total discharges per ZIP code.
    
    The PQI dataset contains rates per 100,000 for specific conditions.
    We'll sum these to get a proxy for total ambulatory care sensitive
    hospitalizations, which represents healthcare system contact.
    """
    log_step_start(logger, "aggregate_sparcs_to_total_discharges")
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # PQI measures represent different condition categories
    # We'll keep them separate for detailed analysis but also compute totals
    logger.info(f"PQI measures in data: {df['pqi_name'].nunique()}")
    
    # Aggregate: sum of observed rates per ZIP (as a visibility proxy)
    # This represents total preventable hospitalizations per 100,000
    agg = df.groupby('patient_zipcode').agg({
        'observed_rate_per_100_000_people': 'sum',  # Sum all PQI rates
        'pqi_number': 'count'  # Number of PQI measures reported
    }).reset_index()
    
    agg = agg.rename(columns={
        'patient_zipcode': 'zip_code',
        'observed_rate_per_100_000_people': 'total_pqi_rate_per_100k',
        'pqi_number': 'pqi_count'
    })
    
    # Convert rate per 100k to rate per 1k for visibility calculation
    agg['rate_per_1000'] = agg['total_pqi_rate_per_100k'] / 100
    
    log_step_end(logger, "aggregate_sparcs_to_total_discharges",
                 zip_count=len(agg),
                 mean_rate=agg['rate_per_1000'].mean())
    
    return agg


def crosswalk_to_nta(
    sparcs_zip: pd.DataFrame,
    zcta_to_nta: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Convert ZIP-level SPARCS data to NTA using population-weighted crosswalk.
    """
    log_step_start(logger, "crosswalk_to_nta")
    
    if len(sparcs_zip) == 0:
        return pd.DataFrame()
    
    # Merge SPARCS data with crosswalk
    merged = sparcs_zip.merge(
        zcta_to_nta,
        left_on='zip_code',
        right_on='source_geo_id',
        how='inner'
    )
    
    logger.info(f"Matched {merged['zip_code'].nunique()} ZIPs to NTAs")
    
    if len(merged) == 0:
        logger.warning("No ZIPs matched to NTAs!")
        return pd.DataFrame()
    
    # Weight the rate by crosswalk weight
    # If a ZIP splits across NTAs, distribute the rate proportionally
    merged['weighted_rate'] = merged['rate_per_1000'] * merged['weight']
    
    # Aggregate to NTA
    # Sum weighted rates for each NTA
    nta_agg = merged.groupby('target_geo_id').agg({
        'weighted_rate': 'sum',
        'weight': 'sum',  # Should sum to ~1 if fully covered
        'zip_code': 'nunique'
    }).reset_index()
    
    nta_agg = nta_agg.rename(columns={
        'target_geo_id': 'geo_id',
        'weighted_rate': 'observed_rate',
        'weight': 'coverage_weight',
        'zip_code': 'source_zip_count'
    })
    
    log_step_end(logger, "crosswalk_to_nta",
                 nta_count=len(nta_agg),
                 mean_rate=nta_agg['observed_rate'].mean())
    
    return nta_agg


def compute_sparcs_visibility(
    sparcs_nta: pd.DataFrame,
    denominators: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute SPARCS healthcare system visibility.
    
    Visibility = observed rate (already per 1,000)
    
    Note: The PQI rates are already population-adjusted (per 100,000), so
    we convert to per 1,000 for consistency with our visibility metric.
    """
    log_step_start(logger, "compute_sparcs_visibility")
    
    if len(sparcs_nta) == 0:
        return pd.DataFrame()
    
    # Get total population denominators
    total_denom = denominators[denominators['stratum_id'] == 'total'].copy()
    total_denom = total_denom[['geo_id', 'reference_pop', 'time_window_id']]
    
    # Merge with SPARCS data
    visibility = sparcs_nta.merge(
        total_denom,
        on='geo_id',
        how='inner'
    )
    
    # The observed_rate is already per 1,000 (we converted from per 100k)
    # This represents healthcare system contact intensity
    visibility['visibility'] = visibility['observed_rate']
    
    # We can also compute an "observed count" estimate for context
    # observed_count = rate_per_1000 * pop / 1000 = rate_per_1000 * pop / 1000
    visibility['observed_count'] = (
        visibility['observed_rate'] * visibility['reference_pop'] / 1000
    )
    
    # Add metadata
    visibility['source_id'] = 'sparcs'
    visibility['stratum_id'] = 'total'
    visibility['numerator_type'] = 'encounters'  # PQI discharges
    visibility['indicator_id'] = 'pqi_total'
    
    # Reliability flags based on coverage and population
    # Flag as low reliability if coverage weight is low or population is small
    params = read_yaml(paths.configs / "params.yml")
    min_pop = params.get('small_numbers_policy', {}).get('min_denominator', 50)
    
    visibility['reliability_flag'] = 'high'
    visibility.loc[visibility['reference_pop'] < min_pop, 'reliability_flag'] = 'suppressed'
    visibility.loc[
        (visibility['coverage_weight'] < 0.5) & (visibility['reliability_flag'] != 'suppressed'),
        'reliability_flag'
    ] = 'low'
    
    # Keep only needed columns
    output_cols = [
        'geo_id', 'stratum_id', 'time_window_id', 'source_id',
        'observed_count', 'reference_pop', 'visibility',
        'reliability_flag', 'numerator_type', 'indicator_id',
        'coverage_weight', 'source_zip_count'
    ]
    
    visibility = visibility[output_cols]
    
    log_step_end(logger, "compute_sparcs_visibility",
                 nta_count=len(visibility),
                 mean_visibility=visibility['visibility'].mean(),
                 high_reliability=sum(visibility['reliability_flag'] == 'high'),
                 low_reliability=sum(visibility['reliability_flag'] == 'low'),
                 suppressed=sum(visibility['reliability_flag'] == 'suppressed'))
    
    return visibility


def create_indicator_level_data(
    raw_df: pd.DataFrame,
    zcta_to_nta: pd.DataFrame,
    denominators: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Create indicator-level SPARCS data (each PQI measure separately).
    This is saved as the raw numerator file.
    """
    log_step_start(logger, "create_indicator_level_data")
    
    if len(raw_df) == 0:
        return pd.DataFrame()
    
    # Create indicator-level aggregation by ZIP
    indicator_data = raw_df.groupby(['patient_zipcode', 'pqi_number', 'pqi_name']).agg({
        'observed_rate_per_100_000_people': 'first',
        'expected_rate_per_100_000_people': 'first'
    }).reset_index()
    
    indicator_data['rate_per_1000'] = indicator_data['observed_rate_per_100_000_people'] / 100
    
    # Crosswalk to NTA
    merged = indicator_data.merge(
        zcta_to_nta,
        left_on='patient_zipcode',
        right_on='source_geo_id',
        how='inner'
    )
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Weight rates
    merged['weighted_rate'] = merged['rate_per_1000'] * merged['weight']
    
    # Aggregate by NTA and indicator
    nta_indicators = merged.groupby(['target_geo_id', 'pqi_number', 'pqi_name']).agg({
        'weighted_rate': 'sum',
        'weight': 'sum',
        'observed_rate_per_100_000_people': 'mean',
        'expected_rate_per_100_000_people': 'mean'
    }).reset_index()
    
    nta_indicators = nta_indicators.rename(columns={
        'target_geo_id': 'geo_id',
        'pqi_number': 'indicator_id',
        'pqi_name': 'indicator_name',
        'weighted_rate': 'rate_per_1000',
        'weight': 'coverage_weight',
        'observed_rate_per_100_000_people': 'observed_rate_per_100k',
        'expected_rate_per_100_000_people': 'expected_rate_per_100k'
    })
    
    # Add metadata
    nta_indicators['source_id'] = 'sparcs'
    nta_indicators['numerator_type'] = 'encounters'
    
    log_step_end(logger, "create_indicator_level_data",
                 rows=len(nta_indicators),
                 indicators=nta_indicators['indicator_id'].nunique(),
                 ntas=nta_indicators['geo_id'].nunique())
    
    return nta_indicators


def main():
    """Main pipeline function."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load config
        params = read_yaml(paths.configs / "params.yml")
        time_window = params.get('time_windows', {}).get('primary', {})
        # Use a year within our time window (2019 matches CHS)
        year = 2019
        
        # Ensure output directories exist
        ensure_dir(paths.processed_numerators)
        ensure_dir(paths.processed_visibility)
        ensure_dir(paths.raw_sparcs)
        
        # Load dependencies
        logger.info("Loading dependencies...")
        
        denominators = read_parquet(paths.processed_denominators / "acs_denominators.parquet")
        logger.info(f"Loaded {len(denominators)} denominator records")
        
        try:
            zcta_to_nta = read_parquet(paths.processed_xwalk / "zcta_to_nta_pop_weights.parquet")
            logger.info(f"Loaded ZCTA→NTA crosswalk: {len(zcta_to_nta)} pairs")
        except FileNotFoundError:
            logger.error("ZCTA→NTA crosswalk not found. Run 01_build_crosswalks.py first.")
            raise
        
        # Step 1: Download SPARCS PQI data
        logger.info(f"\n{'='*60}")
        logger.info("Step 1: Download SPARCS PQI data")
        logger.info(f"{'='*60}")
        
        raw_df = download_sparcs_pqi_data(logger, year)
        
        if len(raw_df) == 0:
            logger.error("No SPARCS data downloaded!")
            raise RuntimeError("SPARCS download failed")
        
        logger.info(f"\nSPARCS data summary:")
        logger.info(f"  Records: {len(raw_df)}")
        logger.info(f"  ZIP codes: {raw_df['patient_zipcode'].nunique()}")
        logger.info(f"  PQI measures: {raw_df['pqi_name'].nunique()}")
        
        # Log sample of PQI measures
        logger.info(f"\nPQI measures found:")
        for pqi in raw_df['pqi_name'].unique()[:10]:
            logger.info(f"  - {pqi}")
        
        # Step 2: Create indicator-level data (raw numerator file)
        logger.info(f"\n{'='*60}")
        logger.info("Step 2: Create indicator-level data")
        logger.info(f"{'='*60}")
        
        indicator_data = create_indicator_level_data(
            raw_df, zcta_to_nta, denominators, logger
        )
        
        if len(indicator_data) > 0:
            numerator_file = paths.processed_numerators / "sparcs.parquet"
            atomic_write_parquet(numerator_file, indicator_data)
            log_output_written(logger, numerator_file, len(indicator_data))
        
        # Step 3: Aggregate to total discharges per ZIP
        logger.info(f"\n{'='*60}")
        logger.info("Step 3: Aggregate to total discharges")
        logger.info(f"{'='*60}")
        
        sparcs_zip = aggregate_sparcs_to_total_discharges(raw_df, logger)
        
        # Step 4: Crosswalk to NTA
        logger.info(f"\n{'='*60}")
        logger.info("Step 4: Crosswalk to NTA geography")
        logger.info(f"{'='*60}")
        
        sparcs_nta = crosswalk_to_nta(sparcs_zip, zcta_to_nta, logger)
        
        # Step 5: Compute visibility
        logger.info(f"\n{'='*60}")
        logger.info("Step 5: Compute healthcare system visibility")
        logger.info(f"{'='*60}")
        
        visibility = compute_sparcs_visibility(sparcs_nta, denominators, logger)
        
        if len(visibility) == 0:
            logger.error("No visibility data computed!")
            raise RuntimeError("Visibility computation failed")
        
        # Write visibility output
        visibility_file = paths.processed_visibility / "sparcs_visibility.parquet"
        atomic_write_parquet(visibility_file, visibility)
        log_output_written(logger, visibility_file, len(visibility))
        
        # Write metadata
        metadata = {
            "run_id": run_id,
            "script": SCRIPT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "year": year,
            "dataset_id": SPARCS_PQI_DATASET_ID,
            "source": "NYS Health Data - SPARCS PQI by ZIP Code",
            "numerator_type": "encounters",
            "numerator_semantic": "Hospital inpatient discharges for Prevention Quality Indicators",
            "visibility_interpretation": "Healthcare system contact intensity (PQI hospitalizations per 1,000 residents)",
            "nta_count": len(visibility),
            "mean_visibility": float(visibility['visibility'].mean()),
            "median_visibility": float(visibility['visibility'].median()),
            "reliability_summary": {
                "high": int(sum(visibility['reliability_flag'] == 'high')),
                "low": int(sum(visibility['reliability_flag'] == 'low')),
                "suppressed": int(sum(visibility['reliability_flag'] == 'suppressed'))
            }
        }
        
        metadata_file = paths.processed_visibility / "sparcs_visibility_metadata.json"
        atomic_write_json(metadata_file, metadata)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("SPARCS Pipeline Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"\n✅ Summary:")
        logger.info(f"   Year: {year}")
        logger.info(f"   NTAs with visibility: {len(visibility)}")
        logger.info(f"   Mean visibility: {visibility['visibility'].mean():.2f} per 1,000")
        logger.info(f"   Median visibility: {visibility['visibility'].median():.2f} per 1,000")
        logger.info(f"   Range: {visibility['visibility'].min():.2f} - {visibility['visibility'].max():.2f}")
        logger.info(f"\n   Reliability breakdown:")
        logger.info(f"     High: {sum(visibility['reliability_flag'] == 'high')}")
        logger.info(f"     Low: {sum(visibility['reliability_flag'] == 'low')}")
        logger.info(f"     Suppressed: {sum(visibility['reliability_flag'] == 'suppressed')}")
        
        logger.info(f"\n✅ Outputs:")
        logger.info(f"   {numerator_file}")
        logger.info(f"   {visibility_file}")
        logger.info(f"   {metadata_file}")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

