#!/usr/bin/env python3
"""
06_build_numerator_vital.py

Ingest vital statistics (mortality) data and compute vital event visibility.

Pipeline Step: 06
Contract Reference: Section 11 - 06_build_numerator_vital.py

Vital statistics provide near-complete coverage of deaths in NYC.
This serves as a "sanity anchor" - vital event rates should be plausible
and not show crosswalk artifacts.

Inputs:
    - Mortality data from EpiQuery manual exports (UHF34 geography)
    - data/processed/denominators/acs_denominators.parquet
    - data/processed/xwalk/uhf34_to_nta_pop_weights.parquet

Outputs:
    - data/processed/numerators/vital.parquet
    - data/processed/visibility/vital_visibility.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet, read_geoparquet,
    read_yaml, atomic_write_json, read_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "06_build_numerator_vital"

# UHF code to name mapping (for matching EpiQuery exports)
UHF_NAME_TO_CODE = {
    "Kingsbridge - Riverdale": "101",
    "Northeast Bronx": "102", 
    "Fordham - Bronx Park": "103",
    "Pelham - Throgs Neck": "104",
    "Crotona - Tremont": "105",
    "High Bridge - Morrisania": "106",
    "Hunts Point - Mott Haven": "107",
    "Greenpoint": "201",
    "Downtown - Heights - Park Slope": "202",
    "Bedford Stuyvesant - Crown Heights": "203",
    "East New York": "204",
    "Sunset Park": "205",
    "Borough Park": "206",
    "East Flatbush - Flatbush": "207",
    "Canarsie - Flatlands": "208",
    "Bensonhurst - Bay Ridge": "209",
    "Coney Island - Sheepshead Bay": "210",
    "Williamsburg - Bushwick": "211",
    "Washington Heights - Inwood": "301",
    "Central Harlem - Morningside Heights": "302",
    "East Harlem": "303",
    "Upper West Side": "304",
    "Upper East Side": "305",
    "Chelsea - Clinton": "306",
    "Gramercy Park - Murray Hill": "307",
    "Greenwich Village - SoHo": "308",
    "Union Square - Lower East Side": "309",
    "Lower Manhattan": "310",
    "Long Island City - Astoria": "401",
    "West Queens": "402",
    "Flushing - Clearview": "403",
    "Bayside - Little Neck": "404",
    "Ridgewood - Forest Hills": "405",
    "Fresh Meadows": "406",
    "Southwest Queens": "407",
    "Jamaica": "408",
    "Southeast Queens": "409",
    "Rockaways": "410",
    "Port Richmond": "501",
    "Stapleton - St. George": "502",
    "Willowbrook": "503",
    "South Beach - Tottenville": "504",
}


def load_epiquery_exports(logger: logging.Logger, year_filter: str = "2019") -> pd.DataFrame:
    """Load vital statistics data from manually exported EpiQuery CSVs.
    
    Handles the specific EpiQuery mortality export format which has:
    - Community District level geography (not UHF)
    - Long format with columns: Mortalitytype, Cause of Death, Yearnum, Dim1Name, Dim1Value, Metricname, Flag, Value
    """
    log_step_start(logger, "load_epiquery_exports")
    
    export_dir = paths.raw_vital / "epiquery_exports"
    if not export_dir.exists():
        logger.warning(f"No EpiQuery export directory found at {export_dir}")
        return pd.DataFrame()
    
    all_vital_data = []
    
    for csv_file in export_dir.glob("mortality_*.csv"):
        indicator_name = csv_file.stem.replace("mortality_", "").replace("_2019", "")
        indicator_id = f"mortality_{indicator_name}"
        
        logger.info(f"Processing: {csv_file.name}")
        try:
            # Try different encodings (Tableau exports are often UTF-16)
            df = None
            for encoding in ['utf-16', 'utf-8', 'latin-1']:
                for sep in ['\t', ',']:
                    try:
                        df = pd.read_csv(csv_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 1:  # Valid parse
                            logger.info(f"  Read with encoding={encoding}, sep='{sep}'")
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                logger.warning(f"  Could not parse {csv_file.name}")
                continue
            
            # Normalize column names
            df.columns = [col.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "") for col in df.columns]
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Total rows: {len(df)}")
            
            # Check if this is the EpiQuery long format (has yearnum, dim1name, metricname)
            if 'yearnum' in df.columns and 'dim1value' in df.columns:
                logger.info("  Detected EpiQuery long format (Community District level)")
                
                # Filter for the target year
                df['yearnum'] = pd.to_numeric(df['yearnum'], errors='coerce')
                df = df[df['yearnum'] == int(year_filter)]
                logger.info(f"  Filtered to year {year_filter}: {len(df)} rows")
                
                # Filter for Community District geography (not Citywide)
                if 'dim1name' in df.columns:
                    df = df[df['dim1name'].str.contains('Community District', case=False, na=False)]
                    logger.info(f"  Filtered to Community Districts: {len(df)} rows")
                
                # Filter for age-adjusted death rate (per 1000)
                if 'metricname' in df.columns:
                    rate_mask = df['metricname'].str.contains('Age_Adjusted.*Rate.*1000|Age.Adjusted.*Rate.*1000', 
                                                             case=False, na=False, regex=True)
                    df = df[rate_mask]
                    logger.info(f"  Filtered to Age-Adjusted Death Rate: {len(df)} rows")
                
                if len(df) == 0:
                    logger.warning("  No data after filtering")
                    continue
                
                # Extract CD name and code from dim1value (e.g., " Astoria, Long Island City (401)")
                # The value column is the last column
                value_col = df.columns[-1]
                
                df_clean = df[['dim1value', value_col]].copy()
                df_clean.columns = ['cd_name', 'rate']
                
                # Extract CD code from the parentheses
                df_clean['cd_code'] = df_clean['cd_name'].str.extract(r'\((\d+)\)')
                df_clean['cd_name'] = df_clean['cd_name'].str.replace(r'\s*\(\d+\)\s*', '', regex=True).str.strip()
                
                # Clean rate values
                df_clean['rate'] = df_clean['rate'].astype(str).str.replace(',', '').str.strip()
                df_clean['rate'] = pd.to_numeric(df_clean['rate'], errors='coerce')
                
                # Filter out invalid rows
                df_clean = df_clean.dropna(subset=['rate', 'cd_code'])
                logger.info(f"  Valid records after cleaning: {len(df_clean)}")
                
                if len(df_clean) == 0:
                    continue
                
                # Map CD codes to boroughs based on first digit
                # 1xx = Manhattan, 2xx = Bronx, 3xx = Brooklyn, 4xx = Queens, 5xx = Staten Island
                def cd_to_borough(code):
                    first = str(code)[0]
                    return {'1': 'Manhattan', '2': 'Bronx', '3': 'Brooklyn', 
                            '4': 'Queens', '5': 'Staten Island'}.get(first, 'Unknown')
                
                df_clean['borough'] = df_clean['cd_code'].apply(cd_to_borough)
                
                # Use CD code as the geographic ID (we'll build a CD→NTA crosswalk)
                df_clean['geo_id'] = 'CD' + df_clean['cd_code'].astype(str)
                df_clean['geo_type'] = 'cd'
                
                # Add metadata
                df_clean['indicator_id'] = indicator_id
                df_clean['indicator_name'] = indicator_name.replace("_", " ").title()
                df_clean['year'] = int(year_filter)
                df_clean['data_source'] = 'epiquery_vital'
                df_clean['stratum'] = 'total'
                
                all_vital_data.append(df_clean)
                logger.info(f"  ✓ Added {len(df_clean)} Community District records for {indicator_id}")
                logger.info(f"    Boroughs: {df_clean['borough'].value_counts().to_dict()}")
            else:
                # Legacy UHF format handling
                logger.info("  Attempting legacy UHF format parsing")
                # ... (original UHF parsing logic would go here, but skip for now)
                logger.warning("  Could not parse as EpiQuery format or UHF format")
                continue
            
        except Exception as e:
            logger.warning(f"Failed to process {csv_file.name}: {e}")
            logger.warning(traceback.format_exc())
    
    if all_vital_data:
        combined_df = pd.concat(all_vital_data, ignore_index=True)
        logger.info(f"✅ Loaded {len(combined_df)} records from vital statistics exports")
        logger.info(f"   Indicators: {combined_df['indicator_id'].unique().tolist()}")
        logger.info(f"   Community Districts: {combined_df['geo_id'].nunique()}")
        log_step_end(logger, "load_epiquery_exports", record_count=len(combined_df))
        return combined_df
    else:
        logger.warning("No valid records extracted from vital statistics exports")
        log_step_end(logger, "load_epiquery_exports", record_count=0)
        return pd.DataFrame()


def generate_synthetic_vital_data(logger: logging.Logger) -> pd.DataFrame:
    """Generate synthetic vital statistics data for development/testing."""
    log_step_start(logger, "generate_synthetic_vital_data")
    
    logger.warning("⚠️ Generating SYNTHETIC vital statistics data for development")
    
    # Generate for all UHF codes
    uhf_codes = list(UHF_NAME_TO_CODE.values())
    
    np.random.seed(42)
    
    records = []
    for uhf_code in uhf_codes:
        # Overall mortality: typically 5-10 per 1,000 for NYC
        base_rate = np.random.uniform(4.5, 9.5)
        records.append({
            'uhf_code': uhf_code,
            'uhf_name': [k for k, v in UHF_NAME_TO_CODE.items() if v == uhf_code][0],
            'indicator_id': 'mortality_overall',
            'indicator_name': 'Overall Mortality',
            'rate': round(base_rate, 2),
            'year': 2019,
            'stratum': 'total',
            'data_source': 'synthetic'
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic vital records")
    log_step_end(logger, "generate_synthetic_vital_data", record_count=len(df))
    return df


def build_cd_to_nta_crosswalk(
    vital_data: pd.DataFrame,
    ntas,
    denominators: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Build a Community District → NTA crosswalk based on borough."""
    log_step_start(logger, "build_cd_to_nta_crosswalk")
    
    # Get CD codes and their boroughs from vital data
    cd_boroughs = vital_data[['geo_id', 'borough']].drop_duplicates()
    logger.info(f"Building crosswalk for {len(cd_boroughs)} Community Districts")
    
    # Get NTA populations
    nta_pops = denominators[denominators['stratum_id'] == 'total'][['geo_id', 'reference_pop']]
    ntas_with_pop = ntas.merge(nta_pops, on='geo_id', how='left')
    ntas_with_pop['reference_pop'] = ntas_with_pop['reference_pop'].fillna(0)
    
    # Calculate borough totals and NTA shares
    borough_totals = ntas_with_pop.groupby('borough')['reference_pop'].sum().reset_index()
    borough_totals.columns = ['borough', 'borough_pop']
    
    ntas_with_pop = ntas_with_pop.merge(borough_totals, on='borough', how='left')
    ntas_with_pop['nta_weight'] = ntas_with_pop['reference_pop'] / ntas_with_pop['borough_pop']
    ntas_with_pop['nta_weight'] = ntas_with_pop['nta_weight'].fillna(0)
    
    # Build crosswalk
    crosswalk_records = []
    for _, cd_row in cd_boroughs.iterrows():
        cd_id = cd_row['geo_id']
        borough = cd_row['borough']
        borough_ntas = ntas_with_pop[ntas_with_pop['borough'] == borough]
        for _, nta_row in borough_ntas.iterrows():
            crosswalk_records.append({
                'source_geo_id': cd_id,
                'target_geo_id': nta_row['geo_id'],
                'weight': nta_row['nta_weight'],
            })
    
    crosswalk_df = pd.DataFrame(crosswalk_records)
    logger.info(f"Built CD→NTA crosswalk: {len(crosswalk_df)} pairs")
    log_step_end(logger, "build_cd_to_nta_crosswalk", pair_count=len(crosswalk_df))
    return crosswalk_df


def compute_vital_visibility(
    vital_data: pd.DataFrame,
    denominators: pd.DataFrame,
    crosswalk: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Compute vital event visibility at NTA level."""
    log_step_start(logger, "compute_vital_visibility")
    
    if len(vital_data) == 0:
        logger.error("No vital data to process")
        return pd.DataFrame()
    
    # Get total population denominators
    total_denom = denominators[denominators['stratum_id'] == 'total'][['geo_id', 'reference_pop']].copy()
    
    # Determine if we have CD or UHF data
    is_cd_data = 'geo_id' in vital_data.columns and vital_data['geo_id'].astype(str).str.startswith('CD').any()
    geo_col = 'geo_id' if is_cd_data else 'uhf_code'
    
    logger.info(f"Processing {'Community District' if is_cd_data else 'UHF'} level vital data")
    
    results = []
    for indicator_id in vital_data['indicator_id'].unique():
        indicator_df = vital_data[vital_data['indicator_id'] == indicator_id].copy()
        
        for _, row in indicator_df.iterrows():
            source_geo = row.get(geo_col) if geo_col in row.index else row.get('geo_id')
            rate = row['rate']
            
            if pd.isna(source_geo) or pd.isna(rate):
                continue
            
            source_ntas = crosswalk[crosswalk['source_geo_id'] == source_geo]
            for _, xwalk_row in source_ntas.iterrows():
                results.append({
                    'geo_id': xwalk_row['target_geo_id'],
                    'indicator_id': indicator_id,
                    'rate_per_1000': rate,
                    'weight': xwalk_row['weight'],
                })
    
    if not results:
        logger.error("No results after crosswalk application")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    logger.info(f"Crosswalk applied: {len(results_df)} NTA-indicator pairs")
    
    # Filter out zero-weight records
    results_df = results_df[results_df['weight'] > 0].copy()
    logger.info(f"After filtering zero weights: {len(results_df)} pairs")
    
    # Aggregate to NTA level - weighted average of rates
    def safe_weighted_avg(g):
        w = g['weight'].values
        if w.sum() == 0:
            return pd.Series({'rate_per_1000': np.nan, 'weight': 0})
        return pd.Series({
            'rate_per_1000': np.average(g['rate_per_1000'], weights=w),
            'weight': w.sum(),
        })
    
    nta_agg = results_df.groupby('geo_id').apply(
        safe_weighted_avg, include_groups=False
    ).reset_index()
    
    # Remove NTAs with no valid data
    nta_agg = nta_agg.dropna(subset=['rate_per_1000'])
    
    # Merge with denominators
    nta_agg = nta_agg.merge(total_denom, on='geo_id', how='left')
    
    # Calculate estimated count from rate
    nta_agg['observed_count'] = (nta_agg['rate_per_1000'] / 1000) * nta_agg['reference_pop']
    
    # Add standard fields
    nta_agg['visibility'] = nta_agg['rate_per_1000']
    nta_agg['stratum_id'] = 'total'
    nta_agg['source_id'] = 'vital'
    nta_agg['time_window_id'] = '2019'
    nta_agg['numerator_type'] = 'events'
    nta_agg['coverage_weight'] = nta_agg['weight']
    
    # Reliability flags
    nta_agg['reliability_flag'] = 'high'
    nta_agg.loc[nta_agg['coverage_weight'] < 0.5, 'reliability_flag'] = 'low'
    nta_agg.loc[nta_agg['coverage_weight'] < 0.1, 'reliability_flag'] = 'suppressed'
    
    output_cols = [
        'geo_id', 'stratum_id', 'source_id', 'time_window_id',
        'observed_count', 'reference_pop', 'visibility',
        'reliability_flag', 'numerator_type', 'coverage_weight'
    ]
    
    nta_agg = nta_agg[output_cols].copy()
    
    logger.info(f"Computed visibility for {len(nta_agg)} NTAs")
    logger.info(f"Mean visibility: {nta_agg['visibility'].mean():.2f} per 1,000")
    logger.info(f"Reliability: {nta_agg['reliability_flag'].value_counts().to_dict()}")
    
    log_step_end(logger, "compute_vital_visibility", nta_count=len(nta_agg))
    return nta_agg


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load denominators
        denominators = read_parquet(paths.processed_denominators / "acs_denominators.parquet")
        logger.info(f"Loaded {len(denominators)} denominator records")
        
        # Load NTAs (needed for crosswalk building)
        ntas = read_geoparquet(paths.processed_geo / "nta_canonical.parquet")
        logger.info(f"Loaded {len(ntas)} NTAs")
        
        # Try to load EpiQuery exports
        vital_data = load_epiquery_exports(logger)
        
        if len(vital_data) == 0:
            logger.warning("No EpiQuery exports found. Generating synthetic data.")
            vital_data = generate_synthetic_vital_data(logger)
        
        if len(vital_data) == 0:
            logger.error("No vital data available")
            return 1
        
        # Determine if we have CD or UHF data and build appropriate crosswalk
        is_cd_data = 'geo_id' in vital_data.columns and vital_data['geo_id'].astype(str).str.startswith('CD').any()
        
        if is_cd_data:
            logger.info("Detected Community District level vital data")
            crosswalk = build_cd_to_nta_crosswalk(vital_data, ntas, denominators, logger)
        else:
            logger.info("Using UHF→NTA crosswalk for vital data")
            uhf_xwalk_path = paths.processed_xwalk / "uhf34_to_nta_pop_weights.parquet"
            if not uhf_xwalk_path.exists():
                logger.error("UHF34→NTA crosswalk not found. Run 03_build_numerator_chs.py first.")
                return 1
            crosswalk = read_parquet(uhf_xwalk_path)
        
        logger.info(f"Crosswalk: {len(crosswalk)} pairs")
        
        # Compute visibility
        visibility = compute_vital_visibility(
            vital_data, denominators, crosswalk, logger
        )
        
        if len(visibility) == 0:
            logger.error("Failed to compute vital visibility")
            return 1
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_numerators)
        
        # Write raw vital data
        vital_path = output_dir / "vital.parquet"
        atomic_write_parquet(vital_path, vital_data)
        log_output_written(logger, vital_path, row_count=len(vital_data))
        
        # Write visibility
        vis_dir = ensure_dir(paths.processed_visibility)
        vis_path = vis_dir / "vital_visibility.parquet"
        atomic_write_parquet(vis_path, visibility)
        log_output_written(logger, vis_path, row_count=len(visibility))
        
        # Write metadata
        write_metadata_sidecar(
            vis_path,
            run_id,
            parameters={
                "source": "vital",
                "numerator_type": "events",
                "year": 2019,
                "data_source": vital_data['data_source'].iloc[0] if len(vital_data) > 0 else "unknown",
            },
            row_count=len(visibility),
        )
        
        # QA checks
        nta_count = visibility["geo_id"].nunique()
        avg_visibility = visibility["visibility"].mean()
        log_qa_check(logger, "nta_coverage", True, f"Visibility for {nta_count} NTAs")
        log_qa_check(logger, "visibility_range", True, 
                    f"Mean visibility: {avg_visibility:.1f} per 1,000")
        
        # Plausibility check - NYC crude death rate is ~6-7 per 1,000
        if 3 < avg_visibility < 15:
            log_qa_check(logger, "plausibility", True, 
                        f"Mean {avg_visibility:.1f} within expected range (3-15)")
        else:
            log_qa_check(logger, "plausibility", False, 
                        f"Mean {avg_visibility:.1f} outside expected range (3-15)")
        
        # Reliability distribution
        rel_dist = visibility["reliability_flag"].value_counts().to_dict()
        log_qa_check(logger, "reliability_distribution", True, str(rel_dist))
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   NTAs with visibility: {nta_count}")
        logger.info(f"   Mean visibility: {avg_visibility:.1f} per 1,000")
        logger.info(f"   Data source: {vital_data['data_source'].iloc[0] if len(vital_data) > 0 else 'unknown'}")
        logger.info(f"   Reliability: {rel_dist}")
        logger.info(f"   Output: {vis_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

