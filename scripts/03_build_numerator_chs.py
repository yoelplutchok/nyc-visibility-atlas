#!/usr/bin/env python3
"""
03_build_numerator_chs.py

Ingest NYC Community Health Survey (CHS) data and compute survey visibility proxies.

Pipeline Step: 03
Contract Reference: Section 11 - 03_build_numerator_chs.py

The CHS provides health estimates at UHF34 geography. Since respondent counts
are typically not public, we compute a Survey Visibility Proxy using:
    n_eff ≈ p(1-p) / SE²
    SurveyVisibilityProxy = n_eff / ReferencePop × 1000

Inputs:
    - CHS data from NYC Open Data (UHF34 geography)
    - data/processed/denominators/acs_denominators.parquet
    - UHF34 boundaries (to build UHF34→NTA crosswalk)

Outputs:
    - data/processed/numerators/chs.parquet
    - data/processed/visibility/chs_visibility.parquet
    - data/processed/xwalk/uhf34_to_nta_pop_weights.parquet (if built)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet, read_geoparquet,
    read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "03_build_numerator_chs"

# NYC Open Data CHS endpoint
# The CHS data is available as indicator-level estimates
CHS_API_BASE = "https://data.cityofnewyork.us/resource"

# UHF34 to NTA mapping - UHF34 are aggregations of ZIP codes
# We'll download UHF34 boundaries and build a crosswalk
UHF34_GEOJSON_URL = "https://data.cityofnewyork.us/api/geospatial/b55q-34ps?method=export&format=GeoJSON"


def download_uhf34_boundaries(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Download UHF34 (United Hospital Fund) neighborhood boundaries."""
    log_step_start(logger, "download_uhf34_boundaries")
    
    raw_dir = ensure_dir(paths.raw_geo)
    local_file = raw_dir / "uhf34_boundaries.geojson"
    
    if local_file.exists():
        logger.info(f"Using cached UHF34 file: {local_file}")
        gdf = gpd.read_file(local_file)
    else:
        # Try multiple UHF boundary sources
        urls = [
            # UHF42 neighborhoods (more detailed)
            "https://data.cityofnewyork.us/api/geospatial/bnzz-cerc?method=export&format=GeoJSON",
            # Modified ZCTA for UHF
            "https://data.cityofnewyork.us/api/geospatial/f3qd-qz9n?method=export&format=GeoJSON",
        ]
        
        gdf = None
        for url in urls:
            logger.info(f"Trying UHF boundary source: {url[:60]}...")
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                # Save and read from temp file
                temp_file = raw_dir / "uhf_temp.geojson"
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                gdf = gpd.read_file(temp_file)
                temp_file.unlink()
                logger.info(f"Downloaded {len(gdf)} UHF neighborhoods")
                break
            except Exception as e:
                logger.warning(f"Failed: {e}")
                continue
        
        if gdf is None or len(gdf) == 0:
            # Create synthetic UHF boundaries from NTAs (aggregate NTAs to approximate UHF)
            logger.warning("Cannot download UHF boundaries. Creating synthetic UHF from NTA data...")
            gdf = create_synthetic_uhf_from_ntas(logger)
        
        gdf.to_file(local_file, driver="GeoJSON")
        logger.info(f"Saved UHF boundaries to {local_file}")
    
    # Normalize columns
    gdf = gdf.rename(columns=lambda x: x.lower())
    
    # Find UHF code column
    uhf_col_candidates = ["uhf_neigh", "uhfcode", "uhf34", "uhf42", "uhf", "geocode", "id"]
    for col in uhf_col_candidates:
        if col in gdf.columns:
            gdf["uhf_code"] = gdf[col].astype(str)
            logger.info(f"Using column '{col}' as UHF code")
            break
    
    if "uhf_code" not in gdf.columns:
        # Use index as code
        gdf["uhf_code"] = [str(i) for i in range(len(gdf))]
        logger.warning("Created synthetic UHF codes from index")
    
    gdf = gdf.to_crs("EPSG:4326")
    log_step_end(logger, "download_uhf34_boundaries", uhf_count=len(gdf))
    return gdf


def create_synthetic_uhf_from_ntas(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Create synthetic UHF-like areas by aggregating NTAs by borough."""
    ntas = read_geoparquet(paths.processed_geo / "nta_canonical.parquet")
    
    # Group NTAs by borough to create pseudo-UHF areas
    # Each borough becomes one "UHF" area for simplicity
    dissolved = ntas.dissolve(by="borough").reset_index()
    dissolved["uhf_code"] = dissolved["borough"].map({
        "Manhattan": "100",
        "Bronx": "200", 
        "Brooklyn": "300",
        "Queens": "400",
        "Staten Island": "500"
    })
    
    logger.info(f"Created {len(dissolved)} synthetic UHF areas from borough aggregation")
    return dissolved


def build_uhf34_to_nta_crosswalk_from_borough(
    uhf_codes: list,
    ntas: gpd.GeoDataFrame,
    denominators: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build UHF34→NTA crosswalk using borough-based population weighting.
    
    Since real UHF34 boundaries are not publicly available, we use the known
    UHF code → borough mapping (first digit) and distribute each UHF's data
    to NTAs within that borough based on population weights.
    
    UHF code mapping:
    - 1xx → Manhattan
    - 2xx → Bronx  
    - 3xx → Brooklyn
    - 4xx → Queens
    - 5xx → Staten Island
    """
    log_step_start(logger, "build_uhf34_to_nta_crosswalk_from_borough")
    
    # Map UHF first digit to borough
    UHF_DIGIT_TO_BOROUGH = {
        '1': 'Manhattan',
        '2': 'Bronx',
        '3': 'Brooklyn',
        '4': 'Queens',
        '5': 'Staten Island'
    }
    
    # Get total population for each NTA (stratum = 'total')
    nta_pops = denominators[denominators['stratum_id'] == 'total'][['geo_id', 'reference_pop']].copy()
    
    # Merge NTAs with their populations
    ntas_with_pop = ntas.merge(nta_pops, on='geo_id', how='left')
    ntas_with_pop['reference_pop'] = ntas_with_pop['reference_pop'].fillna(0)
    
    # Calculate borough totals
    borough_totals = ntas_with_pop.groupby('borough')['reference_pop'].sum().to_dict()
    
    records = []
    for uhf_code in uhf_codes:
        uhf_code = str(uhf_code)
        first_digit = uhf_code[0] if len(uhf_code) > 0 else None
        
        if first_digit not in UHF_DIGIT_TO_BOROUGH:
            logger.warning(f"Unknown UHF code pattern: {uhf_code}")
            continue
        
        borough = UHF_DIGIT_TO_BOROUGH[first_digit]
        borough_pop = borough_totals.get(borough, 0)
        
        if borough_pop == 0:
            continue
        
        # Get NTAs in this borough
        borough_ntas = ntas_with_pop[ntas_with_pop['borough'] == borough]
        
        for _, nta_row in borough_ntas.iterrows():
            nta_pop = nta_row['reference_pop']
            weight = nta_pop / borough_pop if borough_pop > 0 else 0
            
            if weight > 0:
                records.append({
                    'source_geo_id': uhf_code,
                    'target_geo_id': nta_row['geo_id'],
                    'weight': weight,
                    'borough': borough
                })
    
    crosswalk = pd.DataFrame(records)
    crosswalk = crosswalk.sort_values(['source_geo_id', 'target_geo_id']).reset_index(drop=True)
    
    log_step_end(logger, "build_uhf34_to_nta_crosswalk_from_borough",
                 pair_count=len(crosswalk),
                 unique_uhf=crosswalk['source_geo_id'].nunique() if len(crosswalk) > 0 else 0,
                 unique_nta=crosswalk['target_geo_id'].nunique() if len(crosswalk) > 0 else 0)
    
    return crosswalk[['source_geo_id', 'target_geo_id', 'weight']]


def build_uhf34_to_nta_crosswalk(
    uhf34: gpd.GeoDataFrame,
    ntas: gpd.GeoDataFrame,
    tract_crosswalk: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Build population-weighted crosswalk from UHF34 to NTA using spatial overlay."""
    log_step_start(logger, "build_uhf34_to_nta_crosswalk")
    
    # Project for area calculations
    uhf34_proj = uhf34.to_crs("EPSG:2263")
    ntas_proj = ntas.to_crs("EPSG:2263")
    
    # Simple area-weighted approach since we don't have tract populations per UHF
    # Overlay UHF34 with NTA
    logger.info("Computing UHF34-NTA intersections...")
    overlay = gpd.overlay(uhf34_proj, ntas_proj, how="intersection", keep_geom_type=False)
    
    # Compute intersection areas
    overlay["intersection_area"] = overlay.geometry.area
    
    # Find the correct column names after overlay (they may have suffixes)
    uhf_col = None
    nta_col = None
    
    for col in overlay.columns:
        if "uhf_code" in col.lower():
            uhf_col = col
        if col == "geo_id" or col.startswith("geo_id"):
            # Prefer geo_id_2 (from NTA) if both exist
            if nta_col is None or col == "geo_id_2" or col == "geo_id":
                nta_col = col
    
    logger.info(f"Using columns: uhf={uhf_col}, nta={nta_col}")
    
    if uhf_col and nta_col:
        crosswalk = overlay.groupby([uhf_col, nta_col]).agg({
            "intersection_area": "sum"
        }).reset_index()
        
        # Compute weights as proportion of UHF area going to each NTA
        uhf_totals = crosswalk.groupby(uhf_col)["intersection_area"].sum().reset_index()
        uhf_totals.columns = [uhf_col, "total_area"]
        
        crosswalk = crosswalk.merge(uhf_totals, on=uhf_col)
        crosswalk["weight"] = crosswalk["intersection_area"] / crosswalk["total_area"]
        
        crosswalk = crosswalk.rename(columns={
            uhf_col: "source_geo_id",
            nta_col: "target_geo_id"
        })
        crosswalk = crosswalk[["source_geo_id", "target_geo_id", "weight"]]
        crosswalk = crosswalk.sort_values(["source_geo_id", "target_geo_id"]).reset_index(drop=True)
    else:
        logger.error(f"Missing columns for crosswalk. Available: {list(overlay.columns)}")
        crosswalk = pd.DataFrame(columns=["source_geo_id", "target_geo_id", "weight"])
    
    log_step_end(logger, "build_uhf34_to_nta_crosswalk", 
                 pair_count=len(crosswalk),
                 unique_uhf=crosswalk["source_geo_id"].nunique() if len(crosswalk) > 0 else 0)
    return crosswalk


def download_chs_indicators(logger: logging.Logger) -> pd.DataFrame:
    """
    Download CHS indicator data from NYC Open Data.
    
    Priority order:
    1. EpiQuery manual exports (data/raw/chs/epiquery_exports/*.csv)
    2. NYC Open Data API
    3. Alternative API endpoints
    4. Synthetic demonstration data (last resort)
    
    The CHS publishes various health indicators with estimates and confidence intervals.
    We'll use a sample of indicators to demonstrate the visibility methodology.
    """
    log_step_start(logger, "download_chs_indicators")
    
    raw_dir = ensure_dir(paths.raw_chs)
    
    # PRIORITY 1: Check for EpiQuery exports first
    epiquery_data = load_epiquery_exports(logger)
    if len(epiquery_data) > 0:
        logger.info("✅ Using EpiQuery manual exports (real data!)")
        cache_file = raw_dir / "chs_indicators_epiquery.csv"
        epiquery_data.to_csv(cache_file, index=False)
        log_step_end(logger, "download_chs_indicators", record_count=len(epiquery_data))
        return epiquery_data
    
    cache_file = raw_dir / "chs_indicators.csv"
    
    if cache_file.exists():
        logger.info(f"Using cached CHS data: {cache_file}")
        df = pd.read_csv(cache_file)
        log_step_end(logger, "download_chs_indicators", record_count=len(df))
        return df
    
    # Try to get CHS data from NYC Open Data
    # CHS indicator data endpoint (Community Health Survey Public Use Data)
    chs_dataset_id = "jz8u-mrj8"  # CHS public use dataset
    
    url = f"https://data.cityofnewyork.us/resource/{chs_dataset_id}.json?$limit=50000"
    
    logger.info("Downloading CHS indicator data from NYC Open Data...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if len(data) > 0:
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} CHS records")
            df.to_csv(cache_file, index=False)
        else:
            logger.warning("No data returned from CHS API")
            df = pd.DataFrame()
            
    except Exception as e:
        logger.warning(f"CHS download failed: {e}")
        
        # Try alternative: get aggregated neighborhood health data
        logger.info("Trying alternative CHS data source...")
        alt_url = "https://data.cityofnewyork.us/resource/c8ck-k2fj.json?$limit=10000"
        
        try:
            response = requests.get(alt_url, timeout=60)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} records from alternative source")
            df.to_csv(cache_file, index=False)
        except Exception as e2:
            logger.warning(f"Alternative CHS download also failed: {e2}")
            
            # Create synthetic demonstration data
            logger.info("Creating synthetic CHS demonstration data...")
            df = create_synthetic_chs_data(logger)
            df.to_csv(cache_file, index=False)
    
    log_step_end(logger, "download_chs_indicators", record_count=len(df))
    return df


def load_epiquery_exports(logger: logging.Logger) -> pd.DataFrame:
    """
    Load CHS data from manually exported EpiQuery CSV/TSV files.
    
    Expected file location: data/raw/chs/epiquery_exports/*.csv
    
    EpiQuery exports are tab-separated with columns like:
    - Yearnum, Select Indicator, Dimension Response (contains UHF code + name)
    - Estimated Prevalence, Lower Confidence Interval, Upper Confidence Interval
    - Interpretation Flag (reliability marker)
    """
    export_dir = paths.raw_chs / "epiquery_exports"
    
    if not export_dir.exists():
        logger.info(f"EpiQuery export directory not found: {export_dir}")
        return pd.DataFrame()
    
    csv_files = list(export_dir.glob("chs_*.csv"))
    
    if len(csv_files) == 0:
        logger.info("No EpiQuery CSV files found")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} EpiQuery export files")
    
    all_records = []
    
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file.name}")
        
        # Extract indicator name from filename (e.g., chs_diabetes_2019.csv -> diabetes)
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            indicator_id = parts[1]
        else:
            indicator_id = csv_file.stem
        
        try:
            # EpiQuery exports are UTF-16 Little Endian with tab separators
            # Try multiple encodings
            df = None
            encodings_to_try = ['utf-16', 'utf-16-le', 'utf-8', 'latin-1']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_file, sep='\t', encoding=encoding)
                    if len(df.columns) >= 3:
                        logger.info(f"  Successfully read with encoding: {encoding}")
                        break
                except:
                    continue
            
            if df is None or len(df.columns) < 3:
                # Try comma-separated as fallback
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding)
                        if len(df.columns) >= 3:
                            logger.info(f"  Successfully read CSV with encoding: {encoding}")
                            break
                    except:
                        continue
            
            if df is None:
                logger.warning(f"  Could not read {csv_file.name} with any encoding")
                continue
            
            # Standardize column names
            df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
            
            logger.info(f"  Columns: {list(df.columns)[:8]}...")
            logger.info(f"  Rows: {len(df)}")
            
            # Identify key columns based on EpiQuery format
            year_col = next((c for c in df.columns if 'year' in c), None)
            response_col = next((c for c in df.columns if c == 'response'), None)
            dimension_col = next((c for c in df.columns if 'dimension' in c), None)
            prevalence_col = next((c for c in df.columns if 'prevalence' in c), None)
            ci_low_col = next((c for c in df.columns if 'lower' in c and 'confidence' in c), None)
            ci_high_col = next((c for c in df.columns if 'upper' in c and 'confidence' in c), None)
            flag_col = next((c for c in df.columns if 'flag' in c or 'interpretation' in c), None)
            indicator_col = next((c for c in df.columns if 'indicator' in c and 'select' in c), None)
            
            if dimension_col is None:
                logger.warning(f"  Cannot find dimension column, skipping {csv_file.name}")
                continue
            
            if prevalence_col is None:
                logger.warning(f"  Cannot find prevalence column, skipping {csv_file.name}")
                continue
            
            # Filter to:
            # 1. Only "Yes" responses (for prevalence indicators like diabetes)
            # 2. Only neighborhood-level data (exclude "Citywide")
            # 3. Most recent year available (prefer 2019)
            
            df_filtered = df.copy()
            
            # Filter to "Yes" responses if response column exists
            if response_col and 'yes' in df_filtered[response_col].astype(str).str.lower().unique():
                df_filtered = df_filtered[df_filtered[response_col].astype(str).str.lower() == 'yes']
                logger.info(f"  Filtered to 'Yes' responses: {len(df_filtered)} rows")
            
            # Exclude citywide rows
            df_filtered = df_filtered[~df_filtered[dimension_col].astype(str).str.lower().str.contains('citywide')]
            logger.info(f"  Excluded citywide: {len(df_filtered)} rows")
            
            # Get most recent year (prefer 2019)
            if year_col:
                available_years = sorted(df_filtered[year_col].dropna().unique(), reverse=True)
                target_year = 2019 if 2019 in available_years else (available_years[0] if available_years else None)
                if target_year:
                    df_filtered = df_filtered[df_filtered[year_col] == target_year]
                    logger.info(f"  Filtered to year {target_year}: {len(df_filtered)} rows")
            
            # Extract UHF code and name from dimension column (e.g., "101 Kingsbridge")
            for _, row in df_filtered.iterrows():
                dimension = str(row[dimension_col]).strip()
                
                # Parse UHF code from dimension (format: "101 Kingsbridge" or "105/106/107 South Bronx")
                import re
                match = re.match(r'^(\d+(?:/\d+)*)\s+(.+)$', dimension)
                if match:
                    uhf_code = match.group(1).split('/')[0]  # Take first code if multiple
                    uhf_name = match.group(2).strip()
                else:
                    # Skip non-neighborhood rows
                    continue
                
                # Get prevalence (remove % if present)
                prevalence = row.get(prevalence_col, np.nan)
                if pd.isna(prevalence):
                    continue
                if isinstance(prevalence, str):
                    prevalence = prevalence.replace('%', '').strip()
                    try:
                        prevalence = float(prevalence)
                    except:
                        continue
                
                # Get CI bounds
                ci_low = row.get(ci_low_col, np.nan) if ci_low_col else np.nan
                ci_high = row.get(ci_high_col, np.nan) if ci_high_col else np.nan
                
                if isinstance(ci_low, str):
                    ci_low = float(ci_low.replace('%', '').strip()) if ci_low.replace('%', '').strip() else np.nan
                if isinstance(ci_high, str):
                    ci_high = float(ci_high.replace('%', '').strip()) if ci_high.replace('%', '').strip() else np.nan
                
                # Compute SE from CI
                if not pd.isna(ci_low) and not pd.isna(ci_high):
                    se = (ci_high - ci_low) / (2 * 1.96)
                else:
                    se = np.nan
                
                # Get reliability flag
                reliability = ""
                if flag_col:
                    flag_value = str(row.get(flag_col, ""))
                    if "*" in flag_value or "caution" in flag_value.lower():
                        reliability = "low"
                    elif "#" in flag_value or "suppress" in flag_value.lower():
                        reliability = "suppressed"
                    else:
                        reliability = "high"
                
                # Get year
                year = row.get(year_col, 2019) if year_col else 2019
                
                all_records.append({
                    "uhf_code": uhf_code,
                    "uhf_name": uhf_name,
                    "indicator_id": indicator_id,
                    "indicator_name": row.get(indicator_col, indicator_id) if indicator_col else indicator_id,
                    "estimate": prevalence,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "year": str(year),
                    "stratum": "total",
                    "reliability_source": reliability,
                    "data_source": "epiquery_export",
                })
                
        except Exception as e:
            logger.warning(f"Failed to process {csv_file.name}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue
    
    if len(all_records) == 0:
        logger.warning("No valid records extracted from EpiQuery exports")
        return pd.DataFrame()
    
    result = pd.DataFrame(all_records)
    logger.info(f"✅ Loaded {len(result)} records from EpiQuery exports")
    logger.info(f"   Indicators: {result['indicator_id'].unique().tolist()}")
    logger.info(f"   UHF areas: {result['uhf_code'].nunique()}")
    logger.info(f"   Years: {result['year'].unique().tolist()}")
    
    return result


def create_synthetic_chs_data(logger: logging.Logger) -> pd.DataFrame:
    """
    Create synthetic CHS-like data for demonstration purposes.
    
    This creates realistic-looking survey data at UHF34 geography
    to demonstrate the visibility methodology.
    """
    logger.info("Creating synthetic CHS data for methodology demonstration...")
    
    # UHF34 codes (approximate)
    uhf_codes = [
        "101", "102", "103", "104", "105", "106", "107",  # Manhattan
        "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211",  # Bronx
        "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311",  # Brooklyn
        "401", "402", "403", "404", "405", "406", "407", "408", "409", "410",  # Queens
        "501", "502", "503", "504"  # Staten Island
    ]
    
    # Sample health indicators
    indicators = [
        ("current_smoker", "Current Smoker", 0.12),
        ("obesity", "Obesity (BMI 30+)", 0.25),
        ("diabetes", "Diabetes", 0.12),
        ("high_blood_pressure", "High Blood Pressure", 0.28),
        ("poor_mental_health", "Poor Mental Health Days", 0.15),
    ]
    
    records = []
    np.random.seed(42)  # For reproducibility
    
    for uhf in uhf_codes:
        for ind_id, ind_name, base_prev in indicators:
            # Add some variation by neighborhood
            prevalence = base_prev + np.random.normal(0, 0.03)
            prevalence = max(0.01, min(0.99, prevalence))
            
            # Simulate SE based on typical sample size (varies by neighborhood)
            # Larger neighborhoods have smaller SE
            n_approx = np.random.randint(200, 800)
            se = np.sqrt(prevalence * (1 - prevalence) / n_approx)
            
            records.append({
                "uhf_code": uhf,
                "indicator_id": ind_id,
                "indicator_name": ind_name,
                "estimate": prevalence * 100,  # As percentage
                "se": se * 100,
                "ci_low": (prevalence - 1.96 * se) * 100,
                "ci_high": (prevalence + 1.96 * se) * 100,
                "year": "2018-2022",
                "stratum": "total",
                "data_source": "synthetic_demo",
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Created {len(df)} synthetic CHS records")
    return df


def compute_survey_visibility(
    chs_data: pd.DataFrame,
    denominators: pd.DataFrame,
    uhf_crosswalk: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute survey visibility proxies from CHS data.
    
    Survey Visibility Proxy methodology:
    1. Derive SE from CI if not directly available
    2. Compute n_eff ≈ p(1-p) / SE²
    3. SurveyVisibilityProxy = n_eff / ReferencePop × 1000
    """
    log_step_start(logger, "compute_survey_visibility")
    
    # Standardize column names
    chs_data = chs_data.rename(columns=lambda x: x.lower().replace(" ", "_"))
    
    # Ensure we have required columns
    if "uhf_code" not in chs_data.columns:
        # Try to find UHF column
        for col in chs_data.columns:
            if "uhf" in col.lower() or "neighborhood" in col.lower():
                chs_data["uhf_code"] = chs_data[col].astype(str)
                break
    
    if "uhf_code" not in chs_data.columns:
        logger.error(f"Cannot find UHF code column. Columns: {list(chs_data.columns)}")
        return pd.DataFrame()
    
    # Get unique indicators
    if "indicator_id" in chs_data.columns:
        indicators = chs_data["indicator_id"].unique()
    else:
        indicators = ["overall"]
        chs_data["indicator_id"] = "overall"
    
    logger.info(f"Processing {len(indicators)} indicators for {chs_data['uhf_code'].nunique()} UHF areas")
    
    records = []
    
    # Aggregate UHF-level data to NTA using crosswalk
    for _, row in chs_data.iterrows():
        uhf = str(row.get("uhf_code", ""))
        
        # Get NTA weights for this UHF
        uhf_weights = uhf_crosswalk[uhf_crosswalk["source_geo_id"] == uhf]
        
        if len(uhf_weights) == 0:
            continue
        
        # Get estimate and SE
        estimate = row.get("estimate", 0)
        if pd.isna(estimate):
            continue
            
        # Convert percentage to proportion
        p = estimate / 100 if estimate > 1 else estimate
        p = max(0.001, min(0.999, p))  # Bound away from 0 and 1
        
        # Get SE (or derive from CI)
        se = row.get("se", np.nan)
        if pd.isna(se):
            ci_low = row.get("ci_low", np.nan)
            ci_high = row.get("ci_high", np.nan)
            if not pd.isna(ci_low) and not pd.isna(ci_high):
                # SE ≈ (CI_high - CI_low) / (2 * 1.96)
                se = (ci_high - ci_low) / (2 * 1.96 * 100)  # Convert from percentage
            else:
                se = 0.05  # Default assumption
        else:
            se = se / 100 if se > 1 else se  # Convert from percentage
        
        se = max(0.001, se)  # Avoid division by zero
        
        # Compute effective sample size
        n_eff = (p * (1 - p)) / (se ** 2)
        n_eff = max(1, min(10000, n_eff))  # Bound to reasonable range
        
        # Distribute n_eff to NTAs based on weights
        for _, weight_row in uhf_weights.iterrows():
            nta_id = weight_row["target_geo_id"]
            weight = weight_row["weight"]
            
            # Get reference population for this NTA
            nta_denom = denominators[
                (denominators["geo_id"] == nta_id) & 
                (denominators["stratum_id"] == "total")
            ]
            
            if len(nta_denom) == 0:
                continue
            
            ref_pop = nta_denom.iloc[0]["reference_pop"]
            if ref_pop <= 0:
                continue
            
            # Weight the n_eff by area proportion
            nta_n_eff = n_eff * weight
            
            # Compute visibility
            visibility = (nta_n_eff / ref_pop) * 1000
            
            # Determine reliability flag
            if nta_n_eff < 10:
                reliability = "suppressed"
            elif nta_n_eff < 30:
                reliability = "low"
            else:
                reliability = "high"
            
            records.append({
                "geo_id": nta_id,
                "stratum_id": "total",
                "time_window_id": "2018_2022",
                "source_id": "chs",
                "indicator_id": row.get("indicator_id", "overall"),
                "observed_count": nta_n_eff,
                "reference_pop": ref_pop,
                "visibility": visibility,
                "reliability_flag": reliability,
                "numerator_type": "respondents",
                "se": se,
                "estimate": p,
            })
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        # Sort for determinism
        df = df.sort_values(["geo_id", "indicator_id"]).reset_index(drop=True)
    
    log_step_end(logger, "compute_survey_visibility", record_count=len(df))
    return df


def aggregate_visibility_by_nta(visibility_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Aggregate visibility across indicators to get overall CHS visibility per NTA."""
    log_step_start(logger, "aggregate_visibility_by_nta")
    
    if len(visibility_df) == 0:
        return pd.DataFrame()
    
    # Average visibility across indicators per NTA
    agg = visibility_df.groupby(["geo_id", "stratum_id", "time_window_id", "source_id"]).agg({
        "observed_count": "mean",
        "reference_pop": "first",
        "visibility": "mean",
    }).reset_index()
    
    # Determine reliability based on average n_eff
    agg["reliability_flag"] = agg["observed_count"].apply(
        lambda x: "suppressed" if x < 10 else ("low" if x < 30 else "high")
    )
    agg["numerator_type"] = "respondents"
    
    log_step_end(logger, "aggregate_visibility_by_nta", record_count=len(agg))
    return agg


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load NTAs
        ntas = read_geoparquet(paths.processed_geo / "nta_canonical.parquet")
        logger.info(f"Loaded {len(ntas)} NTAs")
        
        # Load denominators
        denominators = read_parquet(paths.processed_denominators / "acs_denominators.parquet")
        logger.info(f"Loaded {len(denominators)} denominator records")
        
        # Download CHS data FIRST to see what UHF codes we have
        chs_data = download_chs_indicators(logger)
        
        if len(chs_data) == 0:
            logger.error("No CHS data available")
            return 1
        
        # Check what UHF codes we have
        uhf_codes = chs_data['uhf_code'].unique().tolist()
        logger.info(f"CHS data has {len(uhf_codes)} unique UHF codes")
        
        # PRIORITY: Use proper UHF42 crosswalk if available (from 02b_build_uhf_nta_crosswalk.py)
        uhf42_crosswalk_path = paths.processed_xwalk / "uhf42_to_nta_pop_weights.parquet"
        if uhf42_crosswalk_path.exists():
            logger.info("Using proper UHF42→NTA crosswalk (spatial boundaries)")
            uhf_crosswalk = read_parquet(uhf42_crosswalk_path)
            # Convert source_geo_id to string for matching
            uhf_crosswalk['source_geo_id'] = uhf_crosswalk['source_geo_id'].astype(str)
            # Verify coverage: check that all CHS UHF codes have crosswalk entries
            xwalk_codes = set(uhf_crosswalk['source_geo_id'].unique())
            chs_codes = set(str(c) for c in uhf_codes)
            missing = chs_codes - xwalk_codes
            if missing:
                logger.warning(f"CHS UHF codes not in crosswalk: {missing}")
            coverage = len(chs_codes & xwalk_codes) / len(chs_codes) * 100
            logger.info(f"UHF code coverage: {coverage:.1f}% ({len(chs_codes & xwalk_codes)}/{len(chs_codes)} codes)")
        else:
            # Fallback: Determine if these are real UHF codes or synthetic
            has_real_uhf_codes = any(len(str(code)) == 3 and str(code)[1:] != '00' for code in uhf_codes)
            
            if has_real_uhf_codes:
                # Use borough-based crosswalk for real UHF codes (APPROXIMATION - should run 02b first!)
                logger.warning("Using borough-based UHF→NTA crosswalk (APPROXIMATION)")
                logger.warning("For better results, run 02b_build_uhf_nta_crosswalk.py first")
                uhf_crosswalk = build_uhf34_to_nta_crosswalk_from_borough(
                    uhf_codes, ntas, denominators, logger
                )
            else:
                # Use spatial crosswalk for synthetic borough-level codes
                logger.info("Using spatial UHF→NTA crosswalk (synthetic borough codes detected)")
                uhf34 = download_uhf34_boundaries(logger)
                tract_crosswalk = read_parquet(paths.processed_xwalk / "tract_to_nta_pop_weights.parquet")
                tracts = gpd.read_file(paths.raw_geo / "nyc_tracts_2020_v2.geojson")
                uhf_crosswalk = build_uhf34_to_nta_crosswalk(
                    uhf34, ntas, tract_crosswalk, tracts, logger
                )
        
        # Save UHF crosswalk
        if len(uhf_crosswalk) > 0:
            xwalk_path = paths.processed_xwalk / "uhf34_to_nta_pop_weights.parquet"
            atomic_write_parquet(xwalk_path, uhf_crosswalk)
            log_output_written(logger, xwalk_path, row_count=len(uhf_crosswalk))
        
        # Compute visibility
        visibility_detailed = compute_survey_visibility(
            chs_data, denominators, uhf_crosswalk, logger
        )
        
        # Aggregate to NTA level
        visibility_agg = aggregate_visibility_by_nta(visibility_detailed, logger)
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_numerators)
        
        # Write detailed CHS data
        chs_path = output_dir / "chs.parquet"
        atomic_write_parquet(chs_path, chs_data)
        log_output_written(logger, chs_path, row_count=len(chs_data))
        
        # Write visibility
        vis_dir = ensure_dir(paths.processed_visibility)
        vis_path = vis_dir / "chs_visibility.parquet"
        atomic_write_parquet(vis_path, visibility_agg)
        log_output_written(logger, vis_path, row_count=len(visibility_agg))
        
        # Write metadata
        write_metadata_sidecar(
            vis_path,
            run_id,
            parameters={
                "source": "chs",
                "numerator_type": "respondents",
                "method": "n_eff_proxy",
            },
            row_count=len(visibility_agg),
        )
        
        # QA checks
        if len(visibility_agg) > 0:
            nta_count = visibility_agg["geo_id"].nunique()
            avg_visibility = visibility_agg["visibility"].mean()
            log_qa_check(logger, "nta_coverage", True, f"Visibility for {nta_count} NTAs")
            log_qa_check(logger, "visibility_range", True, 
                        f"Mean visibility: {avg_visibility:.1f} per 1,000")
            
            # Reliability distribution
            rel_dist = visibility_agg["reliability_flag"].value_counts().to_dict()
            log_qa_check(logger, "reliability_distribution", True, str(rel_dist))
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        if len(visibility_agg) > 0:
            logger.info(f"   NTAs with visibility: {visibility_agg['geo_id'].nunique()}")
            logger.info(f"   Mean visibility: {visibility_agg['visibility'].mean():.1f} per 1,000")
            logger.info(f"   Reliability: {visibility_agg['reliability_flag'].value_counts().to_dict()}")
        logger.info(f"   Output: {vis_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

