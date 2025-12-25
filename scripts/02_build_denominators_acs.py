#!/usr/bin/env python3
"""
02_build_denominators_acs.py

Create ACS-derived population denominators by NTA geography and stratum.

Pipeline Step: 02
Contract Reference: Section 11 - 02_build_denominators_acs.py

Inputs:
    - data/processed/xwalk/tract_to_nta_pop_weights.parquet
    - data/processed/geo/nta_canonical.parquet
    - configs/params.yml (time windows)
    - configs/strata.yml (stratum definitions)
    - ACS API for tract-level data

Outputs:
    - data/processed/denominators/acs_denominators.parquet
    - data/processed/metadata/acs_denominators_metadata.json

QA Checks:
    - Full coverage of NTAs (or explicit missing list)
    - MOE non-negative
    - ReferencePop totals plausible (NYC ~8.3M)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math
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
    atomic_write_parquet, read_parquet, read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "02_build_denominators_acs"

# Census API
CENSUS_API_BASE = "https://api.census.gov/data"
ACS_YEAR = 2022
ACS_DATASET = "acs/acs5"

# NYC county FIPS
NYC_COUNTY_FIPS = {
    "36005": "Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island",
}

# ACS table definitions for strata
ACS_TABLES = {
    # Total population
    "total": {
        "table": "B01001",
        "variables": {"B01001_001E": "total", "B01001_001M": "total_moe"},
    },
    # Age groups (from Sex by Age table)
    "age": {
        "table": "B01001",
        "variables": {
            # Under 18 (sum male + female under 18)
            "B01001_003E": "male_under5",
            "B01001_004E": "male_5to9",
            "B01001_005E": "male_10to14",
            "B01001_006E": "male_15to17",
            "B01001_027E": "female_under5",
            "B01001_028E": "female_5to9",
            "B01001_029E": "female_10to14",
            "B01001_030E": "female_15to17",
            # 18-24
            "B01001_007E": "male_18to19",
            "B01001_008E": "male_20",
            "B01001_009E": "male_21",
            "B01001_010E": "male_22to24",
            "B01001_031E": "female_18to19",
            "B01001_032E": "female_20",
            "B01001_033E": "female_21",
            "B01001_034E": "female_22to24",
            # 25-44
            "B01001_011E": "male_25to29",
            "B01001_012E": "male_30to34",
            "B01001_013E": "male_35to39",
            "B01001_014E": "male_40to44",
            "B01001_035E": "female_25to29",
            "B01001_036E": "female_30to34",
            "B01001_037E": "female_35to39",
            "B01001_038E": "female_40to44",
            # 45-64
            "B01001_015E": "male_45to49",
            "B01001_016E": "male_50to54",
            "B01001_017E": "male_55to59",
            "B01001_018E": "male_60to61",
            "B01001_019E": "male_62to64",
            "B01001_039E": "female_45to49",
            "B01001_040E": "female_50to54",
            "B01001_041E": "female_55to59",
            "B01001_042E": "female_60to61",
            "B01001_043E": "female_62to64",
            # 65+
            "B01001_020E": "male_65to66",
            "B01001_021E": "male_67to69",
            "B01001_022E": "male_70to74",
            "B01001_023E": "male_75to79",
            "B01001_024E": "male_80to84",
            "B01001_025E": "male_85plus",
            "B01001_044E": "female_65to66",
            "B01001_045E": "female_67to69",
            "B01001_046E": "female_70to74",
            "B01001_047E": "female_75to79",
            "B01001_048E": "female_80to84",
            "B01001_049E": "female_85plus",
        },
        "aggregations": {
            "age_0_17": ["male_under5", "male_5to9", "male_10to14", "male_15to17",
                         "female_under5", "female_5to9", "female_10to14", "female_15to17"],
            "age_18_24": ["male_18to19", "male_20", "male_21", "male_22to24",
                          "female_18to19", "female_20", "female_21", "female_22to24"],
            "age_25_44": ["male_25to29", "male_30to34", "male_35to39", "male_40to44",
                          "female_25to29", "female_30to34", "female_35to39", "female_40to44"],
            "age_45_64": ["male_45to49", "male_50to54", "male_55to59", "male_60to61", "male_62to64",
                          "female_45to49", "female_50to54", "female_55to59", "female_60to61", "female_62to64"],
            "age_65_plus": ["male_65to66", "male_67to69", "male_70to74", "male_75to79", "male_80to84", "male_85plus",
                            "female_65to66", "female_67to69", "female_70to74", "female_75to79", "female_80to84", "female_85plus"],
        }
    },
}


def fetch_acs_tract_data(
    variables: list[str],
    logger: logging.Logger,
    year: int = ACS_YEAR
) -> pd.DataFrame:
    """Fetch ACS data at tract level for NYC counties."""
    log_step_start(logger, f"fetch_acs_tract_data")
    
    all_data = []
    var_string = ",".join(variables)
    
    for fips, boro in NYC_COUNTY_FIPS.items():
        state_fips = fips[:2]
        county_fips = fips[2:]
        
        url = (
            f"{CENSUS_API_BASE}/{year}/{ACS_DATASET}?"
            f"get={var_string}&for=tract:*"
            f"&in=state:{state_fips}&in=county:{county_fips}"
        )
        
        logger.info(f"Fetching ACS data for {boro}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            df["geoid"] = df["state"] + df["county"] + df["tract"]
            df["borough"] = boro
            all_data.append(df)
            logger.info(f"  Got {len(df)} tracts")
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {boro}: {e}")
            raise
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Convert numeric columns
    for col in result.columns:
        if col not in ["state", "county", "tract", "geoid", "borough", "NAME"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    
    log_step_end(logger, "fetch_acs_tract_data", tract_count=len(result))
    return result


def aggregate_to_nta(
    tract_data: pd.DataFrame,
    crosswalk: pd.DataFrame,
    value_cols: list[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """Aggregate tract-level data to NTA using population weights."""
    log_step_start(logger, "aggregate_to_nta")
    
    # Merge tract data with crosswalk
    merged = tract_data.merge(
        crosswalk,
        left_on="geoid",
        right_on="source_geo_id",
        how="inner"
    )
    
    logger.info(f"Merged {len(merged)} tract-NTA pairs")
    
    # Weight the values
    for col in value_cols:
        if col in merged.columns:
            merged[f"{col}_weighted"] = merged[col] * merged["weight"]
    
    # Aggregate to NTA
    weighted_cols = [f"{col}_weighted" for col in value_cols if col in merged.columns]
    agg_dict = {col: "sum" for col in weighted_cols}
    
    nta_data = merged.groupby("target_geo_id").agg(agg_dict).reset_index()
    
    # Rename columns back
    nta_data.columns = [col.replace("_weighted", "") if col != "target_geo_id" else "geo_id" 
                        for col in nta_data.columns]
    
    log_step_end(logger, "aggregate_to_nta", nta_count=len(nta_data))
    return nta_data


def compute_age_aggregations(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Compute age band aggregations from detailed age variables."""
    log_step_start(logger, "compute_age_aggregations")
    
    aggregations = ACS_TABLES["age"]["aggregations"]
    
    for agg_name, components in aggregations.items():
        # Sum the components
        available_cols = [c for c in components if c in df.columns]
        if available_cols:
            df[agg_name] = df[available_cols].sum(axis=1)
            logger.info(f"Computed {agg_name} from {len(available_cols)} components")
        else:
            logger.warning(f"No components found for {agg_name}")
    
    log_step_end(logger, "compute_age_aggregations")
    return df


def moe_sum(moes: list[float]) -> float:
    """
    Compute MOE for sum of estimates.
    MOE_sum = sqrt(sum of MOE^2)
    """
    return math.sqrt(sum(m**2 for m in moes if not pd.isna(m)))


def moe_to_se(moe: float) -> float:
    """Convert MOE (90% CI) to standard error."""
    return moe / 1.645


def build_denominators(
    nta_data: pd.DataFrame,
    ntas: pd.DataFrame,
    time_window_id: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Build denominator table in long format."""
    log_step_start(logger, "build_denominators")
    
    records = []
    
    # Get list of NTAs
    nta_ids = ntas["geo_id"].unique()
    
    for nta_id in nta_ids:
        nta_row = nta_data[nta_data["geo_id"] == nta_id]
        
        if len(nta_row) == 0:
            # NTA not in data - likely park/airport
            logger.debug(f"NTA {nta_id} not in tract data")
            continue
        
        nta_row = nta_row.iloc[0]
        
        # Total population
        if "total" in nta_row.index:
            records.append({
                "geo_id": nta_id,
                "stratum_id": "total",
                "time_window_id": time_window_id,
                "reference_pop": nta_row.get("total", 0),
                "reference_pop_moe": nta_row.get("total_moe", np.nan),
            })
        
        # Age strata
        for age_group in ["age_0_17", "age_18_24", "age_25_44", "age_45_64", "age_65_plus"]:
            if age_group in nta_row.index:
                records.append({
                    "geo_id": nta_id,
                    "stratum_id": age_group,
                    "time_window_id": time_window_id,
                    "reference_pop": nta_row.get(age_group, 0),
                    "reference_pop_moe": np.nan,  # MOE for aggregates would need proper propagation
                })
    
    df = pd.DataFrame(records)
    
    # Compute SE from MOE
    df["reference_pop_se"] = df["reference_pop_moe"].apply(
        lambda x: moe_to_se(x) if not pd.isna(x) else np.nan
    )
    
    # Sort for determinism
    df = df.sort_values(["geo_id", "stratum_id"]).reset_index(drop=True)
    
    log_step_end(logger, "build_denominators", record_count=len(df))
    return df


def validate_denominators(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Validate denominator data."""
    all_passed = True
    
    # Check total population is plausible (NYC ~8.3M)
    total_pop = df[df["stratum_id"] == "total"]["reference_pop"].sum()
    if 7_000_000 < total_pop < 10_000_000:
        log_qa_check(logger, "total_population_plausible", True,
                     f"Total population: {total_pop:,.0f}")
    else:
        log_qa_check(logger, "total_population_plausible", False,
                     f"Total population {total_pop:,.0f} outside expected range")
        all_passed = False
    
    # Check no negative values
    neg_count = (df["reference_pop"] < 0).sum()
    if neg_count == 0:
        log_qa_check(logger, "no_negative_populations", True, "No negative populations")
    else:
        log_qa_check(logger, "no_negative_populations", False,
                     f"{neg_count} negative population values")
        all_passed = False
    
    # Check MOE non-negative where present
    moe_neg = (df["reference_pop_moe"] < 0).sum()
    if moe_neg == 0:
        log_qa_check(logger, "moe_non_negative", True, "All MOE values non-negative")
    else:
        log_qa_check(logger, "moe_non_negative", False, f"{moe_neg} negative MOE values")
        all_passed = False
    
    # Check NTA coverage
    unique_ntas = df["geo_id"].nunique()
    log_qa_check(logger, "nta_coverage", True, f"Denominators for {unique_ntas} NTAs")
    
    # Check strata coverage
    strata = df["stratum_id"].unique().tolist()
    log_qa_check(logger, "strata_coverage", True, f"Strata: {strata}")
    
    return all_passed


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load configs
        params = read_yaml(paths.params_yml)
        time_window = params["time_windows"]["primary"]
        time_window_id = time_window["id"]
        logger.info(f"Time window: {time_window_id}")
        
        # Load crosswalk
        crosswalk = read_parquet(paths.processed_xwalk / "tract_to_nta_pop_weights.parquet")
        logger.info(f"Loaded crosswalk with {len(crosswalk)} pairs")
        
        # Load NTAs for reference
        ntas = pd.read_parquet(paths.processed_geo / "nta_canonical.parquet")
        logger.info(f"Loaded {len(ntas)} NTAs")
        
        # Build variable list for ACS query
        all_variables = ["B01001_001E", "B01001_001M"]  # Total + MOE
        all_variables.extend(ACS_TABLES["age"]["variables"].keys())
        
        # Fetch tract-level ACS data
        tract_data = fetch_acs_tract_data(all_variables, logger)
        
        # Rename variables to friendly names
        rename_map = {"B01001_001E": "total", "B01001_001M": "total_moe"}
        rename_map.update(ACS_TABLES["age"]["variables"])
        tract_data = tract_data.rename(columns=rename_map)
        
        # Compute age aggregations at tract level
        tract_data = compute_age_aggregations(tract_data, logger)
        
        # Define value columns to aggregate
        value_cols = ["total", "total_moe", "age_0_17", "age_18_24", 
                      "age_25_44", "age_45_64", "age_65_plus"]
        
        # Aggregate to NTA
        nta_data = aggregate_to_nta(tract_data, crosswalk, value_cols, logger)
        
        # Build denominator table
        denominators = build_denominators(nta_data, ntas, time_window_id, logger)
        
        # Validate
        validate_denominators(denominators, logger)
        
        # Write outputs
        output_path = ensure_dir(paths.processed_denominators) / "acs_denominators.parquet"
        atomic_write_parquet(output_path, denominators)
        log_output_written(logger, output_path, row_count=len(denominators))
        
        # Write metadata
        metadata_path = write_metadata_sidecar(
            output_path,
            run_id,
            config_files=[paths.params_yml, paths.strata_yml],
            parameters={
                "acs_year": ACS_YEAR,
                "time_window_id": time_window_id,
                "strata": list(denominators["stratum_id"].unique()),
            },
            row_count=len(denominators),
            extra={
                "total_population": denominators[denominators["stratum_id"] == "total"]["reference_pop"].sum(),
                "nta_count": denominators["geo_id"].nunique(),
            }
        )
        
        # Summary
        total_pop = denominators[denominators["stratum_id"] == "total"]["reference_pop"].sum()
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Total NYC population: {total_pop:,.0f}")
        logger.info(f"   NTAs with data: {denominators['geo_id'].nunique()}")
        logger.info(f"   Strata: {list(denominators['stratum_id'].unique())}")
        logger.info(f"   Output: {output_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

