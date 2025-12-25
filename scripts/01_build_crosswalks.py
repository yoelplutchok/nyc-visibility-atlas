#!/usr/bin/env python3
"""
01_build_crosswalks.py

Build population-weighted crosswalks between geographies.

Pipeline Step: 01
Contract Reference: Section 11 - 01_build_crosswalks.py

Crosswalks needed:
    - Census Tract → NTA (base crosswalk)
    - ZIP/ZCTA → NTA (for SPARCS, enrollment data)
    - UHF34 → NTA (for CHS data)

Method: Population-weighted areal interpolation using census tract populations.

Inputs:
    - data/processed/geo/nta_canonical.parquet
    - Census tract boundaries (downloaded)
    - ZCTA boundaries (downloaded)
    - UHF34 boundaries (downloaded or created)
    - ACS tract-level population

Outputs:
    - data/processed/xwalk/tract_to_nta_pop_weights.parquet
    - data/processed/xwalk/zcta_to_nta_pop_weights.parquet
    - data/processed/xwalk/uhf34_to_nta_pop_weights.parquet
    - data/processed/geo/geography_audit_log.parquet

QA Checks:
    - Weights sum to 1 ± ε within each source geography
    - Stable sorting
    - Report unmatched share
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
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
    atomic_write_parquet, atomic_write_geoparquet,
    read_geoparquet, read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar
from visibility_atlas.qa import check_crs, check_bounds_nyc, EXPECTED_CRS_WGS84


SCRIPT_NAME = "01_build_crosswalks"

# NYC county FIPS codes (for filtering tracts/ZCTAs)
NYC_COUNTY_FIPS = {
    "36005": "Bronx",
    "36047": "Brooklyn",  # Kings County
    "36061": "Manhattan", # New York County
    "36081": "Queens",
    "36085": "Staten Island", # Richmond County
}

# Census API endpoint
CENSUS_API_BASE = "https://api.census.gov/data"
ACS_YEAR = 2022
ACS_DATASET = "acs/acs5"


def download_census_tracts(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Download NYC census tract boundaries from Census TIGER."""
    log_step_start(logger, "download_census_tracts")
    
    raw_dir = ensure_dir(paths.raw_geo)
    local_file = raw_dir / "nyc_tracts_2020_v2.geojson"
    
    if local_file.exists():
        logger.info(f"Using cached tract file: {local_file}")
        gdf = gpd.read_file(local_file)
    else:
        # Use Census cartographic boundary file for tracts (more reliable)
        # Download NY state tracts and filter to NYC counties
        url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_36_tract_500k.zip"
        
        logger.info("Downloading tract boundaries from Census...")
        try:
            gdf = gpd.read_file(url)
            logger.info(f"Downloaded {len(gdf)} tracts for NY state")
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            # Try TIGERweb but with correct tract layer (layer 6, not 8)
            all_tracts = []
            
            for fips, boro in NYC_COUNTY_FIPS.items():
                state_fips = fips[:2]
                county_fips = fips[2:]
                
                # Layer 6 is Census Tracts in Tracts_Blocks service
                url = (
                    f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                    f"Tracts_Blocks/MapServer/6/query?"
                    f"where=STATE='{state_fips}'+AND+COUNTY='{county_fips}'"
                    f"&outFields=*&outSR=4326&f=geojson"
                )
                
                logger.info(f"Downloading tracts for {boro}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                gdf_county = gpd.read_file(response.text)
                gdf_county["borough"] = boro
                all_tracts.append(gdf_county)
                logger.info(f"  Downloaded {len(gdf_county)} tracts")
            
            gdf = pd.concat(all_tracts, ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        
        # Filter to NYC counties
        gdf = gdf.rename(columns=lambda x: x.upper())
        nyc_fips_list = list(NYC_COUNTY_FIPS.keys())
        
        if "GEOID" in gdf.columns:
            gdf["county_fips"] = gdf["GEOID"].str[:5]
            gdf = gdf[gdf["county_fips"].isin(nyc_fips_list)]
        elif "STATEFP" in gdf.columns and "COUNTYFP" in gdf.columns:
            gdf["county_fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]
            gdf = gdf[gdf["county_fips"].isin(nyc_fips_list)]
        
        # Add borough names
        gdf["borough"] = gdf["county_fips"].map(NYC_COUNTY_FIPS)
        
        logger.info(f"Filtered to {len(gdf)} NYC tracts")
        
        # Save for caching
        gdf.to_file(local_file, driver="GeoJSON")
        logger.info(f"Saved tracts to {local_file}")
    
    # Normalize column names to lowercase
    gdf = gdf.rename(columns=lambda x: x.lower())
    
    # Ensure we have a proper tract-level GEOID (11 digits: state + county + tract)
    if "geoid" in gdf.columns:
        # Ensure it's 11 characters (tract level, not block group)
        gdf["geoid"] = gdf["geoid"].astype(str).str[:11]
    elif "tractce" in gdf.columns and "statefp" in gdf.columns and "countyfp" in gdf.columns:
        gdf["geoid"] = gdf["statefp"] + gdf["countyfp"] + gdf["tractce"]
    
    # Remove duplicates (in case block groups were aggregated)
    gdf = gdf.drop_duplicates(subset=["geoid"])
    
    log_step_end(logger, "download_census_tracts", tract_count=len(gdf))
    return gdf


def download_zctas(logger: logging.Logger, ntas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Download ZCTA boundaries and filter to NYC area."""
    log_step_start(logger, "download_zctas")
    
    raw_dir = ensure_dir(paths.raw_geo)
    local_file = raw_dir / "nyc_zctas_2020.geojson"
    
    if local_file.exists():
        logger.info(f"Using cached ZCTA file: {local_file}")
        gdf = gpd.read_file(local_file)
    else:
        # Use Census Bureau's cartographic boundary file (more reliable)
        # This is the national file, we'll filter to NYC after
        url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
        
        logger.info("Downloading ZCTA boundaries from Census (this may take a moment)...")
        try:
            gdf = gpd.read_file(url)
            logger.info(f"Downloaded {len(gdf)} ZCTAs nationally")
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            
            # Try NYC Open Data as alternative source
            logger.info("Trying NYC Open Data for ZIP code boundaries...")
            nyc_zip_url = "https://data.cityofnewyork.us/api/geospatial/i8iw-xf4u?method=export&format=GeoJSON"
            
            try:
                gdf = gpd.read_file(nyc_zip_url)
                logger.info(f"Downloaded {len(gdf)} ZIP areas from NYC Open Data")
            except Exception as e2:
                logger.error(f"All ZCTA download methods failed: {e2}")
                logger.error("Please manually download ZCTA boundaries and save to:")
                logger.error(f"  {local_file}")
                raise FileNotFoundError(f"Cannot download ZCTA boundaries")
        
        # Filter to NYC area using NTA bounds
        gdf = gdf.to_crs("EPSG:4326")
        nyc_union = ntas.to_crs("EPSG:4326").union_all().buffer(0.01)
        gdf = gdf[gdf.intersects(nyc_union)]
        
        logger.info(f"Filtered to {len(gdf)} ZCTAs in NYC area")
        
        gdf.to_file(local_file, driver="GeoJSON")
        logger.info(f"Saved ZCTAs to {local_file}")
    
    # Normalize column names
    gdf = gdf.rename(columns=lambda x: x.lower())
    
    # Find the ZCTA/ZIP code column
    if "zcta5ce20" in gdf.columns:
        gdf["zcta"] = gdf["zcta5ce20"]
    elif "geoid20" in gdf.columns:
        gdf["zcta"] = gdf["geoid20"]
    elif "zipcode" in gdf.columns:
        gdf["zcta"] = gdf["zipcode"]
    elif "modzcta" in gdf.columns:
        gdf["zcta"] = gdf["modzcta"]
    else:
        # Look for any column with zip-like values
        for col in gdf.columns:
            if gdf[col].dtype == "object" and gdf[col].str.match(r"^\d{5}$").any():
                gdf["zcta"] = gdf[col]
                logger.info(f"Using column '{col}' as ZCTA identifier")
                break
    
    log_step_end(logger, "download_zctas", zcta_count=len(gdf))
    return gdf


def get_tract_populations(tracts: gpd.GeoDataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Get population by census tract from ACS API."""
    log_step_start(logger, "get_tract_populations")
    
    raw_dir = ensure_dir(paths.raw_acs)
    cache_file = raw_dir / "tract_populations_2022.csv"
    
    if cache_file.exists():
        logger.info(f"Using cached tract populations: {cache_file}")
        pop_df = pd.read_csv(cache_file, dtype={"geoid": str})
        log_step_end(logger, "get_tract_populations", tract_count=len(pop_df))
        return pop_df
    
    # Query ACS API for total population by tract
    all_pops = []
    
    for fips, boro in NYC_COUNTY_FIPS.items():
        state_fips = fips[:2]
        county_fips = fips[2:]
        
        url = (
            f"{CENSUS_API_BASE}/{ACS_YEAR}/{ACS_DATASET}?"
            f"get=B01001_001E,NAME&for=tract:*"
            f"&in=state:{state_fips}&in=county:{county_fips}"
        )
        
        logger.info(f"Fetching populations for {boro}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # First row is headers
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            df["geoid"] = df["state"] + df["county"] + df["tract"]
            df["population"] = pd.to_numeric(df["B01001_001E"], errors="coerce")
            df["borough"] = boro
            
            all_pops.append(df[["geoid", "population", "borough"]])
            logger.info(f"  Got {len(df)} tracts")
            
        except Exception as e:
            logger.error(f"Failed to get populations for {boro}: {e}")
            raise
    
    pop_df = pd.concat(all_pops, ignore_index=True)
    pop_df.to_csv(cache_file, index=False)
    logger.info(f"Cached populations to {cache_file}")
    
    log_step_end(logger, "get_tract_populations", tract_count=len(pop_df))
    return pop_df


def build_tract_to_nta_crosswalk(
    tracts: gpd.GeoDataFrame,
    ntas: gpd.GeoDataFrame,
    tract_pops: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build population-weighted crosswalk from census tracts to NTAs.
    
    Method: For each tract, compute intersection area with each NTA,
    then weight by tract population.
    """
    log_step_start(logger, "build_tract_to_nta_crosswalk")
    
    # Ensure consistent CRS (use projected for area calculations)
    tracts_proj = tracts.to_crs("EPSG:2263")
    ntas_proj = ntas.to_crs("EPSG:2263")
    
    # Merge population into tracts
    tracts_proj = tracts_proj.merge(
        tract_pops[["geoid", "population"]], 
        on="geoid", 
        how="left"
    )
    tracts_proj["population"] = tracts_proj["population"].fillna(0)
    
    # Compute tract areas
    tracts_proj["tract_area"] = tracts_proj.geometry.area
    
    # Spatial overlay to get intersections
    logger.info("Computing tract-NTA intersections...")
    overlay = gpd.overlay(tracts_proj, ntas_proj, how="intersection", keep_geom_type=False)
    
    # Compute intersection areas
    overlay["intersection_area"] = overlay.geometry.area
    
    # Weight = (intersection_area / tract_area) * population
    overlay["area_fraction"] = overlay["intersection_area"] / overlay["tract_area"]
    overlay["weighted_pop"] = overlay["area_fraction"] * overlay["population"]
    
    # Aggregate by tract-NTA pair
    crosswalk = overlay.groupby(["geoid", "geo_id"]).agg({
        "weighted_pop": "sum",
        "area_fraction": "sum"
    }).reset_index()
    
    # Compute weights: proportion of tract population going to each NTA
    tract_totals = crosswalk.groupby("geoid")["weighted_pop"].sum().reset_index()
    tract_totals.columns = ["geoid", "total_weighted_pop"]
    
    crosswalk = crosswalk.merge(tract_totals, on="geoid")
    crosswalk["weight"] = crosswalk["weighted_pop"] / crosswalk["total_weighted_pop"]
    crosswalk["weight"] = crosswalk["weight"].fillna(0)
    
    # Clean up
    crosswalk = crosswalk.rename(columns={
        "geoid": "source_geo_id",
        "geo_id": "target_geo_id"
    })
    crosswalk = crosswalk[["source_geo_id", "target_geo_id", "weight"]]
    
    # Sort for determinism
    crosswalk = crosswalk.sort_values(["source_geo_id", "target_geo_id"]).reset_index(drop=True)
    
    log_step_end(logger, "build_tract_to_nta_crosswalk", 
                 pair_count=len(crosswalk),
                 unique_tracts=crosswalk["source_geo_id"].nunique())
    return crosswalk


def build_zcta_to_nta_crosswalk(
    zctas: gpd.GeoDataFrame,
    ntas: gpd.GeoDataFrame,
    tract_crosswalk: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    tract_pops: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build population-weighted crosswalk from ZCTAs to NTAs.
    
    Method: Use tract-to-NTA crosswalk as intermediary.
    For each ZCTA, determine which tracts it contains/overlaps,
    then aggregate the tract-to-NTA weights.
    """
    log_step_start(logger, "build_zcta_to_nta_crosswalk")
    
    # Project for area calculations
    zctas_proj = zctas.to_crs("EPSG:2263")
    tracts_proj = tracts.to_crs("EPSG:2263")
    
    # Add populations to tracts
    tracts_proj = tracts_proj.merge(
        tract_pops[["geoid", "population"]], 
        on="geoid", 
        how="left"
    )
    tracts_proj["population"] = tracts_proj["population"].fillna(0)
    tracts_proj["tract_area"] = tracts_proj.geometry.area
    
    # Overlay ZCTAs with tracts
    logger.info("Computing ZCTA-tract intersections...")
    overlay = gpd.overlay(zctas_proj, tracts_proj, how="intersection", keep_geom_type=False)
    overlay["intersection_area"] = overlay.geometry.area
    
    # Fraction of tract in each ZCTA
    overlay["tract_fraction"] = overlay["intersection_area"] / overlay["tract_area"]
    overlay["weighted_pop"] = overlay["tract_fraction"] * overlay["population"]
    
    # Join with tract-to-NTA crosswalk
    overlay = overlay.merge(
        tract_crosswalk,
        left_on="geoid",
        right_on="source_geo_id",
        how="left"
    )
    
    # Weight for ZCTA→NTA = tract_fraction * tract_to_nta_weight * population
    overlay["zcta_nta_weighted_pop"] = overlay["weighted_pop"] * overlay["weight"]
    
    # Aggregate by ZCTA-NTA pair
    crosswalk = overlay.groupby(["zcta", "target_geo_id"]).agg({
        "zcta_nta_weighted_pop": "sum"
    }).reset_index()
    
    # Normalize weights to sum to 1 per ZCTA
    zcta_totals = crosswalk.groupby("zcta")["zcta_nta_weighted_pop"].sum().reset_index()
    zcta_totals.columns = ["zcta", "total_pop"]
    
    crosswalk = crosswalk.merge(zcta_totals, on="zcta")
    crosswalk["weight"] = crosswalk["zcta_nta_weighted_pop"] / crosswalk["total_pop"]
    crosswalk["weight"] = crosswalk["weight"].fillna(0)
    
    # Clean up
    crosswalk = crosswalk.rename(columns={
        "zcta": "source_geo_id",
        "target_geo_id": "target_geo_id"
    })
    crosswalk = crosswalk[["source_geo_id", "target_geo_id", "weight"]]
    
    # Filter out zero weights and sort
    crosswalk = crosswalk[crosswalk["weight"] > 0.001]
    crosswalk = crosswalk.sort_values(["source_geo_id", "target_geo_id"]).reset_index(drop=True)
    
    log_step_end(logger, "build_zcta_to_nta_crosswalk",
                 pair_count=len(crosswalk),
                 unique_zctas=crosswalk["source_geo_id"].nunique())
    return crosswalk


def validate_crosswalk_weights(
    crosswalk: pd.DataFrame,
    name: str,
    logger: logging.Logger,
    tolerance: float = 0.01
) -> bool:
    """Validate that crosswalk weights sum to 1 within tolerance."""
    weight_sums = crosswalk.groupby("source_geo_id")["weight"].sum()
    
    off_by = (weight_sums - 1.0).abs()
    bad_count = (off_by > tolerance).sum()
    
    if bad_count == 0:
        log_qa_check(logger, f"{name}_weights_sum_to_1", True,
                     f"All {len(weight_sums)} source geos have weights summing to 1 ± {tolerance}")
        return True
    else:
        log_qa_check(logger, f"{name}_weights_sum_to_1", False,
                     f"{bad_count} source geos have weights off by more than {tolerance}")
        return False


def create_geography_audit_log(
    crosswalks: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Create geography audit log for all crosswalks."""
    log_step_start(logger, "create_geography_audit_log")
    
    records = []
    
    for name, (xwalk, method) in crosswalks.items():
        weight_sums = xwalk.groupby("source_geo_id")["weight"].sum()
        
        records.append({
            "crosswalk_name": name,
            "source_geography": name.split("_to_")[0],
            "target_geography": name.split("_to_")[1].replace("_pop_weights", ""),
            "method": method,
            "source_geo_count": xwalk["source_geo_id"].nunique(),
            "target_geo_count": xwalk["target_geo_id"].nunique(),
            "pair_count": len(xwalk),
            "min_weight_sum": weight_sums.min(),
            "max_weight_sum": weight_sums.max(),
            "mean_weight_sum": weight_sums.mean(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    audit_df = pd.DataFrame(records)
    log_step_end(logger, "create_geography_audit_log", record_count=len(records))
    return audit_df


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load canonical NTAs
        nta_path = paths.processed_geo / "nta_canonical.parquet"
        logger.info(f"Loading NTAs from {nta_path}")
        ntas = read_geoparquet(nta_path)
        logger.info(f"Loaded {len(ntas)} NTAs")
        
        # Download census tracts
        tracts = download_census_tracts(logger)
        
        # Download ZCTAs (pass NTAs for spatial filtering)
        zctas = download_zctas(logger, ntas)
        
        # Get tract populations from ACS
        tract_pops = get_tract_populations(tracts, logger)
        
        # Build tract → NTA crosswalk
        tract_to_nta = build_tract_to_nta_crosswalk(tracts, ntas, tract_pops, logger)
        validate_crosswalk_weights(tract_to_nta, "tract_to_nta", logger)
        
        # Build ZCTA → NTA crosswalk
        zcta_to_nta = build_zcta_to_nta_crosswalk(
            zctas, ntas, tract_to_nta, tracts, tract_pops, logger
        )
        validate_crosswalk_weights(zcta_to_nta, "zcta_to_nta", logger)
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_xwalk)
        
        # Tract → NTA
        tract_path = output_dir / "tract_to_nta_pop_weights.parquet"
        atomic_write_parquet(tract_path, tract_to_nta)
        log_output_written(logger, tract_path, row_count=len(tract_to_nta))
        write_metadata_sidecar(tract_path, run_id, row_count=len(tract_to_nta))
        
        # ZCTA → NTA
        zcta_path = output_dir / "zcta_to_nta_pop_weights.parquet"
        atomic_write_parquet(zcta_path, zcta_to_nta)
        log_output_written(logger, zcta_path, row_count=len(zcta_to_nta))
        write_metadata_sidecar(zcta_path, run_id, row_count=len(zcta_to_nta))
        
        # Create and write audit log
        crosswalks = {
            "tract_to_nta_pop_weights": (tract_to_nta, "population_weighted_areal"),
            "zcta_to_nta_pop_weights": (zcta_to_nta, "population_weighted_via_tract"),
        }
        audit_log = create_geography_audit_log(crosswalks, logger)
        
        audit_path = paths.processed_geo / "geography_audit_log.parquet"
        atomic_write_parquet(audit_path, audit_log)
        log_output_written(logger, audit_path, row_count=len(audit_log))
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Tract→NTA: {len(tract_to_nta)} pairs, {tract_to_nta['source_geo_id'].nunique()} tracts")
        logger.info(f"   ZCTA→NTA: {len(zcta_to_nta)} pairs, {zcta_to_nta['source_geo_id'].nunique()} ZCTAs")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

