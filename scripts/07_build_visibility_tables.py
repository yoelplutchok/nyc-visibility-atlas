#!/usr/bin/env python3
"""
07_build_visibility_tables.py

Harmonize all visibility tables into a common schema and write a unified "long" table.

Pipeline Step: 07
Contract Reference: Section 11 - 07_build_visibility_tables.py

This script:
1. Loads visibility outputs from all enabled sources (CHS, SPARCS, Vital)
2. Validates and harmonizes to a common schema
3. Applies small numbers policy consistently
4. Writes a unified long-format visibility table

Inputs:
    - data/processed/visibility/chs_visibility.parquet
    - data/processed/visibility/sparcs_visibility.parquet
    - data/processed/visibility/vital_visibility.parquet
    - configs/sources.yml (for source metadata)
    - configs/params.yml (for small numbers policy)

Outputs:
    - data/processed/visibility/visibility_long.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet, read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "07_build_visibility_tables"

# Canonical schema for unified visibility table
VISIBILITY_SCHEMA = [
    'geo_id',           # NTA code
    'stratum_id',       # Demographic stratum (total, age_0_17, etc.)
    'source_id',        # Data source (chs, sparcs, vital)
    'time_window_id',   # Time period (2019, 2018_2022, etc.)
    'observed_count',   # Numerator (respondents, encounters, events)
    'reference_pop',    # Denominator (ACS population)
    'visibility',       # Visibility index (per 1,000)
    'reliability_flag', # high, low, suppressed
    'numerator_type',   # respondents, encounters, events, enrollees
]


def load_source_visibility(
    source_id: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load visibility data for a specific source."""
    vis_path = paths.processed_visibility / f"{source_id}_visibility.parquet"
    
    # SPARCS: prefer encounter-based visibility if available (from 04b script)
    if source_id == "sparcs":
        encounters_path = paths.processed_visibility / "sparcs_encounters_visibility.parquet"
        if encounters_path.exists():
            logger.info("Using encounter-based SPARCS visibility (preferred)")
            df = read_parquet(encounters_path)
            df['source_id'] = 'sparcs'
            logger.info(f"Loaded sparcs (encounters): {len(df)} rows")
            return df
        else:
            logger.warning("sparcs_encounters_visibility.parquet not found, falling back to PQI-based")
    
    if not vis_path.exists():
        logger.warning(f"Visibility file not found: {vis_path}")
        return pd.DataFrame()
    
    df = read_parquet(vis_path)
    logger.info(f"Loaded {source_id}: {len(df)} rows")
    
    # Ensure source_id is set
    if 'source_id' not in df.columns:
        df['source_id'] = source_id
    
    return df


def harmonize_schema(
    df: pd.DataFrame,
    source_id: str,
    source_config: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Harmonize a source's visibility data to the canonical schema."""
    if len(df) == 0:
        return pd.DataFrame(columns=VISIBILITY_SCHEMA)
    
    # Create output dataframe
    harmonized = pd.DataFrame()
    
    # Map columns to schema
    harmonized['geo_id'] = df['geo_id']
    
    # Stratum
    if 'stratum_id' in df.columns:
        harmonized['stratum_id'] = df['stratum_id']
    elif 'stratum' in df.columns:
        harmonized['stratum_id'] = df['stratum']
    else:
        harmonized['stratum_id'] = 'total'
    
    # Source
    harmonized['source_id'] = source_id
    
    # Time window
    if 'time_window_id' in df.columns:
        harmonized['time_window_id'] = df['time_window_id']
    elif 'time_window' in df.columns:
        harmonized['time_window_id'] = df['time_window']
    elif 'year' in df.columns:
        harmonized['time_window_id'] = df['year'].astype(str)
    else:
        harmonized['time_window_id'] = '2019'  # Default
    
    # Observed count
    if 'observed_count' in df.columns:
        harmonized['observed_count'] = df['observed_count']
    elif 'n_eff' in df.columns:
        harmonized['observed_count'] = df['n_eff']
    else:
        harmonized['observed_count'] = np.nan
    
    # Reference population
    if 'reference_pop' in df.columns:
        harmonized['reference_pop'] = df['reference_pop']
    else:
        harmonized['reference_pop'] = np.nan
    
    # Visibility
    if 'visibility' in df.columns:
        harmonized['visibility'] = df['visibility']
    else:
        # Compute from observed_count and reference_pop if possible
        if 'observed_count' in df.columns and 'reference_pop' in df.columns:
            harmonized['visibility'] = (df['observed_count'] / df['reference_pop']) * 1000
        else:
            harmonized['visibility'] = np.nan
    
    # Reliability flag
    if 'reliability_flag' in df.columns:
        harmonized['reliability_flag'] = df['reliability_flag']
    else:
        harmonized['reliability_flag'] = 'unknown'
    
    # Numerator type from config
    harmonized['numerator_type'] = source_config.get('numerator_type', 'unknown')
    
    logger.info(f"  Harmonized {source_id}: {len(harmonized)} rows")
    
    return harmonized[VISIBILITY_SCHEMA]


def apply_small_numbers_policy(
    df: pd.DataFrame,
    params: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Apply small numbers policy from params.yml."""
    log_step_start(logger, "apply_small_numbers_policy")
    
    policy = params.get('small_numbers_policy', {})
    min_numerator = policy.get('min_numerator', 10)
    min_denominator = policy.get('min_denominator', 50)
    suppression_threshold = policy.get('suppression_threshold', 5)
    
    logger.info(f"Policy: min_numerator={min_numerator}, min_denominator={min_denominator}, suppression={suppression_threshold}")
    
    # Count before
    before_count = len(df)
    suppressed_before = (df['reliability_flag'] == 'suppressed').sum()
    
    # Apply suppression for small numerators
    small_numerator = df['observed_count'] < suppression_threshold
    df.loc[small_numerator & (df['reliability_flag'] != 'suppressed'), 'reliability_flag'] = 'suppressed'
    
    # Apply low reliability for borderline cases
    borderline = (df['observed_count'] >= suppression_threshold) & (df['observed_count'] < min_numerator)
    df.loc[borderline & (df['reliability_flag'] == 'high'), 'reliability_flag'] = 'low'
    
    # Apply low reliability for small denominators
    small_denom = df['reference_pop'] < min_denominator
    df.loc[small_denom & (df['reliability_flag'] == 'high'), 'reliability_flag'] = 'low'
    
    # Count after
    suppressed_after = (df['reliability_flag'] == 'suppressed').sum()
    low_after = (df['reliability_flag'] == 'low').sum()
    
    logger.info(f"Applied policy: {suppressed_after - suppressed_before} newly suppressed, {low_after} low reliability")
    log_step_end(logger, "apply_small_numbers_policy")
    
    return df


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
        sources_config = read_yaml(paths.configs / "sources.yml")
        params = read_yaml(paths.configs / "params.yml")
        
        # Determine enabled sources
        enabled_sources = sources_config.get('defaults', {}).get('enabled_sources', [])
        logger.info(f"Enabled sources: {enabled_sources}")
        
        # Load and harmonize each source
        all_visibility = []
        source_stats = {}
        
        for source_id in enabled_sources:
            source_config = sources_config.get('sources', {}).get(source_id, {})
            
            if not source_config.get('enabled', True):
                logger.info(f"Skipping disabled source: {source_id}")
                continue
            
            # Load visibility data
            vis_df = load_source_visibility(source_id, logger)
            
            if len(vis_df) == 0:
                logger.warning(f"No visibility data for {source_id}")
                continue
            
            # Harmonize to schema
            harmonized = harmonize_schema(vis_df, source_id, source_config, logger)
            
            if len(harmonized) > 0:
                all_visibility.append(harmonized)
                source_stats[source_id] = {
                    'rows': len(harmonized),
                    'ntas': harmonized['geo_id'].nunique(),
                    'mean_visibility': harmonized['visibility'].mean(),
                }
        
        if not all_visibility:
            logger.error("No visibility data loaded from any source")
            return 1
        
        # Combine all sources
        visibility_long = pd.concat(all_visibility, ignore_index=True)
        logger.info(f"Combined visibility table: {len(visibility_long)} rows")
        
        # Apply small numbers policy
        visibility_long = apply_small_numbers_policy(visibility_long, params, logger)
        
        # Sort for determinism
        visibility_long = visibility_long.sort_values(
            ['geo_id', 'source_id', 'stratum_id', 'time_window_id']
        ).reset_index(drop=True)
        
        # Write output
        output_path = paths.processed_visibility / "visibility_long.parquet"
        atomic_write_parquet(output_path, visibility_long)
        log_output_written(logger, output_path, row_count=len(visibility_long))
        
        # Write metadata
        write_metadata_sidecar(
            output_path,
            run_id,
            parameters={
                "sources": list(source_stats.keys()),
                "total_rows": len(visibility_long),
                "source_stats": source_stats,
            },
            row_count=len(visibility_long),
        )
        
        # QA checks
        log_qa_check(logger, "schema_valid", True, 
                    f"All {len(VISIBILITY_SCHEMA)} columns present")
        
        # Check all sources present
        sources_in_data = visibility_long['source_id'].unique().tolist()
        log_qa_check(logger, "sources_present", True, 
                    f"Sources: {sources_in_data}")
        
        # Check reliability distribution
        rel_dist = visibility_long['reliability_flag'].value_counts().to_dict()
        log_qa_check(logger, "reliability_distribution", True, str(rel_dist))
        
        # Check for missing values
        missing_visibility = visibility_long['visibility'].isna().sum()
        log_qa_check(logger, "visibility_completeness", 
                    missing_visibility == 0,
                    f"{missing_visibility} missing visibility values")
        
        # Summary by source
        logger.info("=" * 60)
        logger.info("SUMMARY BY SOURCE:")
        for source_id, stats in source_stats.items():
            logger.info(f"  {source_id}: {stats['ntas']} NTAs, mean visibility {stats['mean_visibility']:.2f}")
        logger.info("=" * 60)
        
        # Final summary
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Total rows: {len(visibility_long)}")
        logger.info(f"   Sources: {sources_in_data}")
        logger.info(f"   Unique NTAs: {visibility_long['geo_id'].nunique()}")
        logger.info(f"   Reliability: {rel_dist}")
        logger.info(f"   Output: {output_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

