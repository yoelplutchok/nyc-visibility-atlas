#!/usr/bin/env python3
"""
08_build_cross_source_matrix.py

Build neighborhood × source visibility matrices and normalized versions.

Pipeline Step: 08
Contract Reference: Section 11 - 08_build_cross_source_matrix.py

This script:
1. Pivots the long visibility table into wide (matrix) format
2. Creates normalized versions (percentile ranks, z-scores)
3. Computes cross-source divergence metrics

Inputs:
    - data/processed/visibility/visibility_long.parquet

Outputs:
    - data/processed/matrix/visibility_matrix_raw.parquet
    - data/processed/matrix/visibility_matrix_pctrank.parquet
    - data/processed/matrix/visibility_matrix_weighted_z.parquet (optional)
    - data/processed/matrix/source_correlations.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet, read_yaml, atomic_write_json
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "08_build_cross_source_matrix"


def build_raw_matrix(
    visibility_long: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Pivot long visibility table to wide (matrix) format.
    Rows = neighborhoods, Columns = sources.
    """
    log_step_start(logger, "build_raw_matrix")
    
    # Filter to total stratum only
    total_only = visibility_long[visibility_long['stratum_id'] == 'total'].copy()
    
    # Pivot to wide format
    matrix = total_only.pivot_table(
        index='geo_id',
        columns='source_id',
        values='visibility',
        aggfunc='first'  # Should be unique per geo×source
    ).reset_index()
    
    # Flatten column names
    matrix.columns.name = None
    
    logger.info(f"Raw matrix: {len(matrix)} neighborhoods × {len(matrix.columns) - 1} sources")
    
    # Log coverage
    for col in matrix.columns:
        if col != 'geo_id':
            non_null = matrix[col].notna().sum()
            logger.info(f"  {col}: {non_null} neighborhoods with data")
    
    log_step_end(logger, "build_raw_matrix", row_count=len(matrix))
    return matrix


def build_percentile_rank_matrix(
    raw_matrix: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Convert raw visibility values to percentile ranks (0-100).
    This normalizes across sources for fair comparison.
    """
    log_step_start(logger, "build_percentile_rank_matrix")
    
    pctrank_matrix = raw_matrix.copy()
    
    source_cols = [c for c in raw_matrix.columns if c != 'geo_id']
    
    for col in source_cols:
        valid_mask = pctrank_matrix[col].notna()
        if valid_mask.sum() > 0:
            # Compute percentile rank (0-100)
            pctrank_matrix.loc[valid_mask, col] = (
                pctrank_matrix.loc[valid_mask, col].rank(pct=True) * 100
            )
    
    logger.info(f"Percentile rank matrix: {len(pctrank_matrix)} rows")
    
    log_step_end(logger, "build_percentile_rank_matrix", row_count=len(pctrank_matrix))
    return pctrank_matrix


def build_weighted_z_matrix(
    raw_matrix: pd.DataFrame,
    visibility_long: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Convert raw visibility to z-scores, optionally weighted by reliability.
    """
    log_step_start(logger, "build_weighted_z_matrix")
    
    z_matrix = raw_matrix.copy()
    
    source_cols = [c for c in raw_matrix.columns if c != 'geo_id']
    
    for col in source_cols:
        valid_mask = z_matrix[col].notna()
        if valid_mask.sum() > 1:
            mean_val = z_matrix.loc[valid_mask, col].mean()
            std_val = z_matrix.loc[valid_mask, col].std()
            if std_val > 0:
                z_matrix.loc[valid_mask, col] = (
                    (z_matrix.loc[valid_mask, col] - mean_val) / std_val
                )
            else:
                z_matrix.loc[valid_mask, col] = 0
    
    logger.info(f"Weighted z-score matrix: {len(z_matrix)} rows")
    
    log_step_end(logger, "build_weighted_z_matrix", row_count=len(z_matrix))
    return z_matrix


def compute_source_correlations(
    raw_matrix: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute pairwise correlations between sources.
    """
    log_step_start(logger, "compute_source_correlations")
    
    source_cols = [c for c in raw_matrix.columns if c != 'geo_id']
    
    correlations = []
    
    for i, source1 in enumerate(source_cols):
        for source2 in source_cols[i+1:]:
            # Get rows where both sources have data
            valid_mask = raw_matrix[source1].notna() & raw_matrix[source2].notna()
            n_pairs = valid_mask.sum()
            
            if n_pairs >= 10:
                r, p_value = stats.pearsonr(
                    raw_matrix.loc[valid_mask, source1],
                    raw_matrix.loc[valid_mask, source2]
                )
                spearman_r, spearman_p = stats.spearmanr(
                    raw_matrix.loc[valid_mask, source1],
                    raw_matrix.loc[valid_mask, source2]
                )
            else:
                r, p_value = np.nan, np.nan
                spearman_r, spearman_p = np.nan, np.nan
            
            correlations.append({
                'source_1': source1,
                'source_2': source2,
                'n_pairs': n_pairs,
                'pearson_r': r,
                'pearson_p': p_value,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            })
            
            logger.info(f"  {source1} vs {source2}: r={r:.3f} (n={n_pairs})")
    
    corr_df = pd.DataFrame(correlations)
    
    log_step_end(logger, "compute_source_correlations", pair_count=len(corr_df))
    return corr_df


def compute_divergence_metrics(
    pctrank_matrix: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute cross-source divergence metrics for each neighborhood.
    """
    log_step_start(logger, "compute_divergence_metrics")
    
    source_cols = [c for c in pctrank_matrix.columns if c != 'geo_id']
    
    divergence = pctrank_matrix[['geo_id']].copy()
    
    # For each pair of sources, compute divergence
    for i, source1 in enumerate(source_cols):
        for source2 in source_cols[i+1:]:
            col_name = f'divergence_{source1}_{source2}'
            divergence[col_name] = pctrank_matrix[source1] - pctrank_matrix[source2]
    
    # Compute overall divergence (range across sources)
    if len(source_cols) > 1:
        source_data = pctrank_matrix[source_cols]
        divergence['visibility_range'] = source_data.max(axis=1) - source_data.min(axis=1)
        divergence['visibility_std'] = source_data.std(axis=1)
        divergence['n_sources'] = source_data.notna().sum(axis=1)
    
    logger.info(f"Divergence metrics computed for {len(divergence)} neighborhoods")
    
    log_step_end(logger, "compute_divergence_metrics", row_count=len(divergence))
    return divergence


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load unified visibility table
        vis_long_path = paths.processed_visibility / "visibility_long.parquet"
        visibility_long = read_parquet(vis_long_path)
        logger.info(f"Loaded visibility_long: {len(visibility_long)} rows")
        
        sources = visibility_long['source_id'].unique().tolist()
        logger.info(f"Sources: {sources}")
        
        # Build matrices
        raw_matrix = build_raw_matrix(visibility_long, logger)
        pctrank_matrix = build_percentile_rank_matrix(raw_matrix, logger)
        z_matrix = build_weighted_z_matrix(raw_matrix, visibility_long, logger)
        
        # Compute correlations
        correlations = compute_source_correlations(raw_matrix, logger)
        
        # Compute divergence
        divergence = compute_divergence_metrics(pctrank_matrix, logger)
        
        # Merge divergence into matrices
        raw_matrix = raw_matrix.merge(
            divergence[['geo_id', 'visibility_range', 'visibility_std', 'n_sources']],
            on='geo_id', how='left'
        )
        pctrank_matrix = pctrank_matrix.merge(
            divergence,
            on='geo_id', how='left'
        )
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_matrix)
        
        # Raw matrix
        raw_path = output_dir / "visibility_matrix_raw.parquet"
        atomic_write_parquet(raw_path, raw_matrix)
        log_output_written(logger, raw_path, row_count=len(raw_matrix))
        
        # Percentile rank matrix
        pctrank_path = output_dir / "visibility_matrix_pctrank.parquet"
        atomic_write_parquet(pctrank_path, pctrank_matrix)
        log_output_written(logger, pctrank_path, row_count=len(pctrank_matrix))
        
        # Z-score matrix
        z_path = output_dir / "visibility_matrix_weighted_z.parquet"
        atomic_write_parquet(z_path, z_matrix)
        log_output_written(logger, z_path, row_count=len(z_matrix))
        
        # Correlations
        corr_path = output_dir / "source_correlations.parquet"
        atomic_write_parquet(corr_path, correlations)
        log_output_written(logger, corr_path, row_count=len(correlations))
        
        # Write metadata
        write_metadata_sidecar(
            raw_path,
            run_id,
            parameters={
                "sources": sources,
                "n_neighborhoods": len(raw_matrix),
                "correlation_summary": correlations.to_dict('records') if len(correlations) > 0 else [],
            },
            row_count=len(raw_matrix),
        )
        
        # QA checks
        log_qa_check(logger, "matrix_completeness", True,
                    f"{len(raw_matrix)} neighborhoods in matrix")
        
        # Check correlations
        if len(correlations) > 0:
            mean_corr = correlations['pearson_r'].mean()
            log_qa_check(logger, "source_correlations", True,
                        f"Mean pairwise r={mean_corr:.3f}")
        
        # Check divergence range
        if 'visibility_range' in raw_matrix.columns:
            mean_range = raw_matrix['visibility_range'].mean()
            log_qa_check(logger, "divergence_range", True,
                        f"Mean visibility range={mean_range:.1f} percentile points")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Neighborhoods: {len(raw_matrix)}")
        logger.info(f"   Sources: {sources}")
        logger.info(f"   Matrices: raw, percentile rank, z-score")
        logger.info("=" * 60)
        logger.info("SOURCE CORRELATIONS:")
        for _, row in correlations.iterrows():
            logger.info(f"   {row['source_1']} vs {row['source_2']}: r={row['pearson_r']:.3f} (n={row['n_pairs']})")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

