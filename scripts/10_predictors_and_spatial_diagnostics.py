#!/usr/bin/env python3
"""
10_predictors_and_spatial_diagnostics.py

Model structural predictors of visibility patterns + spatial diagnostics.

Pipeline Step: 10
Contract Reference: Section 11 - 10_predictors_and_spatial_diagnostics.py

This script:
1. Loads ACS neighborhood characteristics as predictors
2. Fits GLMs to predict visibility from socioeconomic factors
3. Computes Moran's I for spatial autocorrelation diagnostics
4. Produces model summaries and coefficient tables

Inputs:
    - data/processed/visibility/visibility_long.parquet
    - data/processed/denominators/acs_denominators.parquet
    - data/processed/geo/nta_canonical.parquet

Outputs:
    - data/processed/models/predictor_models.parquet
    - reports/tables/predictor_model_summary.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import warnings
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
    atomic_write_parquet, read_parquet, read_geoparquet,
    read_yaml, atomic_write_json, atomic_write_text
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "10_predictors_and_spatial_diagnostics"

# Suppress warnings
warnings.filterwarnings('ignore')


def load_neighborhood_characteristics(
    denominators: pd.DataFrame,
    ntas: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load and prepare neighborhood characteristics as predictors.
    Uses population by age band as proxy for demographic structure.
    """
    log_step_start(logger, "load_neighborhood_characteristics")
    
    # Pivot denominators to get age structure
    total_pop = denominators[denominators['stratum_id'] == 'total'][['geo_id', 'reference_pop']].copy()
    total_pop.columns = ['geo_id', 'total_pop']
    
    # Get age band populations
    age_pops = denominators[denominators['stratum_id'] != 'total'].pivot_table(
        index='geo_id',
        columns='stratum_id', 
        values='reference_pop',
        aggfunc='first'
    ).reset_index()
    
    # Merge with totals
    chars = total_pop.merge(age_pops, on='geo_id', how='left')
    
    # Calculate age proportions
    age_cols = [c for c in chars.columns if c.startswith('age_')]
    for col in age_cols:
        pct_col = f'pct_{col}'
        chars[pct_col] = (chars[col] / chars['total_pop'] * 100).fillna(0)
    
    # Add borough from NTAs
    chars = chars.merge(ntas[['geo_id', 'borough']], on='geo_id', how='left')
    
    # Create borough dummies
    borough_dummies = pd.get_dummies(chars['borough'], prefix='borough')
    chars = pd.concat([chars, borough_dummies], axis=1)
    
    logger.info(f"Loaded characteristics for {len(chars)} neighborhoods")
    logger.info(f"Columns: {list(chars.columns)}")
    
    log_step_end(logger, "load_neighborhood_characteristics", n_neighborhoods=len(chars))
    return chars


def fit_visibility_model(
    visibility: pd.DataFrame,
    characteristics: pd.DataFrame,
    source_id: str,
    logger: logging.Logger
) -> dict:
    """
    Fit a linear model predicting visibility from neighborhood characteristics.
    
    Returns model coefficients and diagnostics.
    """
    log_step_start(logger, f"fit_model_{source_id}")
    
    # Get visibility for this source
    source_vis = visibility[visibility['source_id'] == source_id][['geo_id', 'visibility']].copy()
    
    # Merge with characteristics
    model_data = source_vis.merge(characteristics, on='geo_id', how='inner')
    
    if len(model_data) < 20:
        logger.warning(f"Insufficient data for {source_id}: {len(model_data)} rows")
        return None
    
    # Select predictors - use age proportions and borough dummies
    predictor_cols = [c for c in model_data.columns 
                     if c.startswith('pct_age_') or c.startswith('borough_')]
    
    # Drop one borough for reference category (avoid multicollinearity)
    if 'borough_Manhattan' in predictor_cols:
        predictor_cols.remove('borough_Manhattan')
    
    # Prepare X and y - ensure numeric types
    X = model_data[predictor_cols].astype(float).values
    y = model_data['visibility'].astype(float).values
    
    # Handle any NaN/inf
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(y) < 20:
        logger.warning(f"Insufficient valid data for {source_id}")
        return None
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Fit OLS
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        
        # Compute R-squared
        y_pred = X_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Store results
        coef_names = ['intercept'] + predictor_cols
        results = {
            'source_id': source_id,
            'n_observations': len(y),
            'r_squared': r_squared,
            'coefficients': dict(zip(coef_names, coeffs)),
            'residuals': y - y_pred,
            'y_pred': y_pred,
            'geo_ids': model_data.loc[valid_mask, 'geo_id'].values,
        }
        
        logger.info(f"  {source_id}: R²={r_squared:.3f}, n={len(y)}")
        
    except Exception as e:
        logger.warning(f"Model fitting failed for {source_id}: {e}")
        return None
    
    log_step_end(logger, f"fit_model_{source_id}")
    return results


def compute_morans_i(
    geo_ids: np.ndarray,
    residuals: np.ndarray,
    ntas: pd.DataFrame,
    logger: logging.Logger
) -> dict:
    """
    Compute Moran's I for spatial autocorrelation of residuals.
    Uses queen contiguity weights (shared boundaries).
    """
    try:
        # Build simple distance-based weights
        ntas_subset = ntas[ntas['geo_id'].isin(geo_ids)].copy()
        centroids = ntas_subset.geometry.centroid
        
        n = len(residuals)
        
        # Simple contiguity approximation: weight based on inverse distance
        coords = np.array([[c.x, c.y] for c in centroids])
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(coords, coords)
        
        # Create inverse distance weights (with threshold)
        threshold = np.percentile(distances[distances > 0], 10)  # 10th percentile
        W = np.where((distances > 0) & (distances < threshold * 3), 1 / distances, 0)
        
        # Row standardize
        row_sums = W.sum(axis=1, keepdims=True)
        W = np.where(row_sums > 0, W / row_sums, 0)
        
        # Compute Moran's I
        z = residuals - np.mean(residuals)
        numerator = n * np.sum(W * np.outer(z, z))
        denominator = np.sum(W) * np.sum(z ** 2)
        
        morans_i = numerator / denominator if denominator > 0 else 0
        
        return {'morans_i': morans_i, 'n': n}
        
    except Exception as e:
        logger.warning(f"Moran's I computation failed: {e}")
        return {'morans_i': np.nan, 'n': 0}


def create_model_summary_md(
    model_results: list,
    output_path: Path,
    logger: logging.Logger
):
    """Create markdown summary of model results."""
    
    lines = [
        "# Visibility Predictor Model Summary",
        "",
        "## Overview",
        "Linear regression models predicting visibility from neighborhood characteristics.",
        "",
        "## Model Results by Source",
        "",
    ]
    
    for result in model_results:
        if result is None:
            continue
            
        lines.extend([
            f"### {result['source_id'].upper()}",
            "",
            f"- **N observations:** {result['n_observations']}",
            f"- **R²:** {result['r_squared']:.3f}",
            f"- **Moran's I (residuals):** {result.get('morans_i', 'N/A')}",
            "",
            "**Coefficients:**",
            "",
            "| Predictor | Coefficient |",
            "|-----------|-------------|",
        ])
        
        for name, coef in result['coefficients'].items():
            lines.append(f"| {name} | {coef:.4f} |")
        
        lines.append("")
    
    lines.extend([
        "## Interpretation Notes",
        "",
        "- R² indicates proportion of variance explained by neighborhood demographics",
        "- Borough effects capture systematic differences across boroughs",
        "- Low R² suggests visibility is NOT strongly determined by measurable demographics",
        "- Moran's I > 0 indicates positive spatial autocorrelation of residuals",
    ])
    
    atomic_write_text(output_path, "\n".join(lines))
    logger.info(f"Wrote model summary to {output_path}")


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load data
        visibility = read_parquet(paths.processed_visibility / "visibility_long.parquet")
        denominators = read_parquet(paths.processed_denominators / "acs_denominators.parquet")
        ntas = read_geoparquet(paths.processed_geo / "nta_canonical.parquet")
        
        logger.info(f"Loaded visibility: {len(visibility)} rows")
        logger.info(f"Loaded denominators: {len(denominators)} rows")
        logger.info(f"Loaded NTAs: {len(ntas)} rows")
        
        # Load neighborhood characteristics
        characteristics = load_neighborhood_characteristics(denominators, ntas, logger)
        
        # Fit models for each source
        sources = visibility['source_id'].unique()
        model_results = []
        
        for source_id in sources:
            result = fit_visibility_model(visibility, characteristics, source_id, logger)
            if result:
                # Compute Moran's I
                morans = compute_morans_i(
                    result['geo_ids'], 
                    result['residuals'],
                    ntas, 
                    logger
                )
                result['morans_i'] = morans['morans_i']
                model_results.append(result)
        
        # Create coefficient table
        coef_rows = []
        for result in model_results:
            for name, coef in result['coefficients'].items():
                coef_rows.append({
                    'source_id': result['source_id'],
                    'predictor': name,
                    'coefficient': coef,
                    'r_squared': result['r_squared'],
                    'morans_i': result.get('morans_i', np.nan),
                    'n_observations': result['n_observations'],
                })
        
        coef_df = pd.DataFrame(coef_rows)
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_models)
        
        coef_path = output_dir / "predictor_models.parquet"
        atomic_write_parquet(coef_path, coef_df)
        log_output_written(logger, coef_path, row_count=len(coef_df))
        
        # Write markdown summary
        reports_dir = ensure_dir(paths.reports / "tables")
        summary_path = reports_dir / "predictor_model_summary.md"
        create_model_summary_md(model_results, summary_path, logger)
        
        # Write metadata
        write_metadata_sidecar(
            coef_path,
            run_id,
            parameters={
                "sources_modeled": [r['source_id'] for r in model_results],
                "r_squared_values": {r['source_id']: r['r_squared'] for r in model_results},
            },
            row_count=len(coef_df),
        )
        
        # QA checks
        for result in model_results:
            log_qa_check(logger, f"model_{result['source_id']}_fit", 
                        result['r_squared'] >= 0,
                        f"R²={result['r_squared']:.3f}")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info("MODEL R² VALUES:")
        for result in model_results:
            morans_val = result.get('morans_i')
            morans_str = f"{morans_val:.3f}" if isinstance(morans_val, float) and not np.isnan(morans_val) else 'N/A'
            logger.info(f"   {result['source_id']}: R²={result['r_squared']:.3f}, Moran's I={morans_str}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

