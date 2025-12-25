#!/usr/bin/env python3
"""
09_typology_clustering.py

Cluster neighborhoods into visibility typologies and quantify stability.

Pipeline Step: 09
Contract Reference: Section 11 - 09_typology_clustering.py

This script:
1. Clusters neighborhoods based on cross-source visibility patterns
2. Labels typologies with human-readable names
3. Assesses cluster stability via bootstrap resampling
4. Produces typology assignments and stability metrics

Inputs:
    - data/processed/matrix/visibility_matrix_pctrank.parquet

Outputs:
    - data/processed/typologies/typology_assignments.parquet
    - data/processed/typologies/typology_stability.parquet
    - data/processed/typologies/typology_profiles.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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


SCRIPT_NAME = "09_typology_clustering"

# Random seed for reproducibility
RANDOM_SEED = 42


def select_optimal_k(
    X: np.ndarray,
    k_range: range,
    logger: logging.Logger
) -> int:
    """
    Select optimal number of clusters using silhouette score.
    """
    log_step_start(logger, "select_optimal_k")
    
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1
        
        scores.append({'k': k, 'silhouette': score})
        logger.info(f"  k={k}: silhouette={score:.3f}")
    
    scores_df = pd.DataFrame(scores)
    optimal_k = scores_df.loc[scores_df['silhouette'].idxmax(), 'k']
    
    logger.info(f"Optimal k: {optimal_k}")
    log_step_end(logger, "select_optimal_k", optimal_k=optimal_k)
    
    return int(optimal_k)


def cluster_neighborhoods(
    pctrank_matrix: pd.DataFrame,
    n_clusters: int,
    logger: logging.Logger
) -> tuple:
    """
    Cluster neighborhoods using K-means on percentile rank visibility.
    """
    log_step_start(logger, "cluster_neighborhoods")
    
    # Get source columns
    source_cols = ['chs', 'sparcs', 'vital']
    source_cols = [c for c in source_cols if c in pctrank_matrix.columns]
    
    # Prepare data - drop rows with any missing values
    cluster_data = pctrank_matrix[['geo_id'] + source_cols].copy()
    complete_mask = cluster_data[source_cols].notna().all(axis=1)
    cluster_data = cluster_data[complete_mask].reset_index(drop=True)
    
    logger.info(f"Clustering {len(cluster_data)} neighborhoods with complete data")
    
    # Prepare feature matrix
    X = cluster_data[source_cols].values
    
    # Standardize (optional, but percentile ranks are already on same scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    cluster_data['cluster_id'] = labels
    
    # Compute silhouette
    silhouette = silhouette_score(X_scaled, labels)
    logger.info(f"Silhouette score: {silhouette:.3f}")
    
    log_step_end(logger, "cluster_neighborhoods", n_neighborhoods=len(cluster_data))
    
    return cluster_data, kmeans, X_scaled, silhouette


def label_typologies(
    cluster_data: pd.DataFrame,
    source_cols: list,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Assign human-readable labels to clusters based on their visibility profiles.
    """
    log_step_start(logger, "label_typologies")
    
    # Compute cluster means
    cluster_profiles = cluster_data.groupby('cluster_id')[source_cols].mean()
    
    # Label based on visibility patterns
    labels = {}
    
    for cluster_id, row in cluster_profiles.iterrows():
        # Categorize each source as high (>66), medium (33-66), or low (<33)
        categorized = {}
        for source in source_cols:
            val = row[source]
            if val >= 66:
                categorized[source] = 'high'
            elif val >= 33:
                categorized[source] = 'medium'
            else:
                categorized[source] = 'low'
        
        # Generate label
        label_parts = []
        
        # Check for uniform patterns
        if all(c == 'high' for c in categorized.values()):
            label = "High visibility across systems"
        elif all(c == 'low' for c in categorized.values()):
            label = "Low visibility across systems"
        elif all(c == 'medium' for c in categorized.values()):
            label = "Medium visibility across systems"
        else:
            # Mixed patterns
            high_sources = [s for s, c in categorized.items() if c == 'high']
            low_sources = [s for s, c in categorized.items() if c == 'low']
            
            if high_sources and low_sources:
                high_str = '/'.join(s.upper() for s in high_sources)
                low_str = '/'.join(s.upper() for s in low_sources)
                label = f"High {high_str}, Low {low_str}"
            elif high_sources:
                high_str = '/'.join(s.upper() for s in high_sources)
                label = f"High {high_str} only"
            elif low_sources:
                low_str = '/'.join(s.upper() for s in low_sources)
                label = f"Low {low_str}, medium elsewhere"
            else:
                label = f"Mixed visibility pattern"
        
        labels[cluster_id] = label
        logger.info(f"  Cluster {cluster_id}: {label}")
        logger.info(f"    Profile: {dict(row.round(1))}")
    
    # Add labels to data
    cluster_data['typology_label'] = cluster_data['cluster_id'].map(labels)
    
    log_step_end(logger, "label_typologies", n_typologies=len(labels))
    
    return cluster_data


def assess_stability(
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Assess cluster stability via bootstrap resampling.
    """
    log_step_start(logger, "assess_stability")
    
    n_samples = len(labels)
    n_clusters = len(np.unique(labels))
    
    # Track how often each sample is assigned to its original cluster
    stability_scores = np.zeros(n_samples)
    
    np.random.seed(RANDOM_SEED)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[boot_idx]
        
        # Fit new clustering
        kmeans_boot = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED + i, n_init=5)
        labels_boot = kmeans_boot.fit_predict(X)
        
        # Count matching assignments (simplified - just count consistent neighbors)
        for j in range(n_samples):
            same_original = labels == labels[j]
            same_boot = labels_boot == labels_boot[j]
            stability_scores[j] += np.mean(same_original == same_boot)
    
    stability_scores /= n_bootstrap
    
    mean_stability = stability_scores.mean()
    logger.info(f"Mean stability score: {mean_stability:.3f}")
    
    log_step_end(logger, "assess_stability", mean_stability=mean_stability)
    
    return stability_scores


def create_typology_profiles(
    cluster_data: pd.DataFrame,
    source_cols: list,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Create summary profiles for each typology.
    """
    profiles = cluster_data.groupby(['cluster_id', 'typology_label']).agg({
        'geo_id': 'count',
        **{col: ['mean', 'std', 'min', 'max'] for col in source_cols}
    }).reset_index()
    
    # Flatten column names
    profiles.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                       for col in profiles.columns]
    profiles = profiles.rename(columns={'geo_id_count': 'n_neighborhoods'})
    
    return profiles


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info("=" * 60)
    
    try:
        # Load percentile rank matrix
        pctrank_path = paths.processed_matrix / "visibility_matrix_pctrank.parquet"
        pctrank_matrix = read_parquet(pctrank_path)
        logger.info(f"Loaded percentile rank matrix: {len(pctrank_matrix)} rows")
        
        # Get source columns
        source_cols = ['chs', 'sparcs', 'vital']
        source_cols = [c for c in source_cols if c in pctrank_matrix.columns]
        logger.info(f"Source columns: {source_cols}")
        
        # Prepare data for clustering
        complete_data = pctrank_matrix[['geo_id'] + source_cols].dropna()
        X = complete_data[source_cols].values
        
        # Select optimal k
        logger.info("Selecting optimal number of clusters...")
        optimal_k = select_optimal_k(X, range(3, 8), logger)
        
        # Use 4 or 5 clusters for interpretability (or optimal_k)
        n_clusters = min(max(optimal_k, 4), 6)
        logger.info(f"Using {n_clusters} clusters")
        
        # Cluster neighborhoods
        cluster_data, kmeans, X_scaled, silhouette = cluster_neighborhoods(
            pctrank_matrix, n_clusters, logger
        )
        
        # Label typologies
        cluster_data = label_typologies(cluster_data, source_cols, logger)
        
        # Assess stability
        stability_scores = assess_stability(
            X_scaled, cluster_data['cluster_id'].values, 
            n_bootstrap=50, logger=logger
        )
        cluster_data['stability_score'] = stability_scores
        
        # Create profiles
        profiles = create_typology_profiles(cluster_data, source_cols, logger)
        
        # Create stability summary
        stability_summary = cluster_data.groupby('cluster_id').agg({
            'stability_score': ['mean', 'std', 'min', 'max'],
            'geo_id': 'count'
        }).reset_index()
        stability_summary.columns = ['cluster_id', 'mean_stability', 'std_stability', 
                                    'min_stability', 'max_stability', 'n_neighborhoods']
        
        # Write outputs
        output_dir = ensure_dir(paths.processed_typologies)
        
        # Typology assignments
        assignments_path = output_dir / "typology_assignments.parquet"
        atomic_write_parquet(assignments_path, cluster_data)
        log_output_written(logger, assignments_path, row_count=len(cluster_data))
        
        # Stability summary
        stability_path = output_dir / "typology_stability.parquet"
        atomic_write_parquet(stability_path, stability_summary)
        log_output_written(logger, stability_path, row_count=len(stability_summary))
        
        # Profiles
        profiles_path = output_dir / "typology_profiles.parquet"
        atomic_write_parquet(profiles_path, profiles)
        log_output_written(logger, profiles_path, row_count=len(profiles))
        
        # Write metadata
        write_metadata_sidecar(
            assignments_path,
            run_id,
            parameters={
                "n_clusters": n_clusters,
                "random_seed": RANDOM_SEED,
                "silhouette_score": float(silhouette),
                "mean_stability": float(cluster_data['stability_score'].mean()),
            },
            row_count=len(cluster_data),
        )
        
        # QA checks
        log_qa_check(logger, "silhouette_score", silhouette > 0.2,
                    f"Silhouette={silhouette:.3f}")
        
        mean_stability = cluster_data['stability_score'].mean()
        log_qa_check(logger, "cluster_stability", mean_stability > 0.5,
                    f"Mean stability={mean_stability:.3f}")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Neighborhoods clustered: {len(cluster_data)}")
        logger.info(f"   Number of typologies: {n_clusters}")
        logger.info(f"   Silhouette score: {silhouette:.3f}")
        logger.info(f"   Mean stability: {mean_stability:.3f}")
        logger.info("=" * 60)
        logger.info("TYPOLOGY DISTRIBUTION:")
        for label, count in cluster_data['typology_label'].value_counts().items():
            logger.info(f"   {label}: {count} neighborhoods")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

