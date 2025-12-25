#!/usr/bin/env python3
"""
12_atlas_assets.py

Build final atlas assets: maps, tables, and data exports for visualization.

Pipeline Step: 12
Contract Reference: Section 11 - 12_atlas_assets.py

This script:
1. Joins visibility data with NTA geometries
2. Creates GeoJSON exports for mapping
3. Creates summary tables and statistics
4. Produces README for the final outputs

Inputs:
    - data/processed/geo/nta_canonical.parquet
    - data/processed/visibility/visibility_long.parquet
    - data/processed/matrix/visibility_matrix_pctrank.parquet
    - data/processed/typologies/typology_assignments.parquet
    - data/processed/vulnerability/vulnerability_scores.parquet

Outputs:
    - data/final/nta_visibility_atlas.geojson
    - data/final/nta_visibility_atlas.csv
    - data/final/summary_statistics.json
    - data/final/README.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd

from visibility_atlas.paths import paths, ensure_dir
from visibility_atlas.logging_utils import (
    get_logger, log_step_start, log_step_end,
    log_qa_check, log_output_written, get_run_id
)
from visibility_atlas.io_utils import (
    atomic_write_parquet, read_parquet, read_geoparquet,
    atomic_write_json, atomic_write_geojson, atomic_write_text
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "12_atlas_assets"


def build_atlas_gdf(
    ntas: gpd.GeoDataFrame,
    visibility: pd.DataFrame,
    pctrank: pd.DataFrame,
    typologies: pd.DataFrame,
    vulnerability: pd.DataFrame,
    logger: logging.Logger
) -> gpd.GeoDataFrame:
    """
    Build the main atlas GeoDataFrame with all visibility metrics.
    """
    log_step_start(logger, "build_atlas_gdf")
    
    # Start with NTAs
    atlas = ntas[['geo_id', 'geo_name', 'borough', 'geometry']].copy()
    atlas = atlas.rename(columns={'geo_name': 'nta_name'})
    
    # Pivot visibility to wide format
    vis_wide = visibility.pivot_table(
        index='geo_id',
        columns='source_id',
        values='visibility',
        aggfunc='first'
    ).reset_index()
    vis_wide.columns.name = None
    
    # Rename visibility columns
    vis_rename = {c: f'vis_{c}' for c in vis_wide.columns if c != 'geo_id'}
    vis_wide = vis_wide.rename(columns=vis_rename)
    
    # Add percentile rank columns
    pctrank_cols = ['chs', 'sparcs', 'vital', 'visibility_range']
    pctrank_subset = pctrank[['geo_id'] + [c for c in pctrank_cols if c in pctrank.columns]].copy()
    pctrank_rename = {c: f'pctile_{c}' for c in pctrank_subset.columns if c != 'geo_id'}
    pctrank_subset = pctrank_subset.rename(columns=pctrank_rename)
    
    # Add typology
    if 'typology_label' in typologies.columns:
        typology_subset = typologies[['geo_id', 'cluster_id', 'typology_label', 'stability_score']].copy()
    else:
        typology_subset = typologies[['geo_id']].copy()
        typology_subset['cluster_id'] = -1
        typology_subset['typology_label'] = 'unknown'
        typology_subset['stability_score'] = 0
    
    # Add vulnerability
    vuln_cols = ['vulnerability_score', 'vulnerability_category', 'vulnerability_rank',
                 'consequence_score', 'invisibility_score']
    vuln_subset = vulnerability[['geo_id'] + [c for c in vuln_cols if c in vulnerability.columns]].copy()
    
    # Merge all
    atlas = atlas.merge(vis_wide, on='geo_id', how='left')
    atlas = atlas.merge(pctrank_subset, on='geo_id', how='left')
    atlas = atlas.merge(typology_subset, on='geo_id', how='left')
    atlas = atlas.merge(vuln_subset, on='geo_id', how='left')
    
    logger.info(f"Built atlas with {len(atlas)} neighborhoods and {len(atlas.columns)} columns")
    logger.info(f"Columns: {list(atlas.columns)}")
    
    log_step_end(logger, "build_atlas_gdf", n_neighborhoods=len(atlas))
    return atlas


def compute_summary_statistics(
    atlas: gpd.GeoDataFrame,
    logger: logging.Logger
) -> dict:
    """Compute summary statistics for the atlas."""
    
    stats = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_neighborhoods": len(atlas),
            "n_boroughs": atlas['borough'].nunique(),
        },
        "visibility": {},
        "typology": {},
        "vulnerability": {},
        "correlations": {},
    }
    
    # Visibility stats by source
    for col in atlas.columns:
        if col.startswith('vis_'):
            source = col.replace('vis_', '')
            valid = atlas[col].dropna()
            stats["visibility"][source] = {
                "count": int(len(valid)),
                "mean": float(valid.mean()) if len(valid) > 0 else None,
                "median": float(valid.median()) if len(valid) > 0 else None,
                "min": float(valid.min()) if len(valid) > 0 else None,
                "max": float(valid.max()) if len(valid) > 0 else None,
            }
    
    # Typology distribution
    if 'typology_label' in atlas.columns:
        stats["typology"]["distribution"] = atlas['typology_label'].value_counts().to_dict()
        stats["typology"]["n_clusters"] = atlas['cluster_id'].nunique()
        if 'stability_score' in atlas.columns:
            stats["typology"]["mean_stability"] = float(atlas['stability_score'].mean())
    
    # Vulnerability stats
    if 'vulnerability_category' in atlas.columns:
        stats["vulnerability"]["distribution"] = atlas['vulnerability_category'].value_counts().to_dict()
        if 'vulnerability_score' in atlas.columns:
            vuln = atlas['vulnerability_score'].dropna()
            stats["vulnerability"]["mean_score"] = float(vuln.mean()) if len(vuln) > 0 else None
    
    # Correlations between visibility sources
    vis_cols = [c for c in atlas.columns if c.startswith('vis_')]
    if len(vis_cols) >= 2:
        for i, col1 in enumerate(vis_cols):
            for col2 in vis_cols[i+1:]:
                valid = atlas[[col1, col2]].dropna()
                if len(valid) >= 10:
                    r = valid[col1].corr(valid[col2])
                    stats["correlations"][f"{col1}_vs_{col2}"] = float(r)
    
    return stats


def create_atlas_readme(output_path: Path, stats: dict, logger: logging.Logger):
    """Create README for final atlas outputs."""
    
    lines = [
        "# NYC Visibility Atlas - Final Outputs",
        "",
        f"Generated: {stats['metadata']['generated_at']}",
        "",
        "## Overview",
        "",
        f"This atlas quantifies how different public health data systems 'see' NYC neighborhoods.",
        "",
        f"- **{stats['metadata']['n_neighborhoods']}** neighborhoods analyzed",
        f"- **{stats['metadata']['n_boroughs']}** boroughs",
        f"- **3** data sources: CHS (surveys), SPARCS (hospitals), Vital (deaths)",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `nta_visibility_atlas.geojson` | Full atlas with geometries (for mapping) |",
        "| `nta_visibility_atlas.csv` | Tabular data without geometries |",
        "| `summary_statistics.json` | Summary statistics and correlations |",
        "",
        "## Key Columns",
        "",
        "### Visibility (raw rates)",
        "| Column | Description |",
        "|--------|-------------|",
        "| `vis_chs` | CHS survey visibility (per 1,000 residents) |",
        "| `vis_sparcs` | SPARCS hospital visibility (per 1,000 residents) |",
        "| `vis_vital` | Vital records visibility (per 1,000 residents) |",
        "",
        "### Visibility (percentile ranks, 0-100)",
        "| Column | Description |",
        "|--------|-------------|",
        "| `pctile_chs` | Survey visibility percentile |",
        "| `pctile_sparcs` | Hospital visibility percentile |",
        "| `pctile_vital` | Vital records percentile |",
        "| `pctile_visibility_range` | Cross-system divergence (higher = more discrepancy) |",
        "",
        "### Typology",
        "| Column | Description |",
        "|--------|-------------|",
        "| `cluster_id` | Numeric cluster ID |",
        "| `typology_label` | Human-readable typology name |",
        "| `stability_score` | Bootstrap stability (0-1) |",
        "",
        "### Vulnerability",
        "| Column | Description |",
        "|--------|-------------|",
        "| `consequence_score` | Health burden (0-100) |",
        "| `invisibility_score` | Survey underrepresentation (0-100) |",
        "| `vulnerability_score` | Combined vulnerability (0-100) |",
        "| `vulnerability_category` | low/medium/high |",
        "",
        "## Key Findings",
        "",
    ]
    
    # Add visibility summary
    lines.append("### Visibility by Source")
    lines.append("")
    lines.append("| Source | Mean | Median | Range |")
    lines.append("|--------|------|--------|-------|")
    for source, s in stats.get("visibility", {}).items():
        if s.get("mean") is not None:
            lines.append(f"| {source.upper()} | {s['mean']:.2f} | {s['median']:.2f} | {s['min']:.2f} - {s['max']:.2f} |")
    lines.append("")
    
    # Add correlations
    if stats.get("correlations"):
        lines.append("### Cross-Source Correlations")
        lines.append("")
        lines.append("| Pair | Correlation |")
        lines.append("|------|-------------|")
        for pair, r in stats["correlations"].items():
            lines.append(f"| {pair.replace('vis_', '').replace('_', ' ')} | {r:.3f} |")
        lines.append("")
    
    # Add vulnerability summary
    if stats.get("vulnerability", {}).get("distribution"):
        lines.append("### Vulnerability Distribution")
        lines.append("")
        for cat, count in stats["vulnerability"]["distribution"].items():
            lines.append(f"- **{cat.capitalize()}:** {count} neighborhoods")
        lines.append("")
    
    lines.extend([
        "## Usage",
        "",
        "### In Python",
        "```python",
        "import geopandas as gpd",
        "atlas = gpd.read_file('nta_visibility_atlas.geojson')",
        "atlas.plot(column='vulnerability_score', cmap='OrRd', legend=True)",
        "```",
        "",
        "### In QGIS/ArcGIS",
        "Open `nta_visibility_atlas.geojson` directly.",
        "",
        "## Citation",
        "",
        "NYC Visibility Atlas v1.0",
        "Data sources: NYC Community Health Survey, SPARCS, Vital Statistics",
    ])
    
    atomic_write_text(output_path, "\n".join(lines))
    logger.info(f"Wrote README to {output_path}")


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
        ntas = read_geoparquet(paths.processed_geo / "nta_canonical.parquet")
        visibility = read_parquet(paths.processed_visibility / "visibility_long.parquet")
        pctrank = read_parquet(paths.processed_matrix / "visibility_matrix_pctrank.parquet")
        typologies = read_parquet(paths.processed_typologies / "typology_assignments.parquet")
        vulnerability = read_parquet(paths.processed_vulnerability / "vulnerability_scores.parquet")
        
        logger.info(f"Loaded NTAs: {len(ntas)}")
        logger.info(f"Loaded visibility: {len(visibility)}")
        logger.info(f"Loaded typologies: {len(typologies)}")
        logger.info(f"Loaded vulnerability: {len(vulnerability)}")
        
        # Build atlas
        atlas = build_atlas_gdf(ntas, visibility, pctrank, typologies, vulnerability, logger)
        
        # Compute summary statistics
        stats = compute_summary_statistics(atlas, logger)
        
        # Write outputs
        output_dir = ensure_dir(paths.data_final)
        
        # GeoJSON (main output)
        geojson_path = output_dir / "nta_visibility_atlas.geojson"
        atomic_write_geojson(geojson_path, atlas)
        log_output_written(logger, geojson_path, row_count=len(atlas))
        
        # CSV (no geometry)
        csv_df = atlas.drop(columns=['geometry'])
        csv_path = output_dir / "nta_visibility_atlas.csv"
        csv_df.to_csv(csv_path, index=False)
        log_output_written(logger, csv_path, row_count=len(csv_df))
        
        # Summary statistics JSON
        stats_path = output_dir / "summary_statistics.json"
        atomic_write_json(stats_path, stats)
        log_output_written(logger, stats_path, row_count=1)
        
        # README
        readme_path = output_dir / "README.md"
        create_atlas_readme(readme_path, stats, logger)
        
        # Write metadata sidecar
        write_metadata_sidecar(
            geojson_path,
            run_id,
            parameters={
                "n_neighborhoods": len(atlas),
                "columns": list(atlas.columns),
            },
            row_count=len(atlas),
        )
        
        # QA checks
        log_qa_check(logger, "atlas_completeness", 
                    atlas['geo_id'].notna().all(),
                    f"All {len(atlas)} neighborhoods have geo_id")
        
        vis_cols = [c for c in atlas.columns if c.startswith('vis_')]
        coverage = {c: atlas[c].notna().sum() for c in vis_cols}
        log_qa_check(logger, "visibility_coverage", True, str(coverage))
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Atlas neighborhoods: {len(atlas)}")
        logger.info(f"   Atlas columns: {len(atlas.columns)}")
        logger.info(f"   Output directory: {output_dir}")
        logger.info("=" * 60)
        logger.info("FILES CREATED:")
        logger.info(f"   📍 {geojson_path}")
        logger.info(f"   📊 {csv_path}")
        logger.info(f"   📈 {stats_path}")
        logger.info(f"   📄 {readme_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

