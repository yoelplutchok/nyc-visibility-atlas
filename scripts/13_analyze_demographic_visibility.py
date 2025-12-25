#!/usr/bin/env python3
"""
13_analyze_demographic_visibility.py

Analyze CHS visibility by demographic strata (race, age, poverty).

Pipeline Step: 13 (Enhancement)

This script:
1. Loads demographic CHS exports
2. Computes visibility (n_eff) by demographic group
3. Identifies which groups are most invisible (highest suppression rates)
4. Creates demographic visibility gap analysis

Inputs:
    - data/raw/chs/epiquery_exports/demographics/*.csv

Outputs:
    - data/processed/visibility/demographic_visibility.parquet
    - reports/tables/demographic_visibility_gaps.md
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
    atomic_write_parquet, atomic_write_text
)
from visibility_atlas.hashing import write_metadata_sidecar


SCRIPT_NAME = "13_analyze_demographic_visibility"


def load_demographic_files(
    demographics_dir: Path,
    logger: logging.Logger
) -> dict:
    """Load all demographic CSV files."""
    log_step_start(logger, "load_demographic_files")
    
    data = {
        'by_race': [],
        'by_age': [],
        'by_poverty': [],
        'by_sex': [],
        'by_neighborhood_race': [],  # Cross-tabulation
    }
    
    for csv_file in demographics_dir.glob("*.csv"):
        filename = csv_file.name.lower()
        
        try:
            # Try different encodings
            df = None
            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                for sep in ['\t', ',']:
                    try:
                        df = pd.read_csv(csv_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 3:
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 3:
                    break
            
            if df is None:
                logger.warning(f"Could not parse {csv_file.name}")
                continue
            
            # Normalize column names
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            # Extract indicator from filename
            if 'diabetes' in filename:
                df['indicator'] = 'diabetes'
            elif 'fair_poor_health' in filename or 'health' in filename:
                df['indicator'] = 'fair_poor_health'
            elif 'uninsured' in filename:
                df['indicator'] = 'uninsured'
            else:
                df['indicator'] = 'unknown'
            
            df['source_file'] = csv_file.name
            
            # Categorize by type
            if 'neighb' in filename and 'race' in filename:
                data['by_neighborhood_race'].append(df)
                logger.info(f"Loaded neighborhood×race: {csv_file.name} ({len(df)} rows)")
            elif 'race' in filename:
                data['by_race'].append(df)
                logger.info(f"Loaded by race: {csv_file.name} ({len(df)} rows)")
            elif 'age' in filename:
                data['by_age'].append(df)
                logger.info(f"Loaded by age: {csv_file.name} ({len(df)} rows)")
            elif 'poverty' in filename:
                data['by_poverty'].append(df)
                logger.info(f"Loaded by poverty: {csv_file.name} ({len(df)} rows)")
            elif 'sex' in filename:
                data['by_sex'].append(df)
                logger.info(f"Loaded by sex: {csv_file.name} ({len(df)} rows)")
            else:
                logger.warning(f"Unknown file type: {csv_file.name}")
                
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")
    
    # Combine each category
    combined = {}
    for key, dfs in data.items():
        if dfs:
            combined[key] = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {key}: {len(combined[key])} total rows")
        else:
            combined[key] = pd.DataFrame()
    
    log_step_end(logger, "load_demographic_files")
    return combined


def compute_demographic_visibility(
    df: pd.DataFrame,
    demographic_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Compute visibility metrics for a demographic breakdown."""
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Filter for 2019 and 'Yes' response
    if 'yearnum' in df.columns:
        df = df[df['yearnum'] == 2019].copy()
    
    if 'response' in df.columns:
        df = df[df['response'].isin(['Yes', 'Fair or Poor'])].copy()
    
    # Find the demographic column
    demo_col = None
    for col in df.columns:
        if 'dimension_response' in col or demographic_col.lower() in col.lower():
            demo_col = col
            break
    
    if demo_col is None:
        # Try dim2value for cross-tabulations
        if 'dim2value' in df.columns:
            demo_col = 'dim2value'
        else:
            return pd.DataFrame()
    
    # Extract prevalence and CIs
    prev_col = None
    ci_low_col = None
    ci_high_col = None
    
    for col in df.columns:
        if 'prevalence' in col.lower() or 'percent' in col.lower():
            prev_col = col
        if 'lower' in col.lower() and 'confidence' in col.lower():
            ci_low_col = col
        if 'upper' in col.lower() and 'confidence' in col.lower():
            ci_high_col = col
    
    if prev_col is None:
        return pd.DataFrame()
    
    # Clean and compute
    results = []
    
    for indicator in df['indicator'].unique():
        ind_df = df[df['indicator'] == indicator].copy()
        
        for demo_value in ind_df[demo_col].unique():
            demo_df = ind_df[ind_df[demo_col] == demo_value].copy()
            
            if len(demo_df) == 0:
                continue
            
            # Get the row (should be one per indicator×demo)
            row = demo_df.iloc[0]
            
            # Check for suppression
            interpretation_col = None
            for col in demo_df.columns:
                if 'interpretation' in col.lower() or 'flag' in col.lower():
                    interpretation_col = col
                    break
            
            is_suppressed = False
            if interpretation_col and pd.notna(row.get(interpretation_col)):
                flag = str(row[interpretation_col]).lower()
                is_suppressed = 'suppress' in flag or 'unreliable' in flag
            
            # Parse prevalence
            prev_str = str(row.get(prev_col, ''))
            prev_val = pd.to_numeric(prev_str.replace('%', ''), errors='coerce')
            
            # Parse CIs
            ci_low = None
            ci_high = None
            if ci_low_col and ci_high_col:
                ci_low_str = str(row.get(ci_low_col, ''))
                ci_high_str = str(row.get(ci_high_col, ''))
                ci_low = pd.to_numeric(ci_low_str.replace('%', ''), errors='coerce')
                ci_high = pd.to_numeric(ci_high_str.replace('%', ''), errors='coerce')
            
            # Compute n_eff (effective sample size)
            n_eff = np.nan
            if ci_low is not None and ci_high is not None and not np.isnan(ci_low) and not np.isnan(ci_high):
                # SE = (CI_high - CI_low) / (2 * 1.96)
                se = (ci_high - ci_low) / (2 * 1.96)
                if se > 0 and prev_val is not None and not np.isnan(prev_val):
                    p = prev_val / 100
                    # n_eff ≈ p(1-p) / SE²
                    if 0 < p < 1:
                        n_eff = (p * (1 - p)) / ((se / 100) ** 2)
            
            # Get neighborhood if cross-tab
            neighborhood = None
            if 'dimension_response' in df.columns and demographic_col == 'race':
                neighborhood = row.get('dimension_response')
            
            results.append({
                'indicator': indicator,
                'demographic_type': demographic_col,
                'demographic_value': str(demo_value).strip(),
                'neighborhood': neighborhood,
                'prevalence': prev_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_eff': n_eff,
                'is_suppressed': is_suppressed,
                'reliability': 'suppressed' if is_suppressed else ('low' if (n_eff and n_eff < 50) else 'high'),
            })
    
    return pd.DataFrame(results)


def analyze_visibility_gaps(
    visibility_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Analyze visibility gaps across demographic groups."""
    
    if len(visibility_df) == 0:
        return pd.DataFrame()
    
    # Group by demographic type and value
    gaps = visibility_df.groupby(['demographic_type', 'demographic_value']).agg({
        'n_eff': ['mean', 'median', 'min', 'max', 'count'],
        'is_suppressed': ['sum', 'mean'],  # Count and rate of suppression
        'prevalence': 'mean',
    }).reset_index()
    
    # Flatten column names
    gaps.columns = [
        'demographic_type', 'demographic_value',
        'mean_n_eff', 'median_n_eff', 'min_n_eff', 'max_n_eff', 'n_estimates',
        'n_suppressed', 'suppression_rate', 'mean_prevalence'
    ]
    
    # Sort by suppression rate (highest = most invisible)
    gaps = gaps.sort_values('suppression_rate', ascending=False)
    
    return gaps


def create_demographic_report(
    visibility_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
):
    """Create markdown report of demographic visibility gaps."""
    
    lines = [
        "# Demographic Visibility Gap Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        "This analysis examines which demographic groups are most 'invisible' to the NYC Community Health Survey.",
        "",
        "**Key metrics:**",
        "- **n_eff (effective sample size)**: Higher = more survey respondents = better visibility",
        "- **Suppression rate**: Percentage of estimates marked as unreliable/suppressed",
        "",
        "## Visibility by Demographic Group",
        "",
    ]
    
    # By demographic type
    for demo_type in gaps_df['demographic_type'].unique():
        type_df = gaps_df[gaps_df['demographic_type'] == demo_type].copy()
        
        lines.append(f"### By {demo_type.title()}")
        lines.append("")
        lines.append("| Group | Mean n_eff | Suppression Rate | N Estimates |")
        lines.append("|-------|------------|------------------|-------------|")
        
        for _, row in type_df.iterrows():
            n_eff_str = f"{row['mean_n_eff']:.0f}" if pd.notna(row['mean_n_eff']) else "N/A"
            supp_str = f"{row['suppression_rate']*100:.1f}%"
            lines.append(f"| {row['demographic_value']} | {n_eff_str} | {supp_str} | {row['n_estimates']:.0f} |")
        
        lines.append("")
    
    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])
    
    # Most invisible groups
    most_invisible = gaps_df.nlargest(5, 'suppression_rate')
    if len(most_invisible) > 0:
        lines.append("### Most Invisible Groups (Highest Suppression)")
        lines.append("")
        for _, row in most_invisible.iterrows():
            lines.append(f"- **{row['demographic_value']}** ({row['demographic_type']}): {row['suppression_rate']*100:.1f}% suppressed")
        lines.append("")
    
    # Most visible groups
    most_visible = gaps_df.nsmallest(5, 'suppression_rate')
    if len(most_visible) > 0:
        lines.append("### Most Visible Groups (Lowest Suppression)")
        lines.append("")
        for _, row in most_visible.iterrows():
            lines.append(f"- **{row['demographic_value']}** ({row['demographic_type']}): {row['suppression_rate']*100:.1f}% suppressed")
        lines.append("")
    
    lines.extend([
        "## Interpretation",
        "",
        "Groups with high suppression rates are **underrepresented** in the survey.",
        "This may indicate:",
        "- Smaller population size",
        "- Lower survey response rates",
        "- Harder to reach populations",
        "",
        "**Policy implication:** These groups may be systematically missed by survey-based health monitoring.",
    ])
    
    atomic_write_text(output_path, "\n".join(lines))
    logger.info(f"Wrote demographic report to {output_path}")


def main():
    """Main entry point."""
    run_id = get_run_id()
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("=" * 60)
    logger.info(f"Starting {SCRIPT_NAME}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Load demographic files
        demographics_dir = paths.raw_chs / "epiquery_exports" / "demographics"
        data = load_demographic_files(demographics_dir, logger)
        
        # Process each demographic type
        all_visibility = []
        
        # By race
        if len(data.get('by_race', pd.DataFrame())) > 0:
            race_vis = compute_demographic_visibility(data['by_race'], 'race', logger)
            if len(race_vis) > 0:
                all_visibility.append(race_vis)
                logger.info(f"Computed race visibility: {len(race_vis)} records")
        
        # By age
        if len(data.get('by_age', pd.DataFrame())) > 0:
            age_vis = compute_demographic_visibility(data['by_age'], 'age', logger)
            if len(age_vis) > 0:
                all_visibility.append(age_vis)
                logger.info(f"Computed age visibility: {len(age_vis)} records")
        
        # By poverty
        if len(data.get('by_poverty', pd.DataFrame())) > 0:
            poverty_vis = compute_demographic_visibility(data['by_poverty'], 'poverty', logger)
            if len(poverty_vis) > 0:
                all_visibility.append(poverty_vis)
                logger.info(f"Computed poverty visibility: {len(poverty_vis)} records")
        
        # By sex
        if len(data.get('by_sex', pd.DataFrame())) > 0:
            sex_vis = compute_demographic_visibility(data['by_sex'], 'sex', logger)
            if len(sex_vis) > 0:
                all_visibility.append(sex_vis)
                logger.info(f"Computed sex visibility: {len(sex_vis)} records")
        
        # Neighborhood × Race cross-tabulation
        if len(data.get('by_neighborhood_race', pd.DataFrame())) > 0:
            neighb_race_vis = compute_demographic_visibility(
                data['by_neighborhood_race'], 'neighborhood_race', logger
            )
            if len(neighb_race_vis) > 0:
                all_visibility.append(neighb_race_vis)
                logger.info(f"Computed neighborhood×race visibility: {len(neighb_race_vis)} records")
        
        if not all_visibility:
            logger.error("No visibility data computed")
            return 1
        
        # Combine all
        visibility_df = pd.concat(all_visibility, ignore_index=True)
        logger.info(f"Total visibility records: {len(visibility_df)}")
        
        # Analyze gaps
        gaps_df = analyze_visibility_gaps(visibility_df, logger)
        
        # Write outputs
        vis_path = paths.processed_visibility / "demographic_visibility.parquet"
        atomic_write_parquet(vis_path, visibility_df)
        log_output_written(logger, vis_path, row_count=len(visibility_df))
        
        gaps_path = paths.processed_visibility / "demographic_gaps.parquet"
        atomic_write_parquet(gaps_path, gaps_df)
        log_output_written(logger, gaps_path, row_count=len(gaps_df))
        
        # Create report
        reports_dir = ensure_dir(paths.reports / "tables")
        report_path = reports_dir / "demographic_visibility_gaps.md"
        create_demographic_report(visibility_df, gaps_df, report_path, logger)
        
        # Write metadata
        write_metadata_sidecar(
            vis_path,
            run_id,
            parameters={
                "demographic_types": visibility_df['demographic_type'].unique().tolist(),
                "n_records": len(visibility_df),
            },
            row_count=len(visibility_df),
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"✅ {SCRIPT_NAME} completed successfully")
        logger.info(f"   Total records: {len(visibility_df)}")
        logger.info(f"   Demographic types: {visibility_df['demographic_type'].unique().tolist()}")
        logger.info("=" * 60)
        
        # Key findings
        logger.info("KEY FINDINGS:")
        for demo_type in gaps_df['demographic_type'].unique():
            type_gaps = gaps_df[gaps_df['demographic_type'] == demo_type]
            highest = type_gaps.nlargest(1, 'suppression_rate')
            lowest = type_gaps.nsmallest(1, 'suppression_rate')
            if len(highest) > 0:
                h = highest.iloc[0]
                logger.info(f"   {demo_type}: Most invisible = {h['demographic_value']} ({h['suppression_rate']*100:.1f}% suppressed)")
            if len(lowest) > 0:
                l = lowest.iloc[0]
                logger.info(f"   {demo_type}: Most visible = {l['demographic_value']} ({l['suppression_rate']*100:.1f}% suppressed)")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

