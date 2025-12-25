"""
Command-line interface entry points for pipeline scripts.

This module provides entry points for the pyproject.toml [project.scripts] section,
allowing users to run pipeline steps via installed commands:

    visibility-atlas-geographies    # Run step 00
    visibility-atlas-crosswalks     # Run step 01
    visibility-atlas-run-all        # Run full pipeline
    
These are wrappers around the scripts/ directory files.
"""

import subprocess
import sys
from pathlib import Path

from visibility_atlas.paths import get_project_root


def _run_script(script_name: str) -> int:
    """Run a pipeline script and return its exit code."""
    scripts_dir = get_project_root() / "scripts"
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1
    
    result = subprocess.run([sys.executable, str(script_path)], cwd=get_project_root())
    return result.returncode


def run_00_geographies() -> int:
    """Run step 00: Build canonical geography files."""
    return _run_script("00_build_geographies.py")


def run_01_crosswalks() -> int:
    """Run step 01: Build population-weighted crosswalks."""
    return _run_script("01_build_crosswalks.py")


def run_02_denominators() -> int:
    """Run step 02: Build ACS denominators."""
    return _run_script("02_build_denominators_acs.py")


def run_03_chs() -> int:
    """Run step 03: Build CHS numerator."""
    return _run_script("03_build_numerator_chs.py")


def run_04_sparcs() -> int:
    """Run step 04b: Build SPARCS encounters numerator (canonical)."""
    return _run_script("04b_build_numerator_sparcs_encounters.py")


def run_06_vital() -> int:
    """Run step 06: Build vital statistics numerator."""
    return _run_script("06_build_numerator_vital.py")


def run_07_visibility() -> int:
    """Run step 07: Build unified visibility tables."""
    return _run_script("07_build_visibility_tables.py")


def run_08_matrix() -> int:
    """Run step 08: Build cross-source matrix."""
    return _run_script("08_build_cross_source_matrix.py")


def run_09_typology() -> int:
    """Run step 09: Run typology clustering."""
    return _run_script("09_typology_clustering.py")


def run_10_predictors() -> int:
    """Run step 10: Run predictor models."""
    return _run_script("10_predictors_and_spatial_diagnostics.py")


def run_11_vulnerability() -> int:
    """Run step 11: Compute estimate vulnerability."""
    return _run_script("11_consequence_vulnerability.py")


def run_12_atlas() -> int:
    """Run step 12: Build atlas assets."""
    return _run_script("12_build_atlas_assets.py")


def run_all() -> int:
    """
    Run the full pipeline in order.
    
    Returns the first non-zero exit code, or 0 if all succeed.
    """
    steps = [
        ("00_build_geographies.py", "Building geographies"),
        ("01_build_crosswalks.py", "Building crosswalks"),
        ("02_build_denominators_acs.py", "Building ACS denominators"),
        ("02b_build_uhf_nta_crosswalk.py", "Building UHF→NTA crosswalk"),
        ("03_build_numerator_chs.py", "Building CHS numerator"),
        ("04b_build_numerator_sparcs_encounters.py", "Building SPARCS numerator"),
        ("06_build_numerator_vital.py", "Building vital numerator"),
        ("07_build_visibility_tables.py", "Building visibility tables"),
        ("08_build_cross_source_matrix.py", "Building cross-source matrix"),
        ("09_typology_clustering.py", "Running typology clustering"),
        ("10_predictors_and_spatial_diagnostics.py", "Running predictor models"),
        ("11_consequence_vulnerability.py", "Computing vulnerability"),
        ("12_build_atlas_assets.py", "Building atlas assets"),
    ]
    
    print("=" * 60)
    print("NYC Visibility Atlas - Full Pipeline")
    print("=" * 60)
    
    for script_name, description in steps:
        print(f"\n[{description}]")
        print("-" * 40)
        
        exit_code = _run_script(script_name)
        
        if exit_code != 0:
            print(f"\n❌ Pipeline failed at: {script_name}")
            return exit_code
    
    print("\n" + "=" * 60)
    print("✅ Full pipeline completed successfully!")
    print("=" * 60)
    
    return 0


# For direct execution
if __name__ == "__main__":
    sys.exit(run_all())

