"""
Canonical root detection and path resolution.

This module implements guardrail R1: All scripts use visibility_atlas.paths
to resolve paths. No relative ../ anywhere.

The .project-root file marks the repository root.
"""

from pathlib import Path
from typing import Union

# Cached project root
_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """
    Find and return the project root directory.
    
    Searches upward from this file's location for .project-root marker.
    Result is cached for performance.
    
    Returns:
        Path to the project root directory.
        
    Raises:
        FileNotFoundError: If .project-root marker is not found.
    """
    global _PROJECT_ROOT
    
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    
    # Start from this file's directory and search upward
    current = Path(__file__).resolve().parent
    
    for _ in range(10):  # Limit search depth
        marker = current / ".project-root"
        if marker.exists():
            _PROJECT_ROOT = current
            return _PROJECT_ROOT
        
        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent
    
    raise FileNotFoundError(
        "Could not find .project-root marker. "
        "Ensure you are running from within the NYC Visibility Atlas repository."
    )


def get_path(*parts: str) -> Path:
    """
    Resolve a path relative to the project root.
    
    Args:
        *parts: Path components to join (e.g., "data", "raw", "acs")
        
    Returns:
        Absolute Path object.
        
    Example:
        >>> get_path("data", "processed", "geo")
        PosixPath('/path/to/project/data/processed/geo')
    """
    return get_project_root() / Path(*parts)


# =============================================================================
# Canonical path constants
# =============================================================================

class Paths:
    """
    Canonical path constants for the project.
    
    All paths are resolved relative to the project root.
    Use these constants instead of hardcoding paths.
    """
    
    @property
    def root(self) -> Path:
        """Project root directory."""
        return get_project_root()
    
    # -------------------------------------------------------------------------
    # Config paths
    # -------------------------------------------------------------------------
    @property
    def configs(self) -> Path:
        return get_path("configs")
    
    @property
    def params_yml(self) -> Path:
        return get_path("configs", "params.yml")
    
    @property
    def strata_yml(self) -> Path:
        return get_path("configs", "strata.yml")
    
    @property
    def sources_yml(self) -> Path:
        return get_path("configs", "sources.yml")
    
    @property
    def atlas_yml(self) -> Path:
        return get_path("configs", "atlas.yml")
    
    @property
    def data_inventory_yml(self) -> Path:
        return get_path("configs", "data_inventory.yml")
    
    # -------------------------------------------------------------------------
    # Data paths - Raw
    # -------------------------------------------------------------------------
    @property
    def data_raw(self) -> Path:
        return get_path("data", "raw")
    
    @property
    def raw_manifest(self) -> Path:
        return get_path("data", "raw", "_manifest.json")
    
    @property
    def raw_acs(self) -> Path:
        return get_path("data", "raw", "acs")
    
    @property
    def raw_chs(self) -> Path:
        return get_path("data", "raw", "chs")
    
    @property
    def raw_sparcs(self) -> Path:
        return get_path("data", "raw", "sparcs")
    
    @property
    def raw_enrollment(self) -> Path:
        return get_path("data", "raw", "enrollment")
    
    @property
    def raw_vital(self) -> Path:
        return get_path("data", "raw", "vital")
    
    @property
    def raw_311(self) -> Path:
        return get_path("data", "raw", "311")
    
    @property
    def raw_geo(self) -> Path:
        return get_path("data", "raw", "geo")
    
    # -------------------------------------------------------------------------
    # Data paths - Interim
    # -------------------------------------------------------------------------
    @property
    def data_interim(self) -> Path:
        return get_path("data", "interim")
    
    # -------------------------------------------------------------------------
    # Data paths - Processed
    # -------------------------------------------------------------------------
    @property
    def data_processed(self) -> Path:
        return get_path("data", "processed")
    
    @property
    def processed_geo(self) -> Path:
        return get_path("data", "processed", "geo")
    
    @property
    def processed_xwalk(self) -> Path:
        return get_path("data", "processed", "xwalk")
    
    @property
    def processed_denominators(self) -> Path:
        return get_path("data", "processed", "denominators")
    
    @property
    def processed_numerators(self) -> Path:
        return get_path("data", "processed", "numerators")
    
    @property
    def processed_visibility(self) -> Path:
        return get_path("data", "processed", "visibility")
    
    @property
    def processed_matrix(self) -> Path:
        return get_path("data", "processed", "matrix")
    
    @property
    def processed_typologies(self) -> Path:
        return get_path("data", "processed", "typologies")
    
    @property
    def processed_models(self) -> Path:
        return get_path("data", "processed", "models")
    
    @property
    def processed_vulnerability(self) -> Path:
        return get_path("data", "processed", "vulnerability")
    
    @property
    def processed_atlas_layers(self) -> Path:
        return get_path("data", "processed", "atlas_layers")
    
    @property
    def processed_metadata(self) -> Path:
        return get_path("data", "processed", "metadata")
    
    @property
    def data_final(self) -> Path:
        return get_path("data", "final")
    
    # -------------------------------------------------------------------------
    # Logs
    # -------------------------------------------------------------------------
    @property
    def logs(self) -> Path:
        return get_path("logs")
    
    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------
    @property
    def reports(self) -> Path:
        return get_path("reports")
    
    @property
    def reports_figures(self) -> Path:
        return get_path("reports", "figures")
    
    @property
    def reports_tables(self) -> Path:
        return get_path("reports", "tables")
    
    @property
    def reports_atlas(self) -> Path:
        return get_path("reports", "atlas")
    
    # -------------------------------------------------------------------------
    # Docs
    # -------------------------------------------------------------------------
    @property
    def docs(self) -> Path:
        return get_path("docs")


# Singleton instance for convenience
paths = Paths()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists.
        
    Returns:
        The Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

