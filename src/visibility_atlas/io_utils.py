"""
Atomic writes and I/O helper utilities.

This module implements guardrail R2: Atomic writes only.
Write to temp file → rename/replace. .tmp artifacts are treated as failures.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from visibility_atlas.paths import ensure_dir


def atomic_write(
    target_path: Path | str,
    write_func: callable,
    *args,
    **kwargs
) -> Path:
    """
    Write to a file atomically using a temporary file and rename.
    
    This ensures that the target file is never in a partially-written state.
    If the write fails, the target file is unchanged.
    
    Args:
        target_path: Final destination path.
        write_func: Function that writes to a file. Called as write_func(temp_path, *args, **kwargs).
        *args: Additional arguments to pass to write_func.
        **kwargs: Additional keyword arguments to pass to write_func.
        
    Returns:
        The target path (as Path object).
        
    Raises:
        Exception: Re-raises any exception from write_func after cleanup.
    """
    target_path = Path(target_path)
    ensure_dir(target_path.parent)
    
    # Create temp file in the same directory to ensure same filesystem
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{target_path.stem}_",
        dir=target_path.parent
    )
    temp_path = Path(temp_path)
    
    try:
        # Close the file descriptor (we'll reopen in write_func)
        os.close(temp_fd)
        
        # Write to temp file
        write_func(temp_path, *args, **kwargs)
        
        # Atomic rename
        temp_path.replace(target_path)
        
        return target_path
        
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(target_path: Path | str, data: Any, indent: int = 2) -> Path:
    """
    Write JSON data to a file atomically.
    
    Args:
        target_path: Destination file path.
        data: Data to serialize as JSON.
        indent: JSON indentation level.
        
    Returns:
        The target path.
    """
    def write_json(temp_path: Path, data: Any, indent: int):
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
    
    return atomic_write(target_path, write_json, data, indent)


def atomic_write_text(target_path: Path | str, content: str) -> Path:
    """
    Write text content to a file atomically.
    
    Args:
        target_path: Destination file path.
        content: Text content to write.
        
    Returns:
        The target path.
    """
    def write_text(temp_path: Path, content: str):
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return atomic_write(target_path, write_text, content)


def atomic_write_parquet(
    target_path: Path | str,
    df: "pd.DataFrame",
    **kwargs
) -> Path:
    """
    Write a DataFrame to Parquet atomically.
    
    Args:
        target_path: Destination file path.
        df: DataFrame to write.
        **kwargs: Additional arguments to pass to pyarrow.parquet.write_table.
        
    Returns:
        The target path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    def write_parquet(temp_path: Path, df: "pd.DataFrame", **kwargs):
        table = pa.Table.from_pandas(df)
        pq.write_table(table, temp_path, **kwargs)
    
    return atomic_write(target_path, write_parquet, df, **kwargs)


def atomic_write_geojson(
    target_path: Path | str,
    gdf: "pd.DataFrame",  # Actually GeoDataFrame
) -> Path:
    """
    Write a GeoDataFrame to GeoJSON atomically.
    
    Args:
        target_path: Destination file path.
        gdf: GeoDataFrame to write.
        
    Returns:
        The target path.
    """
    def write_geojson(temp_path: Path, gdf):
        gdf.to_file(temp_path, driver="GeoJSON")
    
    return atomic_write(target_path, write_geojson, gdf)


def atomic_write_geoparquet(
    target_path: Path | str,
    gdf: "pd.DataFrame",  # Actually GeoDataFrame
) -> Path:
    """
    Write a GeoDataFrame to GeoParquet atomically.
    
    Args:
        target_path: Destination file path.
        gdf: GeoDataFrame to write.
        
    Returns:
        The target path.
    """
    def write_geoparquet(temp_path: Path, gdf):
        gdf.to_parquet(temp_path)
    
    return atomic_write(target_path, write_geoparquet, gdf)


# =============================================================================
# Read utilities
# =============================================================================

def read_parquet(file_path: Path | str) -> "pd.DataFrame":
    """
    Read a Parquet file into a DataFrame.
    
    Args:
        file_path: Path to the Parquet file.
        
    Returns:
        DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import pandas as pd
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return pd.read_parquet(file_path)


def read_geoparquet(file_path: Path | str) -> "pd.DataFrame":
    """
    Read a GeoParquet file into a GeoDataFrame.
    
    Args:
        file_path: Path to the GeoParquet file.
        
    Returns:
        GeoDataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import geopandas as gpd
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoParquet file not found: {file_path}")
    return gpd.read_parquet(file_path)


def read_json(file_path: Path | str) -> Any:
    """
    Read a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Parsed JSON data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(file_path: Path | str) -> Any:
    """
    Read a YAML file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Parsed YAML data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to read YAML files. Install with: pip install pyyaml")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Cleanup utilities
# =============================================================================

def clean_tmp_files(directory: Path | str, pattern: str = "*.tmp") -> list[Path]:
    """
    Remove .tmp files from a directory (failed atomic writes).
    
    Args:
        directory: Directory to clean.
        pattern: Glob pattern for temp files.
        
    Returns:
        List of removed file paths.
    """
    directory = Path(directory)
    removed = []
    
    if directory.exists():
        for tmp_file in directory.glob(pattern):
            tmp_file.unlink()
            removed.append(tmp_file)
    
    return removed

