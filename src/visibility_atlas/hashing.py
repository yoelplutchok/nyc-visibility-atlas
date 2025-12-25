"""
File, config, and code hashing utilities for reproducibility.

This module implements guardrail R4: Hash-aware caching.
Each output gets a sidecar metadata JSON with input file hashes,
config digest, code version, library versions, runtime timestamp, and run_id.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visibility_atlas.paths import get_project_root, paths, ensure_dir


def hash_file(file_path: Path | str, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a file.
    
    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm (default: sha256).
        
    Returns:
        Hexadecimal hash string.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")
    
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def hash_files(file_paths: list[Path | str], algorithm: str = "sha256") -> dict[str, str]:
    """
    Compute hashes for multiple files.
    
    Args:
        file_paths: List of file paths.
        algorithm: Hash algorithm.
        
    Returns:
        Dictionary mapping file paths (as strings) to their hashes.
    """
    return {str(p): hash_file(p, algorithm) for p in file_paths}


def hash_string(content: str, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a string.
    
    Args:
        content: String content to hash.
        algorithm: Hash algorithm.
        
    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def hash_dict(data: dict, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a dictionary (JSON-serialized, sorted keys).
    
    Args:
        data: Dictionary to hash.
        algorithm: Hash algorithm.
        
    Returns:
        Hexadecimal hash string.
    """
    # Serialize with sorted keys for determinism
    content = json.dumps(data, sort_keys=True, default=str)
    return hash_string(content, algorithm)


def hash_config(config_path: Path | str) -> str:
    """
    Compute the hash of a YAML or JSON config file.
    
    The config is parsed and re-serialized with sorted keys
    to ensure the hash is content-based, not format-based.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        Hexadecimal hash string.
    """
    config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix in (".yml", ".yaml"):
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                # Fall back to raw file hash if yaml not installed
                return hash_file(config_path)
        elif config_path.suffix == ".json":
            data = json.load(f)
        else:
            # Fall back to raw file hash
            return hash_file(config_path)
    
    return hash_dict(data)


def get_git_commit() -> str | None:
    """
    Get the current git commit hash if available.
    
    Returns:
        Short commit hash (8 chars) or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_dirty() -> bool | None:
    """
    Check if the git working directory has uncommitted changes.
    
    Returns:
        True if dirty, False if clean, None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_code_version() -> dict[str, Any]:
    """
    Get code version information including git commit.
    
    Returns:
        Dictionary with version information.
    """
    return {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
    }


def get_library_versions() -> dict[str, str]:
    """
    Get versions of key libraries.
    
    Returns:
        Dictionary mapping library names to version strings.
    """
    versions = {
        "python": sys.version.split()[0],
    }
    
    # Core libraries
    libraries = [
        "pandas",
        "geopandas", 
        "numpy",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "shapely",
        "pyproj",
    ]
    
    for lib in libraries:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    
    return versions


def create_metadata_sidecar(
    output_path: Path | str,
    run_id: str,
    input_files: list[Path | str] | None = None,
    config_files: list[Path | str] | None = None,
    parameters: dict[str, Any] | None = None,
    row_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a metadata sidecar dictionary for an output file.
    
    Args:
        output_path: Path to the output file.
        run_id: Unique run identifier.
        input_files: List of input file paths to hash.
        config_files: List of config file paths to hash.
        parameters: Dictionary of runtime parameters.
        row_count: Number of rows in the output (if applicable).
        extra: Additional metadata to include.
        
    Returns:
        Metadata dictionary ready to be written as JSON.
    """
    output_path = Path(output_path)
    
    metadata = {
        "output_file": str(output_path.name),
        "output_path": str(output_path),
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": get_code_version(),
        "library_versions": get_library_versions(),
    }
    
    # Hash input files
    if input_files:
        metadata["input_file_hashes"] = {}
        for f in input_files:
            f = Path(f)
            if f.exists():
                metadata["input_file_hashes"][str(f.name)] = hash_file(f)
    
    # Hash config files
    if config_files:
        metadata["config_hashes"] = {}
        for f in config_files:
            f = Path(f)
            if f.exists():
                metadata["config_hashes"][str(f.name)] = hash_config(f)
    
    # Add parameters
    if parameters:
        metadata["parameters"] = parameters
    
    # Add row count
    if row_count is not None:
        metadata["row_count"] = row_count
    
    # Hash the output file itself
    if output_path.exists():
        metadata["output_hash"] = hash_file(output_path)
    
    # Add extra metadata
    if extra:
        metadata["extra"] = extra
    
    return metadata


def write_metadata_sidecar(
    output_path: Path | str,
    run_id: str,
    input_files: list[Path | str] | None = None,
    config_files: list[Path | str] | None = None,
    parameters: dict[str, Any] | None = None,
    row_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Write a metadata sidecar JSON file for an output.
    
    The sidecar is written to the same directory as the output,
    with the same name but with "_metadata.json" suffix.
    
    Args:
        output_path: Path to the output file.
        run_id: Unique run identifier.
        input_files: List of input file paths to hash.
        config_files: List of config file paths to hash.
        parameters: Dictionary of runtime parameters.
        row_count: Number of rows in the output (if applicable).
        extra: Additional metadata to include.
        
    Returns:
        Path to the written metadata file.
    """
    output_path = Path(output_path)
    
    metadata = create_metadata_sidecar(
        output_path=output_path,
        run_id=run_id,
        input_files=input_files,
        config_files=config_files,
        parameters=parameters,
        row_count=row_count,
        extra=extra,
    )
    
    # Determine sidecar path
    sidecar_path = output_path.parent / f"{output_path.stem}_metadata.json"
    
    # Write atomically (import here to avoid circular import)
    from visibility_atlas.io_utils import atomic_write_json
    atomic_write_json(sidecar_path, metadata)
    
    return sidecar_path


def check_hashes_match(
    metadata_path: Path | str,
    input_files: list[Path | str] | None = None,
    config_files: list[Path | str] | None = None,
) -> bool:
    """
    Check if current file hashes match those in a metadata sidecar.
    
    Used to determine if an output needs to be regenerated.
    
    Args:
        metadata_path: Path to the metadata sidecar JSON.
        input_files: List of input file paths to check.
        config_files: List of config file paths to check.
        
    Returns:
        True if all hashes match, False otherwise.
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        return False
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Check input file hashes
    if input_files and "input_file_hashes" in metadata:
        stored_hashes = metadata["input_file_hashes"]
        for f in input_files:
            f = Path(f)
            if not f.exists():
                return False
            current_hash = hash_file(f)
            stored_hash = stored_hashes.get(f.name)
            if current_hash != stored_hash:
                return False
    
    # Check config file hashes
    if config_files and "config_hashes" in metadata:
        stored_hashes = metadata["config_hashes"]
        for f in config_files:
            f = Path(f)
            if not f.exists():
                return False
            current_hash = hash_config(f)
            stored_hash = stored_hashes.get(f.name)
            if current_hash != stored_hash:
                return False
    
    return True

