# NYC Visibility Atlas

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%20passed-brightgreen.svg)](#testing)

**A practical tool for understanding differential visibility across NYC public health data systems.**

> *"Different data systems see different people."* This project quantifies how much — and where.

## Overview

NYC health indicators depend on who each data system can "see." This project builds a **Visibility Atlas** that:

1. Quantifies how multiple public health data sources differentially capture or observe residents across neighborhoods and demographic strata
2. Identifies cross-system **blind spot typologies**
3. Evaluates where common health estimates are most vulnerable to misinterpretation due to **low or skewed visibility**

## Key Concepts

- **Ecological cell**: The unit of analysis is `neighborhood × demographic stratum × time window`
- **Visibility index**: Observed count per 1,000 residents for each data source
- **Coverage**: Population coverage ratio (only for sources with unique resident counts)
- **Reliability layer**: Every visibility value includes an uncertainty/reliability indicator

## Data Sources

The atlas integrates multiple data systems:

- **Survey measurement**: NYC Community Health Survey (CHS)
- **Healthcare utilization**: SPARCS hospitalization/ED data
- **Program enrollment**: Medicaid/Medicare aggregates
- **Vital events**: Birth and death records
- **Civic contact** (optional): 311 service requests

## Project Structure

```
nyc-visibility-atlas/
  .project-root           # Repository root marker
  configs/                # Configuration files (params, strata, sources)
  data/
    raw/                  # Raw data with provenance manifest
    interim/              # Intermediate processing
    processed/            # Final outputs by category
  logs/                   # JSONL structured logs
  src/visibility_atlas/   # Core Python package
  scripts/                # Pipeline scripts (00-12)
  tests/                  # Test suite
  notebooks/              # Read-only visualization notebooks
  docs/                   # Documentation
  reports/                # Generated reports and figures
```

## Setup

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nyc-visibility-atlas

# Create conda environment (development)
conda env create -f environment.yml
conda activate visibility-atlas

# OR for exact reproducibility (recommended)
conda env create -f environment.lock.yml
conda activate visibility-atlas

# Install package in development mode
pip install -e .
```

### Data Acquisition

Some datasets require manual download. See the complete guide:
- **`data/raw/ACQUISITION.md`** — Step-by-step instructions for all data sources

Quick summary:
| Dataset | Auto-download | Manual steps |
|---------|---------------|--------------|
| NTA Boundaries | ✅ Yes | None |
| ACS Populations | ✅ Yes | None |
| SPARCS Encounters | ✅ Yes | None |
| CHS Indicators | ❌ No | EpiQuery export |
| Vital Statistics | ❌ No | EpiQuery export |

## Usage

### Running the Pipeline

**Option 1: Make commands**
```bash
make all             # Run full pipeline
make geographies     # Step 00: Build canonical geographies
make crosswalks      # Step 01: Build crosswalks
make denominators    # Step 02: Build ACS denominators
make test            # Run tests
make test-smoke      # Run smoke tests only
```

**Option 2: CLI entry points** (after `pip install -e .`)
```bash
visibility-atlas-run-all       # Run full pipeline
visibility-atlas-geographies   # Step 00
visibility-atlas-crosswalks    # Step 01
visibility-atlas-chs           # Step 03
visibility-atlas-sparcs        # Step 04
# ... etc.
```

**Option 3: Direct script execution**
```bash
python scripts/00_build_geographies.py
python scripts/01_build_crosswalks.py
# ... etc.
```

### Configuration

All parameters are controlled via YAML files in `configs/`:

- `params.yml`: Geography, time windows, thresholds, small-number policy
- `strata.yml`: Demographic strata definitions and ACS table mappings
- `sources.yml`: Data source configurations and ingestion modes
- `atlas.yml`: Visualization settings

## Reproducibility

This project follows strict reproducibility guardrails:

- **Atomic writes**: All outputs written via temp file → rename
- **Schema validation**: All canonical outputs validated on read/write
- **Hash-aware caching**: Outputs include metadata sidecars with input hashes
- **Deterministic processing**: Stable sorts, fixed RNG seeds where relevant
- **JSONL logging**: Structured logs for every pipeline run

## Documentation

- **[FINDINGS.md](FINDINGS.md)** — Key results, neighborhood patterns, and policy implications
- **[METHODOLOGY.md](METHODOLOGY.md)** — Complete methodology, data sources, and interpretation guidelines
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to contribute to this project
- **[data/raw/ACQUISITION.md](data/raw/ACQUISITION.md)** — Step-by-step data download instructions

## Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| CHS-SPARCS correlation | r = 0.122 | Surveys and hospitals see somewhat different populations |
| CHS-Vital correlation | r = 0.237 | Weak overlap between surveys and death records |
| SPARCS-Vital correlation | r = 0.817 | Hospital burden correlates with mortality burden |
| Typologies | 6 clusters | Stable (98.4% bootstrap stability) |
| Demographic visibility gap | 50× | High-need neighborhoods have 50× less survey precision |

## Interpretation Guidelines

- Use "less visible to this system" not "missing people"
- Results are **ecological** (neighborhood-level), not individual-level
- Utilization ≠ population coverage; events ≠ residents
- Every estimate includes reliability/uncertainty indicators

## Testing

```bash
make test          # Run full test suite (95 tests)
make test-smoke    # Run smoke tests only
```

## Citation

If you use this project, please cite:

```
NYC Visibility Atlas (2025). Quantifying differential visibility 
across NYC public health data systems. GitHub repository.
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

