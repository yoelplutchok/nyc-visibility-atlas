# NYC Visibility Atlas - Pipeline Makefile
# Single-command pipeline runner as specified in Section 9
#
# Usage:
#   make all          - Run full pipeline
#   make geographies  - Run Step 00 only
#   make clean        - Remove all generated outputs
#   make test         - Run test suite
#   make test-smoke   - Run smoke tests only

.PHONY: all clean test test-smoke help

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python

# Directories
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
LOGS := logs
REPORTS := reports

# Pipeline scripts
SCRIPTS := scripts

#------------------------------------------------------------------------------
# HELP
#------------------------------------------------------------------------------
help:
	@echo "NYC Visibility Atlas - Pipeline Commands"
	@echo ""
	@echo "Pipeline Steps:"
	@echo "  make geographies     - 00: Build canonical geography files"
	@echo "  make crosswalks      - 01: Build population-weighted crosswalks"
	@echo "  make denominators    - 02: Build ACS denominators"
	@echo "  make numerator-chs   - 03: Build CHS numerator"
	@echo "  make numerator-sparcs- 04: Build SPARCS numerator"
	@echo "  make numerator-enroll- 05: Build enrollment numerator"
	@echo "  make numerator-vital - 06: Build vital events numerator"
	@echo "  make visibility      - 07: Build harmonized visibility tables"
	@echo "  make matrix          - 08: Build cross-source matrix"
	@echo "  make typology        - 09: Run typology clustering"
	@echo "  make predictors      - 10: Run predictor models"
	@echo "  make vulnerability   - 11: Compute estimate vulnerability"
	@echo "  make atlas           - 12: Build atlas assets"
	@echo ""
	@echo "Aggregate Targets:"
	@echo "  make all             - Run full pipeline (all steps)"
	@echo "  make data            - Run data preparation steps (00-07)"
	@echo ""
	@echo "Utilities:"
	@echo "  make test            - Run full test suite"
	@echo "  make test-smoke      - Run smoke tests only"
	@echo "  make clean           - Remove generated outputs"
	@echo "  make clean-logs      - Remove log files only"
	@echo "  make help            - Show this help message"

#------------------------------------------------------------------------------
# PIPELINE STEPS
#------------------------------------------------------------------------------
geographies:
	$(PYTHON) $(SCRIPTS)/00_build_geographies.py

crosswalks: geographies
	$(PYTHON) $(SCRIPTS)/01_build_crosswalks.py

denominators: crosswalks
	$(PYTHON) $(SCRIPTS)/02_build_denominators_acs.py

numerator-chs: denominators
	$(PYTHON) $(SCRIPTS)/03_build_numerator_chs.py

numerator-sparcs: denominators
	$(PYTHON) $(SCRIPTS)/04_build_numerator_sparcs.py

numerator-enroll: denominators
	$(PYTHON) $(SCRIPTS)/05_build_numerator_enrollment.py

numerator-vital: denominators
	$(PYTHON) $(SCRIPTS)/06_build_numerator_vital.py

visibility: numerator-chs numerator-sparcs numerator-enroll numerator-vital
	$(PYTHON) $(SCRIPTS)/07_build_visibility_tables.py

matrix: visibility
	$(PYTHON) $(SCRIPTS)/08_build_cross_source_matrix.py

typology: matrix
	$(PYTHON) $(SCRIPTS)/09_typology_clustering.py

predictors: typology
	$(PYTHON) $(SCRIPTS)/10_predictors_and_spatial_diagnostics.py

vulnerability: predictors
	$(PYTHON) $(SCRIPTS)/11_consequence_vulnerability.py

atlas: vulnerability
	$(PYTHON) $(SCRIPTS)/12_build_atlas_assets.py

#------------------------------------------------------------------------------
# AGGREGATE TARGETS
#------------------------------------------------------------------------------
data: visibility
	@echo "Data preparation complete (steps 00-07)"

all: atlas
	@echo "Full pipeline complete"

#------------------------------------------------------------------------------
# TESTING
#------------------------------------------------------------------------------
test:
	pytest tests/ -v

test-smoke:
	pytest tests/ -v -m smoke

#------------------------------------------------------------------------------
# CLEANUP
#------------------------------------------------------------------------------
clean-logs:
	rm -rf $(LOGS)/*.jsonl

clean: clean-logs
	rm -rf $(DATA_PROCESSED)/*
	rm -rf $(REPORTS)/figures/*
	rm -rf $(REPORTS)/tables/*
	rm -rf $(REPORTS)/atlas/*
	@echo "Cleaned generated outputs"

