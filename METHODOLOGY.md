# NYC Visibility Atlas: Methodology

This document describes the complete methodology for quantifying differential visibility across NYC public health data systems.

---

## 1. Motivation

Public health surveillance in New York City relies on multiple data systems—surveys, hospital records, vital statistics, and administrative databases. Each system captures a different slice of the population:

- **Surveys** reach people who answer phones and consent to participate
- **Hospitals** see people who seek care and have access to facilities
- **Vital records** capture births and deaths, but miss living conditions
- **311 calls** reflect civic engagement, not health status

When these systems systematically under-capture certain neighborhoods or demographics, health estimates become biased. The **Visibility Atlas** quantifies these blind spots.

---

## 2. Core Concepts

### 2.1 Ecological Cell

The unit of analysis is:

```
Neighborhood (NTA) × Demographic Stratum × Time Window
```

- **55 Neighborhood Tabulation Areas (NTAs)** across NYC
- **Demographic strata**: age groups, race/ethnicity, sex
- **Time window**: 2017–2021 (with 2020 excluded for COVID anomalies)

### 2.2 Visibility Index

For each data source and ecological cell:

```
Visibility = (Observed Events / Population) × 1,000
```

This gives **events per 1,000 residents**—a normalized rate that enables comparison across neighborhoods of different sizes.

### 2.3 Reliability Flags

Every visibility estimate includes a reliability indicator:

| Flag | Meaning | Criteria |
|------|---------|----------|
| `reliable` | Stable estimate | Denominator ≥ 50, numerator ≥ 10 |
| `flagged` | Use with caution | Small numbers, wide confidence intervals |
| `suppressed` | Not reportable | Numerator < 5 (privacy protection) |

---

## 3. Data Sources

### 3.1 Community Health Survey (CHS)

**What it captures:** Self-reported health behaviors and conditions from ~10,000 NYC adults annually.

**Visibility meaning:** Survey response rates vary by neighborhood. Low visibility = the survey has less statistical power to estimate health in that area.

**Key indicators:**
- Diabetes prevalence
- Obesity prevalence
- Smoking rates
- Health insurance status
- Flu vaccination

**Geographic unit:** UHF (United Hospital Fund) neighborhoods → crosswalked to NTA

### 3.2 SPARCS Hospital Encounters

**What it captures:** All inpatient hospitalizations and emergency department visits at NY State hospitals.

**Visibility meaning:** Healthcare utilization rates. High visibility = more hospital contact (which may indicate health burden OR access).

**Key metric:** Encounter rate per 1,000 residents

**Geographic unit:** 3-digit ZIP code → crosswalked to NTA via population weighting

### 3.3 Vital Statistics (Mortality)

**What it captures:** All registered deaths in NYC.

**Visibility meaning:** Mortality burden by neighborhood.

**Key metric:** Age-adjusted death rate per 1,000

**Geographic unit:** Community District → crosswalked to NTA

---

## 4. Geographic Harmonization

Different data sources use different geographic units. We harmonize everything to **NTA (Neighborhood Tabulation Area)** using population-weighted crosswalks:

| Source Geography | Target | Method |
|------------------|--------|--------|
| UHF-42 | NTA | Census tract population weights |
| 3-digit ZIP | NTA | ZCTA-to-tract population weights |
| Community District | NTA | Tract overlap weights |

**Crosswalk principle:** When source X overlaps multiple NTAs, we allocate values proportionally to population.

---

## 5. Pipeline Architecture

The analysis runs as a reproducible pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Geography & Denominators                              │
├─────────────────────────────────────────────────────────────────┤
│  00_build_geographies.py    → Canonical NTA boundaries          │
│  01_build_crosswalks.py     → UHF/ZIP/CD to NTA mappings        │
│  02_build_denominators.py   → ACS population estimates          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Numerators (Source-Specific)                          │
├─────────────────────────────────────────────────────────────────┤
│  03_build_numerator_chs.py      → Survey indicators             │
│  04_build_numerator_sparcs.py   → Hospital encounters           │
│  06_build_numerator_vital.py    → Mortality counts              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Visibility Computation                                │
├─────────────────────────────────────────────────────────────────┤
│  07_build_visibility_tables.py  → Rates + reliability flags     │
│  08_build_cross_source_matrix.py → Source correlations          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Analysis & Synthesis                                  │
├─────────────────────────────────────────────────────────────────┤
│  09_typology_clustering.py      → Neighborhood typologies       │
│  10_predictors_spatial.py       → What predicts visibility?     │
│  11_consequence_vulnerability.py → Vulnerability scores         │
│  12_atlas_assets.py             → Final outputs                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Typology Clustering

We identify **6 distinct visibility profiles** using k-means clustering on z-scored visibility indices:

| Typology | Characteristics |
|----------|-----------------|
| **Survey-Dominant** | High CHS visibility, lower hospital contact |
| **Hospital-Dominant** | High SPARCS visibility, lower survey response |
| **Uniformly High** | Well-captured by all systems |
| **Uniformly Low** | Blind spots across all systems |
| **Mixed** | Variable patterns |
| **Mortality-Divergent** | High mortality visibility, lower survey/hospital |

**Stability:** 98.4% of neighborhoods maintain their typology across 1,000 bootstrap iterations.

---

## 7. Key Findings

### 7.1 Cross-Source Correlations

| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| CHS ↔ SPARCS | r = 0.12 | Surveys and hospitals see different populations |
| CHS ↔ Vital | r = 0.24 | Weak overlap between surveys and mortality |
| SPARCS ↔ Vital | r = 0.82 | Hospital burden correlates with mortality burden |

### 7.2 Demographic Visibility Gaps

- **Highest-need neighborhoods** have 50× less survey precision than lowest-need
- **Hispanic residents** most visible to hospital systems, least to surveys
- **Asian residents** least visible to all systems

### 7.3 Vulnerability Hotspots

Neighborhoods with the highest "estimation vulnerability" (low visibility + high health burden):
1. Hunts Point (Bronx)
2. Brownsville (Brooklyn)
3. East New York (Brooklyn)

---

## 8. Interpretation Guidelines

### What Visibility IS:
- A measure of how much each data system "sees" a neighborhood
- An indicator of statistical power and estimation precision
- A relative comparison across neighborhoods

### What Visibility IS NOT:
- A count of individual people
- A measure of health status
- A judgment of data quality

### Language to use:
- ✅ "This neighborhood is less visible to the hospital system"
- ❌ "This neighborhood is missing from the data"
- ✅ "Survey estimates for this area have wider uncertainty"
- ❌ "We don't know anything about this area"

---

## 9. Reproducibility

### Environment
```bash
conda env create -f environment.yml
pip install -e .
```

### Run Pipeline
```bash
make all  # or: visibility-atlas-run-all
```

### Verify Outputs
```bash
make test
```

### Key Guarantees
- **Atomic writes**: All outputs written via temp file → rename
- **Schema validation**: All outputs validated against defined schemas
- **Metadata sidecars**: Every output includes provenance metadata
- **Deterministic processing**: Fixed seeds, stable sorts

---

## 10. Limitations

1. **Ecological fallacy**: Neighborhood-level patterns don't apply to individuals
2. **Temporal mismatch**: Data sources span slightly different years
3. **Geographic precision**: Crosswalks introduce allocation uncertainty
4. **Missing sources**: Medicaid, immunization registries not yet integrated
5. **COVID disruption**: 2020 data excluded; 2021 may have residual effects

---

## 11. Citation

```
NYC Visibility Atlas (2025). Quantifying differential visibility 
across NYC public health data systems. GitHub repository.
https://github.com/[username]/nyc-visibility-atlas
```

---

## 12. Contact

For questions about methodology or collaboration opportunities, please open a GitHub issue.

