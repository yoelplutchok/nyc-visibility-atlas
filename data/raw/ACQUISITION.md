# Data Acquisition Guide

This document provides step-by-step instructions for acquiring all data required by the NYC Visibility Atlas pipeline.

**Last Updated:** 2025-12-25

---

## Quick Reference

| Dataset | Source | Auto-Download | Manual Steps |
|---------|--------|---------------|--------------|
| NTA Boundaries | NYC Open Data | ✅ Yes | None |
| Census Tracts | Census Bureau | ✅ Yes | None |
| ACS Populations | Census API | ✅ Yes | None |
| SPARCS Encounters | Health Data NY | ✅ Yes | None |
| CHS Indicators | EpiQuery | ❌ No | **See Section 2** |
| UHF Boundaries | NYC DOHMH GitHub | ✅ Yes | None |
| Vital Statistics | EpiQuery | ❌ No | **See Section 3** |

---

## 1. Automatic Downloads (No Action Required)

The following datasets are downloaded automatically by pipeline scripts:

### NTA Boundaries (Step 00)
- **Source:** NYC Open Data
- **URL:** https://data.cityofnewyork.us/api/geospatial/9nt8-h7nd
- **Script:** `00_build_geographies.py`

### Census Tracts & ZCTA (Step 01)
- **Source:** Census Bureau Cartographic Boundaries
- **Script:** `01_build_crosswalks.py`

### ACS Population Denominators (Step 02)
- **Source:** Census Bureau API
- **Vintage:** 2022 5-year (2018-2022)
- **Script:** `02_build_denominators_acs.py`

### SPARCS Hospital Discharges (Step 04b)
- **Source:** Health Data NY API
- **URL:** https://health.data.ny.gov/
- **Script:** `15_enhanced_sparcs_multiyear.py`

### UHF42 Boundaries (Step 02b)
- **Source:** NYC DOHMH EHDP-data GitHub
- **URL:** https://github.com/nychealth/EHDP-data
- **File:** Already downloaded to `data/raw/geo/uhf42_boundaries.geojson`

---

## 2. CHS (Community Health Survey) - MANUAL EXPORT REQUIRED

The CHS data must be manually exported from NYC EpiQuery because:
- The API does not provide neighborhood-level estimates
- Confidence intervals are only available through the web interface

### Step-by-Step Instructions

1. **Navigate to EpiQuery:**
   https://a816-health.nyc.gov/hdi/epiquery/

2. **Select Survey:**
   - Choose "Community Health Survey (CHS)"
   - Select year: **2019** (for consistency with pipeline)

3. **For each indicator, export neighborhood data:**

   **Required Indicators:**
   | Indicator | Query Path |
   |-----------|------------|
   | Diabetes | Chronic Disease → Diabetes → Current diabetes |
   | Asthma | Chronic Disease → Asthma → Current asthma |
   | Obesity | Body Weight → Obesity (BMI ≥30) |
   | High Blood Pressure | Chronic Disease → Hypertension |
   | Uninsured | Insurance → Currently uninsured |
   | Flu Vaccination | Immunization → Flu shot (last 12 months) |
   | Current Smoking | Tobacco → Current smoker |
   | Binge Drinking | Alcohol → Binge drinking |

4. **For each indicator:**
   - Select "Analyze by Neighborhood (UHF)"
   - Select Response: "Yes" (for prevalence indicators)
   - Click "Create Table"
   - Click "Export" → "Download Crosstab"
   - Save as: `data/raw/chs/epiquery_exports/chs_<indicator>_2019.csv`

5. **File naming convention:**
   ```
   chs_diabetes_2019.csv
   chs_asthma_2019.csv
   chs_obesity_2019.csv
   chs_highbp_2019.csv
   chs_uninsured_2019.csv
   chs_fluvax_2019.csv
   chs_smoking_2019.csv
   chs_binge_2019.csv
   ```

6. **Verify exports:**
   - Each file should contain ~34 rows (one per UHF neighborhood)
   - Columns should include: neighborhood name, prevalence, CI bounds

### Expected File Format

EpiQuery exports are UTF-16 tab-separated with columns like:
```
Yearnum  Select Indicator  Dimension Response  Estimated Prevalence  Lower CI  Upper CI  Interpretation Flag
```

The pipeline (`03_build_numerator_chs.py`) automatically parses this format.

---

## 3. Vital Statistics (Mortality) - MANUAL EXPORT REQUIRED

Mortality data must be manually exported from EpiQuery.

### Step-by-Step Instructions

1. **Navigate to EpiQuery:**
   https://a816-health.nyc.gov/hdi/epiquery/

2. **Select Dataset:**
   - Choose "Vital Statistics"
   - Select "Deaths" → "Mortality (All Causes)"
   - Select year: **2019**

3. **Export by Community District:**
   - Select "Analyze by Community District"
   - Click "Create Table"
   - Click "Export" → "Download Crosstab"
   - Save as: `data/raw/vital/epiquery_exports/mortality_overall_2019.csv`

4. **Verify export:**
   - Should contain ~59 rows (one per Community District)
   - Columns: CD name, crude rate, age-adjusted rate, CI bounds

---

## 4. Optional: Demographic Stratification

For demographic visibility analysis (Step 13), export stratified CHS data:

### Required Demographic Exports

| Stratification | Output File |
|----------------|-------------|
| By Race (citywide) | `chs_diabetes_by_race_2019.csv` |
| By Age (citywide) | `chs_diabetes_by_age_2019.csv` |
| By Poverty (citywide) | `chs_diabetes_by_poverty_2019.csv` |
| By Neighborhood × Race | `chs_diabetes_by_neighbd_race_2019.csv` |

Save these to: `data/raw/chs/epiquery_exports/demographics/`

---

## 5. 311 Data (Optional Civic Layer)

311 data is downloaded automatically but requires network access:
- **Source:** NYC Open Data API
- **Note:** As of Dec 2025, data is split into two datasets:
  - 2020-Present: Same API endpoint
  - 2010-2019: Separate historical dataset
- **Script:** `16_add_311_civic_layer.py`

---

## 6. Verification Checklist

After completing manual exports, verify:

```bash
# Check CHS exports exist
ls -la data/raw/chs/epiquery_exports/*.csv

# Expected: 8 files (diabetes, asthma, obesity, highbp, uninsured, fluvax, smoking, binge)

# Check vital exports exist  
ls -la data/raw/vital/epiquery_exports/*.csv

# Expected: mortality_overall_2019.csv
```

---

## 7. Troubleshooting

### EpiQuery exports are empty or have wrong format
- Ensure you selected "Analyze by Neighborhood (UHF)" not citywide
- Check that you downloaded "Crosstab" not "Data"
- Verify year is 2019

### Pipeline can't find CHS data
- Check file naming matches expected pattern: `chs_<indicator>_2019.csv`
- Files must be in `data/raw/chs/epiquery_exports/`

### UHF codes don't match
- CHS uses UHF42 codes (101, 102, etc.)
- The pipeline's UHF42 crosswalk handles this automatically

---

## 8. Data Provenance

After all data is acquired, update the manifest:

```json
// data/raw/_manifest.json
{
  "chs": {
    "source": "EpiQuery manual export",
    "date_acquired": "2025-12-25",
    "year": 2019,
    "n_files": 8
  },
  "vital": {
    "source": "EpiQuery manual export", 
    "date_acquired": "2025-12-25",
    "year": 2019
  }
}
```

---

## Contact

For questions about data acquisition:
- CHS data: NYC DOHMH https://www1.nyc.gov/site/doh/data/data-sets/community-health-survey.page
- EpiQuery support: https://a816-health.nyc.gov/hdi/epiquery/

