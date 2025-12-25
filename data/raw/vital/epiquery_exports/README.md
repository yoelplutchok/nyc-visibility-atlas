# Vital Statistics EpiQuery Export Instructions

## Overview
Export mortality data from NYC EpiQuery to use as the third visibility layer in the NYC Visibility Atlas.

**Target Year:** 2019 (aligns with CHS and SPARCS data)

## Export Steps

### Step 1: Navigate to EpiQuery Mortality Data
1. Go to: https://a816-health.nyc.gov/hdi/epiquery/
2. Under **Topic**, click **"+Birth and Death"** to expand
3. Click **"Mortality and Premature Mortality"**

### Step 2: Select the Indicator
1. In the data table, click on **"Overall"** under **Mortality** (Age-adjusted rate per 1,000 population)
2. This will open the detailed view with tabs

### Step 3: Navigate to Neighborhood View
1. Look for the tabs at the bottom of the visualization area:
   - Return to Indicator List
   - Indicator Overview
   - **Analyze by Neighborhood** ← Click this one
   - Analyze by Demographics
   - Leading Causes of Death

### Step 4: Filter to Year 2019
1. In the filter controls, select **Year: 2019**
2. Make sure **Mortality** (not Premature Mortality) is selected
3. Make sure **Metric: Age-Adjusted Death Rate per 1000 Population** is selected

### Step 5: Export the Data
1. Look for the download/export button (usually in the toolbar)
2. Export as **CSV** or use **Crosstab** export
3. Save as: `mortality_overall_2019.csv`

## Required Exports (Priority Order)

| Indicator | Filename | Notes |
|-----------|----------|-------|
| **Overall Mortality** | `mortality_overall_2019.csv` | Age-adjusted rate per 1,000 |
| Heart Disease | `mortality_heart_disease_2019.csv` | Optional but valuable |
| Diabetes | `mortality_diabetes_2019.csv` | Optional but valuable |

## Expected Data Format
The export should include columns like:
- `Neighborhood` or `UHF` (neighborhood name/code)
- `Rate` or `Age-Adjusted Rate` (per 1,000 population)
- Possibly confidence intervals

## After Export
Place all CSV files in this directory:
```
data/raw/vital/epiquery_exports/
```

The `06_build_numerator_vital.py` script will automatically detect and process these files.

## Notes
- The data is at **UHF34 neighborhood level** (same as CHS)
- We'll use the same UHF→NTA crosswalk we built for CHS
- Vital statistics have near-complete coverage (all deaths are reported)

