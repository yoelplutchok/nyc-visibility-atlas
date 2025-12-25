# CHS Demographic Stratification Export Instructions

## Overview
Export CHS indicators broken down by **race/ethnicity** and **age group** to reveal which demographic groups are most invisible to health surveys.

**Target Year:** 2019 (consistent with existing data)

---

## Required Exports

### Export 1: Diabetes by Race/Ethnicity
1. Go to: https://a816-health.nyc.gov/hdi/epiquery/
2. Click **"+Chronic Conditions"** → **"Diabetes"**
3. Click on **"Diabetes (ever told)"** in the data table
4. Click **"Analyze by Demographics"** tab (NOT Neighborhood)
5. Select **Year: 2019**
6. Find the **Race/Ethnicity** breakdown table
7. Click **Download** → Save as CSV
8. **Filename:** `chs_diabetes_by_race_2019.csv`

### Export 2: Diabetes by Age Group
1. Same indicator as above
2. Find the **Age Group** breakdown table
3. Click **Download** → Save as CSV
4. **Filename:** `chs_diabetes_by_age_2019.csv`

### Export 3: General Health by Race/Ethnicity
1. Click **"+General Health"** → **"Self-Reported Health Status"**
2. Click on **"Self-Reported Fair or Poor Health"**
3. Click **"Analyze by Demographics"** tab
4. Select **Year: 2019**
5. Download Race/Ethnicity breakdown
6. **Filename:** `chs_fair_poor_health_by_race_2019.csv`

### Export 4: General Health by Age Group
1. Same indicator as above
2. Download Age Group breakdown
3. **Filename:** `chs_fair_poor_health_by_age_2019.csv`

### Export 5: Insurance by Race/Ethnicity
1. Click **"+Health Care Access"** → **"Insurance"**
2. Click on **"Currently uninsured"**
3. Click **"Analyze by Demographics"** tab
4. Select **Year: 2019**
5. Download Race/Ethnicity breakdown
6. **Filename:** `chs_uninsured_by_race_2019.csv`

### Export 6: Insurance by Age Group
1. Same indicator as above
2. Download Age Group breakdown
3. **Filename:** `chs_uninsured_by_age_2019.csv`

---

## Save Location

Save all files to:
```
/Users/yoelplutchok/Desktop/NYC Visibility Atlas/data/raw/chs/epiquery_exports/demographics/
```

---

## Expected Columns

The demographic exports should have columns like:
- `Race/Ethnicity` or `Age Group` — the demographic category
- `Estimated Prevalence` or `Percent` — the estimate
- `Lower Confidence Interval` / `Upper Confidence Interval` — for SE calculation
- `Sample Size` or `N` (if available) — actual respondent counts

---

## Alternative: Neighborhood × Demographics Cross-tabulation

If EpiQuery allows, the most valuable export would be:
- **Indicator by Neighborhood AND Race/Ethnicity** (cross-tabulation)

This would show visibility by demographic group *within* each neighborhood.

Look for options like:
- "Analyze by Neighborhood" → then filter by demographic
- Cross-tabulation or pivot options

If you find this option, export it with filename:
`chs_diabetes_by_neighborhood_and_race_2019.csv`

---

## Notes

- Some demographic breakdowns may have suppressed cells (small sample sizes)
- This is expected — suppression itself reveals visibility gaps
- If an export fails or data is unavailable, note it in the manifest below

---

## Manifest (fill in after export)

```json
{
  "export_date": null,
  "files_exported": [],
  "files_missing": [],
  "notes": ""
}
```

