# EpiQuery CHS Data Exports

## Instructions

Export the following indicators from EpiQuery to this folder:

### URL
https://a816-health.nyc.gov/hdi/epiquery/visualizations?PageType=ps&PopulationSource=CHS

### Indicators to Export

| # | Indicator | Topic | Subtopic | Filename |
|---|-----------|-------|----------|----------|
| 1 | Diabetes (ever) | Diseases and Conditions | Chronic Diseases | `chs_diabetes_2019.csv` |
| 2 | High blood pressure (ever) | Diseases and Conditions | Chronic Diseases | `chs_highbp_2019.csv` |
| 3 | Obesity (BMI ≥30) | Healthy Living | Nutrition | `chs_obesity_2019.csv` |
| 4 | Current smoker | Healthy Living | Smoking | `chs_smoking_2019.csv` |
| 5 | Current asthma | Diseases and Conditions | Chronic Diseases | `chs_asthma_2019.csv` |
| 6 | Mental health treatment | Mental Health | Mental Health Counseling | `chs_mentalhealth_2019.csv` |
| 7 | Binge drinking | Healthy Living | Drug and Alcohol Use | `chs_binge_2019.csv` |
| 8 | Uninsured | Health Care Access | Health Insurance | `chs_uninsured_2019.csv` |
| 9 | No primary care provider | Health Care Access | Health Care Use | `chs_nopcp_2019.csv` |
| 10 | Flu vaccine (past 12 mo) | Healthy Living | Vaccinations | `chs_fluvax_2019.csv` |

### Export Steps (for each indicator)

1. Select Topic and Subtopic from dropdowns
2. Click on the indicator cell in the table (for year 2019 or most recent)
3. Click **"Analyze by Neighborhood"** tab
4. Set Year filter to 2019 (or most recent)
5. Right-click on the neighborhood table → Export Data (or find download icon)
6. Save as CSV with the filename from the table above

### Required Columns

Each CSV should have (column names may vary):
- `uhf_name` or `Neighborhood` - UHF neighborhood name
- `estimate` or `Percent` or `Prevalence` - Prevalence percentage
- `ci_low` or `Lower CI` - Lower 95% confidence interval
- `ci_high` or `Upper CI` - Upper 95% confidence interval

### After Exporting

Run: `python scripts/03_build_numerator_chs.py --use-epiquery`

The script will automatically process these files.

