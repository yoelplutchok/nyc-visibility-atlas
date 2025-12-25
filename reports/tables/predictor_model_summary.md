# Visibility Predictor Model Summary

## Overview
Linear regression models predicting visibility from neighborhood characteristics.

## Model Results by Source

### CHS

- **N observations:** 259
- **R²:** 1.000
- **Moran's I (residuals):** 0.3896020741960836

**Coefficients:**

| Predictor | Coefficient |
|-----------|-------------|
| intercept | 0.0000 |
| pct_age_0_17 | 0.0009 |
| pct_age_18_24 | 0.0009 |
| pct_age_25_44 | 0.0009 |
| pct_age_45_64 | 0.0009 |
| pct_age_65_plus | 0.0009 |
| borough_Bronx | -0.0072 |
| borough_Brooklyn | -0.0432 |
| borough_Queens | -0.0454 |
| borough_Staten Island | 0.1551 |

### SPARCS

- **N observations:** 223
- **R²:** 0.067
- **Moran's I (residuals):** 0.035909498429531284

**Coefficients:**

| Predictor | Coefficient |
|-----------|-------------|
| intercept | 0.0203 |
| pct_age_0_17 | 1.0797 |
| pct_age_18_24 | -0.4522 |
| pct_age_25_44 | 0.0204 |
| pct_age_45_64 | 0.6285 |
| pct_age_65_plus | 0.7523 |
| borough_Bronx | 7.5879 |
| borough_Brooklyn | -10.8858 |
| borough_Queens | -2.9594 |
| borough_Staten Island | -14.6860 |

### VITAL

- **N observations:** 259
- **R²:** 1.000
- **Moran's I (residuals):** 0.024855635537200996

**Coefficients:**

| Predictor | Coefficient |
|-----------|-------------|
| intercept | 0.0037 |
| pct_age_0_17 | 0.0731 |
| pct_age_18_24 | 0.0731 |
| pct_age_25_44 | 0.0731 |
| pct_age_45_64 | 0.0731 |
| pct_age_65_plus | 0.0731 |
| borough_Bronx | -0.3536 |
| borough_Brooklyn | -1.1429 |
| borough_Queens | -0.7778 |
| borough_Staten Island | -0.7950 |

## Interpretation Notes

- R² indicates proportion of variance explained by neighborhood demographics
- Borough effects capture systematic differences across boroughs
- Low R² suggests visibility is NOT strongly determined by measurable demographics
- Moran's I > 0 indicates positive spatial autocorrelation of residuals