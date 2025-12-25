# NYC Visibility Atlas: Key Findings

## Executive Summary

This project reveals **systematic blind spots** in how NYC's public health data systems "see" different neighborhoods. The core finding: **different data systems capture fundamentally different populations**, with correlations as low as r=0.12 between surveys and hospital records.

---

## 1. The Core Discovery: Data Systems See Different People

### Cross-Source Correlations

| Data Source Pair | Correlation | Statistical Significance | Interpretation |
|------------------|-------------|--------------------------|----------------|
| **CHS ↔ SPARCS** | r = 0.12 | p = 0.063 (marginal) | Surveys and hospitals see **almost entirely different** populations |
| **CHS ↔ Vital** | r = 0.24 | p = 0.0003 | Weak overlap between surveys and death records |
| **SPARCS ↔ Vital** | r = 0.82 | p < 10⁻⁶³ | Hospital burden and mortality are **strongly linked** |

### What This Means

- **Community Health Survey (CHS)** captures people who answer phones, speak English, and consent to participate
- **Hospital records (SPARCS)** capture people who seek emergency or inpatient care
- **Vital statistics** capture deaths — a fundamentally different "event" than the living population

The near-zero correlation between CHS and SPARCS (r = 0.12) is remarkable: neighborhoods with high survey response have *no relationship* to neighborhoods with high hospital utilization.

---

## 2. Borough-Level Patterns

| Borough | Avg Vulnerability | Health Burden | Invisibility | Key Pattern |
|---------|-------------------|---------------|--------------|-------------|
| **Queens** | 57.1 (highest) | Low (30.2) | Very High (84.0) | Healthy but hidden |
| **Bronx** | 55.2 | Very High (77.9) | Low (32.4) | Sick and visible |
| **Brooklyn** | 54.5 | Medium (54.1) | Medium (55.0) | Mixed |
| **Staten Island** | 34.5 | Medium-High (64.9) | Very Low (4.1) | Well-monitored |
| **Manhattan** | 28.7 (lowest) | Medium (41.7) | Low (15.6) | Over-monitored |

### Interpretation

- **Queens is the most "invisible" borough** — residents are healthy but poorly captured by surveys
- **The Bronx is well-monitored but sick** — data systems see the health burden clearly
- **Manhattan is over-represented** — surveys likely over-sample these areas relative to need

---

## 3. Most Vulnerable Neighborhoods

These neighborhoods have the **worst combination** of health burden AND poor data coverage:

| Rank | Neighborhood | Borough | Vulnerability | Health Burden | Invisibility |
|------|--------------|---------|---------------|---------------|--------------|
| 1 | **South Ozone Park** | Queens | 77.3 | 55.9 | 98.6 |
| 2 | **Rosedale** | Queens | 74.2 | 63.0 | 85.3 |
| 3 | **Cambria Heights** | Queens | 74.1 | 49.6 | 98.6 |
| 4 | **Glen Oaks-Floral Park** | Queens | 73.0 | 60.6 | 85.3 |
| 5 | **Maspeth** | Queens | 72.8 | 46.9 | 98.6 |
| 6 | **St. Albans** | Queens | 72.7 | 60.1 | 85.3 |
| 7 | **Douglaston-Little Neck** | Queens | 71.9 | 58.4 | 85.3 |
| 8 | **Rockaway Beach-Arverne** | Queens | 71.7 | 58.2 | 85.3 |
| 9 | **Breezy Point-Belle Harbor** | Queens | 71.2 | 57.0 | 85.3 |
| 10 | **Ozone Park** | Queens | 70.4 | 42.2 | 98.6 |

### The Queens Pattern

**9 of the top 10 most vulnerable neighborhoods are in Queens.** These areas have:
- Moderate health problems (consequence scores 42-63)
- Extremely poor survey coverage (invisibility scores 85-99)
- Potential for systematic underestimation of health needs

---

## 4. Best-Monitored Neighborhoods

These neighborhoods are **most visible** to data systems:

| Rank | Neighborhood | Borough | Vulnerability |
|------|--------------|---------|---------------|
| 1 | Greenwich Village | Manhattan | 17.7 |
| 2 | Central Park | Manhattan | 19.1 |
| 3 | Upper West Side-Lincoln Square | Manhattan | 19.4 |
| 4 | Battery Park-Governors Island | Manhattan | 19.5 |
| 5 | Gramercy | Manhattan | 20.5 |
| 6 | West Village | Manhattan | 20.7 |
| 7 | Tribeca-Civic Center | Manhattan | 21.5 |
| 8 | Stuyvesant Town | Manhattan | 21.9 |

**All top 10 best-monitored neighborhoods are in Manhattan** — suggesting systematic over-representation of these areas in health surveys.

---

## 5. Six Visibility Typologies

We identified **6 distinct patterns** of how neighborhoods are "seen" by different data systems:

### Type 1: High CHS, Low SPARCS/VITAL (63 neighborhoods)
*"Survey-responsive but healthy"*

**Examples:** Chinatown-Two Bridges, Lower East Side, East Village (Manhattan)

**Pattern:** High survey participation, low hospital/death rates. These communities engage with surveys but don't appear in hospital or mortality data.

**Implication:** Survey data may over-represent these areas' health concerns relative to actual burden.

---

### Type 2: Low Visibility Across Systems (52 neighborhoods)
*"The truly invisible"*

**Examples:** Financial District, SoHo-Little Italy, Hudson Square (Manhattan)

**Pattern:** Poor coverage across ALL systems. Neither surveys nor hospitals nor vital records capture these areas well.

**Implication:** Health estimates for these areas are based on the least data — highest uncertainty.

---

### Type 3: High SPARCS, Low CHS (43 neighborhoods)
*"Hospital-visible but survey-invisible"*

**Examples:** Williamsburg, South Williamsburg, East Williamsburg (Brooklyn)

**Pattern:** High hospital utilization but low survey response. These communities interact with healthcare but don't participate in surveys.

**Implication:** Survey-based health estimates may systematically undercount these populations.

---

### Type 4: High CHS/VITAL Only (26 neighborhoods)
*"Surveys capture mortality burden"*

**Examples:** Claremont Village, Crotona Park, Claremont Park (Bronx)

**Pattern:** High survey and death visibility, but lower hospital visibility. Surveys align with mortality patterns.

**Implication:** These areas may have barriers to healthcare access despite health needs.

---

### Type 5: High CHS/SPARCS Only (25 neighborhoods)
*"Living data systems align"*

**Examples:** Greenpoint, Brooklyn Heights, Downtown Brooklyn-DUMBO (Brooklyn)

**Pattern:** Surveys and hospitals see similar populations, but mortality patterns differ.

**Implication:** Good alignment between living-population data sources.

---

### Type 6: High VITAL Only (23 neighborhoods)
*"Death visible, life invisible"*

**Examples:** Morrisania, Crotona Park East, Mount Eden-Claremont (Bronx)

**Pattern:** High mortality visibility but poor survey and hospital coverage.

**Implication:** These areas experience mortality burden that isn't reflected in living-population surveys.

---

## 6. Paradoxical Cases

### High Health Burden, Good Visibility
*"The system sees their problems"*

| Neighborhood | Borough | Health Burden | Invisibility |
|--------------|---------|---------------|--------------|
| Mott Haven-Port Morris | Bronx | 94 | 29 |
| Melrose | Bronx | 94 | 29 |
| Longwood | Bronx | 90 | 29 |
| Hunts Point | Bronx | 89 | 39 |

These South Bronx neighborhoods have severe health problems AND good data coverage. **Policy interventions here can be data-informed** because we actually measure the problem.

---

### Low Health Burden, Poor Visibility
*"Hidden but healthy"*

| Neighborhood | Borough | Health Burden | Invisibility |
|--------------|---------|---------------|--------------|
| Astoria (East)-Woodside (North) | Queens | 16 | 99 |
| Sunnyside Yards (North) | Queens | 15 | 85 |
| Old Astoria-Hallets Point | Queens | 20 | 85 |

These Queens neighborhoods are healthy but almost completely invisible to surveys. **They may be systematically excluded from resource allocation** that uses survey-based health metrics.

---

## 7. Policy Implications

### For Resource Allocation

| Finding | Implication |
|---------|-------------|
| Queens neighborhoods are systematically invisible | Survey-based funding formulas may undercount Queens health needs |
| South Bronx is well-measured despite high burden | Resources can be targeted effectively using existing data |
| Manhattan is over-represented in surveys | Health estimates for Manhattan may be more precise than warranted |

### For Survey Design

| Finding | Implication |
|---------|-------------|
| CHS-SPARCS correlation = 0.12 | Surveys miss populations that interact with hospitals |
| 18-24 age group has lowest sample size | Young adults are systematically under-surveyed |
| Some neighborhoods have 66% suppression rate | Fine-grained estimates are unreliable for many areas |

### For Health Equity

| Finding | Implication |
|---------|-------------|
| 9/10 most vulnerable neighborhoods in Queens | Geographic inequity in data coverage |
| Typology clusters are 91% stable | Visibility patterns are persistent, not random |
| Hospital and mortality correlate (r=0.82) but surveys don't | Survey-based interventions may target wrong populations |

---

## 8. Limitations

1. **Ecological analysis**: These patterns are at the neighborhood level, not individual level
2. **Temporal alignment**: Data sources span slightly different time periods (2017-2021)
3. **Missing sources**: Medicaid enrollment, immunization registries not yet integrated
4. **COVID disruption**: 2020 excluded; 2021 may have residual pandemic effects

---

## 9. Technical Details

- **Unit of analysis**: 262 Neighborhood Tabulation Areas (NTAs)
- **Time period**: 2017-2021 (excluding 2020)
- **Data sources**: CHS (surveys), SPARCS (hospital records), Vital Statistics (deaths)
- **Clustering**: K-means with k=6, 91.2% bootstrap stability
- **Vulnerability score**: Average of health burden (percentile) and invisibility (percentile)

---

## Citation

```
NYC Visibility Atlas (2025). Quantifying differential visibility 
across NYC public health data systems. 
https://github.com/yoelplutchok/nyc-visibility-atlas
```

