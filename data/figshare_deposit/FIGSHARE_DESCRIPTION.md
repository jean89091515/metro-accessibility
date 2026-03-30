# Data Deposit: Metro Service Accessibility in Chinese Cities (2000–2025)

## Title
City-level metro service accessibility panel data for 44 Chinese cities (2000–2025)

## Description
This dataset accompanies the paper "Urban Development Trajectories in China since 2000: A Metro Service Assessment Perspective" submitted to *Nature Cities*.

It provides city-level aggregated metrics derived from a two-dimensional accessibility framework coupling service intensity (S1, population within 1 km of each station) with network accessibility (S2, cumulative population reachable within 30 minutes via metro). The data covers 44 Chinese cities with operational metro systems over the period 2000–2025, computed from over 6,000 station-year records.

## Files

### 1. city_level_accessibility_panel.csv (555 rows × 19 columns)
City-year panel dataset. Each row represents one city in one year (e.g., Beijing in 2010). Includes:
- Station count per city-year
- Mean, median, standard deviation, min, and max of S1 (service intensity)
- Mean, median, and standard deviation of S2 (network accessibility)
- Gini coefficient of intra-city S1 distribution
- Equality metric (1 − Gini)
- Efficiency metric (normalised mean S2)
- Proportions of four station types (HH, HL, LH, LL)
- City tier classification

### 2. national_yearly_summary.csv (26 rows × 12 columns)
National aggregate statistics for each year (2000–2025). Includes number of cities with metro, total stations, mean/median S1 and S2, Gini coefficient, and station type proportions.

## Methodology
- **S1 (Service Intensity):** Total population within a 1-km buffer around each metro station, computed using WorldPop 100-m population grids calibrated with China's national census data (2000, 2010, 2020).
- **S2 (Network Accessibility):** Cumulative population reachable within 30 minutes through the metro network (average speed: 35 km/h; transfer penalty: 5 min).
- **Station classification:** Stations are classified into HH/HL/LH/LL types using yearly median thresholds for S1 and S2.
- **Aggregation:** City-level statistics are computed from station-level values within each city-year.

## Data Sources
- Metro station data: China Public Transport Operation Network Dataset (CPTOND-2025)
- Population data: WorldPop (https://www.worldpop.org/) + China 5th/6th/7th National Population Census
- City tier classification follows the standard four-tier system based on economic development level

## Code Availability
Analysis and visualization code is available at: https://github.com/jean89091515/metro-accessibility

## Authors
Liang Wang, Yu Guan, Mei-Po Kwan, Xiaodong Zhang, Xinhua Zhang, Yu Liu

## License
CC BY 4.0

## Keywords
metro accessibility, urban development, China, public transit, service intensity, network accessibility, urbanisation

## Categories
Urban Studies, Transportation, Geography
