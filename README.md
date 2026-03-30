# Metro Service Accessibility in Chinese Cities (2000–2025)

**Companion code and data for:**

> Wang, L., Guan, Y., Kwan, M.-P., Zhang, X., Zhang, X., & Liu, Y. (2026). Urban Development Trajectories in China since 2000: A Metro Service Assessment Perspective. *Nature Cities* (under review).

---

## Overview

This repository provides the analysis code and city-level aggregated data for a study that develops a two-dimensional framework—coupling **service intensity (S1)** with **network accessibility (S2)**—to evaluate metro service quality across 44 Chinese cities from 2000 to 2025.

**Key findings:**
- Network scale and service quality are only weakly correlated (R² = 0.03).
- Expansion systematically dilutes average service intensity: 80% of cities experienced S1 decline by 2020–2025.
- Only 18.2% of cities achieve a "win–win" between efficiency and equality.
- Over three-quarters of stations remain locked in their initial service category (path dependence).

## Repository Structure

```
metro-accessibility/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── code/
│   ├── analysis/
│   │   └── 1-yearly_accessibility.py  # Core S1/S2 computation pipeline
│   │
│   └── visualization/
│       ├── figure1_scale_vs_quality.py           # Fig. 1: Scale ≠ quality
│       ├── figure2_development_trajectories.py   # Fig. 2: Strategic pathways
│       ├── figure3_efficiency_equality.py        # Fig. 3: Efficiency–equality tradeoffs
│       └── figure4_path_dependence.py            # Fig. 4: Station path dependence
│
└── data/
    └── figshare_deposit/
        ├── city_level_accessibility_panel.csv    # City-year panel (555 rows)
        └── national_yearly_summary.csv           # National yearly summary (26 rows)
```

## Data Description

### City-Level Accessibility Panel (`city_level_accessibility_panel.csv`)

City-year panel dataset aggregated from station-level calculations. Each row represents one city in one year. **555 rows × 19 columns.**

| Column | Description |
|---|---|
| `year` | Year (2000–2025) |
| `city` | City name (English) |
| `city_tier` | City tier classification (Tier-1, New Tier-1, Tier-2, Tier-3) |
| `n_stations` | Number of operational metro stations |
| `mean_S1` | Mean service intensity (population within 1 km buffer) |
| `median_S1` | Median S1 |
| `std_S1` | Standard deviation of S1 |
| `min_S1` | Minimum S1 |
| `max_S1` | Maximum S1 |
| `mean_S2` | Mean network accessibility (cumulative population reachable within 30 min) |
| `median_S2` | Median S2 |
| `std_S2` | Standard deviation of S2 |
| `gini_S1` | Gini coefficient of intra-city S1 distribution |
| `equality` | Equality metric (1 − Gini of S1) |
| `efficiency` | Efficiency metric (min–max normalised mean S2 per year) |
| `pct_HH` | Proportion of HH-type stations (high intensity, high accessibility) |
| `pct_HL` | Proportion of HL-type stations (high intensity, low accessibility) |
| `pct_LH` | Proportion of LH-type stations (low intensity, high accessibility) |
| `pct_LL` | Proportion of LL-type stations (low intensity, low accessibility) |

### National Yearly Summary (`national_yearly_summary.csv`)

Aggregate statistics across all cities for each year. **26 rows × 12 columns.**

### Station-Level Data

Station-level accessibility data are available from the corresponding author upon reasonable request. See the Data Availability Statement in the manuscript for details.

## Methodology

### Service Intensity (S1)

S1 is defined as the total population within a 1-km buffer around each station, computed by spatially overlaying 100-m resolution population grids with station buffers:

$$S1_i = \sum_k P_k \cdot w_k$$

where $P_k$ is the population of the *k*-th grid cell and $w_k$ is the areal proportion of that cell within the buffer.

### Network Accessibility (S2)

S2 is defined as the cumulative population reachable from a given station within 30 minutes via the metro network:

$$S2_i = \sum_{j \in R(i)} S1_j$$

where $R(i)$ is the set of all stations reachable from station *i* within 30 minutes (average speed: 35 km/h; transfer penalty: 5 min).

### Station Classification

Using yearly median thresholds for S1 and S2, stations are classified into four types: HH (high–high), HL (high–low), LH (low–high), and LL (low–low).

## Data Sources

| Data | Source | Resolution | Access |
|---|---|---|---|
| Metro station timeline | China Public Transport Operation Network Dataset (CPTOND-2025) | Station-level | Wang et al. (2026) |
| Population grid | WorldPop | 100 m | [worldpop.org](https://www.worldpop.org/) |
| Census population | China 5th/6th/7th National Population Census | County-level | National Bureau of Statistics of China |

## Requirements

```
Python >= 3.9
geopandas >= 0.12
pandas >= 1.5
numpy >= 1.23
rasterio >= 1.3
matplotlib >= 3.6
tqdm
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Reproducing Figures

Each figure script reads the city-level panel data and generates publication-quality figures:

```bash
# Figure 1: Scale vs. quality
python code/visualization/figure1_scale_vs_quality.py data/figshare_deposit/city_level_accessibility_panel.csv output/

# Figure 2: Development trajectories
python code/visualization/figure2_development_trajectories.py data/figshare_deposit/city_level_accessibility_panel.csv output/

# Figure 3: Efficiency–equality tradeoffs
python code/visualization/figure3_efficiency_equality.py data/figshare_deposit/city_level_accessibility_panel.csv output/

# Figure 4: Path dependence
python code/visualization/figure4_path_dependence.py data/figshare_deposit/city_level_accessibility_panel.csv output/
```

### Running the Full Computation Pipeline

The core accessibility computation (`code/analysis/1-yearly_accessibility.py`) requires the original input datasets (metro station shapefile, WorldPop TIF rasters, and county-level census data). See the script header for path configuration.

## Citation

If you use this code or data, please cite:

```bibtex
@article{wang2026metro,
  title={Urban Development Trajectories in China since 2000: A Metro Service Assessment Perspective},
  author={Wang, Liang and Guan, Yu and Kwan, Mei-Po and Zhang, Xiaodong and Zhang, Xinhua and Liu, Yu},
  journal={Nature Cities},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Corresponding author:** Yu Liu ([yuliugis@pku.edu.cn](mailto:yuliugis@pku.edu.cn))
- **Data/code inquiries:** Liang Wang
