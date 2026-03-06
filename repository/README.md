# Numerical Weather Prediction (NWP) Modeling Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code, data processing scripts, and final report for the project **"Numerical Weather Prediction (NWP): Multi‑Variable Temperature Forecasting with ERA5 Data"**, completed as part of MATH 406: Mathematical Modeling at Duke Kunshan University.

The project develops and evaluates two complementary approaches for short‑term (1‑, 6‑, 12‑hour) temperature forecasting across 14 East Asian cities:

1. **Physics‑informed statistical model** – Ridge regression with features derived from atmospheric physics (temperature tendency, advection proxy, lagged variables).
2. **Simplified numerical NWP model** – 2D finite‑difference solver for the primitive equations (shallow‑water + thermal coupling), initialized with ERA5 data.

Key findings include a **seasonal reversal of predictability**: inland cities are harder to predict in summer (convection), while coastal cities become more challenging in winter (ocean‑atmosphere interactions). The statistical model achieves ~1.0°C RMSE for 1‑hour forecasts, while the numerical model reproduces large‑scale patterns without training data (6‑h RMSE ~4.2°C).

---

## 📁 Repository Structure

```
├── README.md
├── data/                                        # (not included – see instructions below)
│   └── era5_eastasia_january26.nc
│   └── era5_eastasia_july25.nc
│   └── era5_pde.nc
├── src/                                         # Source code
│   ├── nwp_numerical_simulation.py              # Numerical simulation and validation model
│   ├── summer_forecast_analysis.py              # Summer statistical forecast
│   ├── winter_forecast_analysis.py              # Winter statistical forecast
│   ├── seasonal_comparison_forecast_analysis.py # Seasonal comparison plots
├── slides/                                      # Progress report slides
│   └── MATH406_NWP1_Eunice_Gu.pdf
│   └── MATH406_NWP2_Eunice_Gu.pdf
│   └── MATH406_NWP3_Eunice_Gu.pdf
│   └── MATH406_NWP4_Eunice_Gu.pdf
├── report/                                      # Final written report
│   └── NWP Mathematical Modeling Report.pdf
└── figures/                                     # Generated figures
    ├── figures_july/                            # Summer forecast plots
    ├── figures_january/                         # Winter forecast plots
    ├── seasonal_comparison_figures/             # Side‑by‑side seasonal comparisons
    └── figures_validation/                      # Numerical model validation plots
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Required packages: `numpy`, `xarray`, `netCDF4`, `scipy`, `matplotlib`, `cartopy`, `scikit-learn`, `pandas`

### Data Acquisition
The project uses **ECMWF ERA5 reanalysis data**. To reproduce the results:

1. Download the required ERA5 subsets from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Place the files in the `data/` directory with the following names:
   - `era5_eastasia_july25.nc`  (summer: 1–4 July 2025)
   - `era5_eastasia_january26.nc`  (winter: 1–4 January 2026)
   - `era5_pde.nc`  (numerical simulation: 1–2 July 2025)
3. Ensure the NetCDF files contain the required variables (`t2m`, `u10`, `v10`, `msl`, `sp`, `z`).

---

## 🔧 Usage

### 1. Statistical Forecasting (Summer / Winter)
- Load ERA5 data for all 14 cities.
- Engineer features (lagged variables, tendency, advection proxy).
- Train Ridge regression models with cross‑validation for 1‑, 6‑, and 12‑hour leads.
- Generate forecast time series, feature importance plots, residual histograms, density scatter plots, and spatial error maps.
- Output figures to `figures_july/` and `figures_january/`.

### 2. Seasonal Comparison
Produces plots (skill degradation, heatmaps, geographic error, diurnal cycles) comparing summer and winter performance. Figures are saved in `seasonal_comparison_figures/`.

### 3. Numerical Model Validation
- Initializes the 2D finite‑difference model with ERA5 data.
- Integrates for 6 hours using adaptive RK4 time stepping.
- Compares final fields with ERA5 "truth" and saves a 6‑panel validation figure in `figures_validation/`.

---

## 📊 Results Summary

| Model            | Lead Time | Mean RMSE (Summer) | Mean RMSE (Winter) | Key Observation |
|------------------|-----------|--------------------|--------------------|-----------------|
| Statistical      | 1h        | 1.28°C             | 1.08°C             | High accuracy for all cities |
| Statistical      | 6h        | 3.30°C             | 3.55°C             | Inland cities worse in summer |
| Statistical      | 12h       | 2.91°C             | 3.59°C             | Coastal cities worse in winter |
| Numerical (PDE)  | 6h        | 4.23°C (global)    | –                  | Reproduces spatial patterns |

Detailed figures and tables are available in the `figures/` directory and the final report.

---

## 📄 Final Report
The final written report (`NWP Mathematical Modeling Report.pdf`) can be found in the repository. It includes:
- Introduction
- Model derivation (statistical + numerical)
- Methodology and data description
- Full results with interpretation
- Discussion of mechanisms, limitations, and future work

---

## 📝 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author
**Yunrong (Eunice) Gu**  
Class of 2027,  
Duke Kunshan University & Duke University  
📧 eunicegu1103@gmail.com  
Project Instructor: Prof. Huaxiong Huang

---

## 🙏 Acknowledgements
- ECMWF for providing the ERA5 reanalysis data.
- The MATH 406 course for guidance and computational resources.
- Dr. Huang for valuable feedback and suggestions.
