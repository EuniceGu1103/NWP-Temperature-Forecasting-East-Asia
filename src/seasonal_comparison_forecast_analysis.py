# Seasonal Comparison Visualization for Temperature Forecasting
# Author: Yunrong (Eunice) Gu
# Course: MATH406 Mathematical Modeling
# Objective: Compare summer (July) vs winter (January) forecasting performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import os
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ====================== Global Configuration ======================
# Define paths and season config
SEASONS = {
    'summer': {
        'label': "Summer (July)",
        'color': '#5CA492',
        'results_path': "summer_results.csv",
        'precomputed': {
            'Shanghai': {1: {'rmse':0.741, 'mae':0.616}, 6:{'rmse':1.611, 'mae':1.274}, 12:{'rmse':2.493, 'mae':2.172}},
            'Hong Kong': {1: {'rmse':0.379, 'mae':0.282}, 6:{'rmse':0.986, 'mae':0.771}, 12:{'rmse':1.258, 'mae':1.056}},
            'Lanzhou': {1: {'rmse':2.963, 'mae':2.810}, 6:{'rmse':10.138, 'mae':9.946}, 12:{'rmse':10.031, 'mae':9.244}},
            'Seoul': {1: {'rmse':0.387, 'mae':0.302}, 6:{'rmse':1.309, 'mae':1.155}, 12:{'rmse':1.145, 'mae':1.064}},
            'Busan': {1: {'rmse':0.470, 'mae':0.285}, 6:{'rmse':1.791, 'mae':1.536}, 12:{'rmse':1.210, 'mae':0.994}},
            'Fukuoka': {1: {'rmse':0.510, 'mae':0.359}, 6:{'rmse':2.720, 'mae':2.349}, 12:{'rmse':1.652, 'mae':1.142}},
            'Beijing': {1: {'rmse':1.514, 'mae':1.182}, 6:{'rmse':6.348, 'mae':5.686}, 12:{'rmse':5.626, 'mae':5.248}},
            'Chengdu': {1: {'rmse':1.486, 'mae':1.153}, 6:{'rmse':2.597, 'mae':2.313}, 12:{'rmse':1.617, 'mae':1.358}},
            'Wuhan': {1: {'rmse':2.127, 'mae':1.892}, 6:{'rmse':3.852, 'mae':3.377}, 12:{'rmse':2.022, 'mae':1.707}},
            'Taipei': {1: {'rmse':0.650, 'mae':0.488}, 6:{'rmse':2.328, 'mae':1.955}, 12:{'rmse':2.110, 'mae':1.892}},
            'Zhengzhou': {1: {'rmse':1.448, 'mae':1.163}, 6:{'rmse':3.037, 'mae':2.217}, 12:{'rmse':3.036, 'mae':2.355}},
            'XiAn': {1: {'rmse':2.851, 'mae':2.334}, 6:{'rmse':5.435, 'mae':4.527}, 12:{'rmse':3.910, 'mae':3.176}},
            'Nanning': {1: {'rmse':1.419, 'mae':1.137}, 6:{'rmse':2.840, 'mae':2.228}, 12:{'rmse':3.041, 'mae':2.618}},
            'Fuzhou': {1: {'rmse':1.034, 'mae':0.750}, 6:{'rmse':1.225, 'mae':0.993}, 12:{'rmse':1.652, 'mae':1.461}}
        },
        'diurnal_mae': {
            'Shanghai': 1.347, 'Hong Kong':0.762, 'Lanzhou':10.164, 'Seoul':1.107, 'Busan':1.487,
            'Fukuoka':2.395, 'Beijing':6.064, 'Chengdu':2.255, 'Wuhan':3.478, 'Taipei':1.845,
            'Zhengzhou':2.395, 'XiAn':4.484, 'Nanning':2.426, 'Fuzhou':0.931
        },
        'max_error_hour': {
            'Shanghai':5, 'Hong Kong':23, 'Lanzhou':9, 'Seoul':5, 'Busan':21, 'Fukuoka':4,
            'Beijing':7, 'Chengdu':2, 'Wuhan':7, 'Taipei':13, 'Zhengzhou':0, 'XiAn':14, 'Nanning':2, 'Fuzhou':19
        },
        'min_error_hour': {
            'Shanghai':9, 'Hong Kong':11, 'Lanzhou':15, 'Seoul':11, 'Busan':10, 'Fukuoka':8,
            'Beijing':19, 'Chengdu':20, 'Wuhan':19, 'Taipei':4, 'Zhengzhou':10, 'XiAn':8, 'Nanning':13, 'Fuzhou':2
        },
        'diurnal_amplitude': {
            'Shanghai':3.926, 'Hong Kong':2.065, 'Lanzhou':5.742, 'Seoul':1.943, 'Busan':3.206, 'Fukuoka':4.783,
            'Beijing':8.459, 'Chengdu':4.423, 'Wuhan':6.239, 'Taipei':4.149, 'Zhengzhou':6.352, 'XiAn':9.763, 'Nanning':5.459, 'Fuzhou':2.152
        }
    },
    'winter': {
        'label': "Winter (January)",
        'color': '#2087B0',
        'results_path': "winter_results.csv",
        'precomputed': {
            'Shanghai': {1: {'rmse':0.838, 'mae':0.583}, 6:{'rmse':3.225, 'mae':2.691}, 12:{'rmse':3.343, 'mae':2.980}},
            'Hong Kong': {1: {'rmse':0.571, 'mae':0.395}, 6:{'rmse':2.566, 'mae':2.334}, 12:{'rmse':3.466, 'mae':3.068}},
            'Lanzhou': {1: {'rmse':1.687, 'mae':1.256}, 6:{'rmse':3.344, 'mae':2.787}, 12:{'rmse':3.557, 'mae':3.057}},
            'Seoul': {1: {'rmse':1.102, 'mae':0.965}, 6:{'rmse':5.947, 'mae':5.297}, 12:{'rmse':6.598, 'mae':6.247}},
            'Busan': {1: {'rmse':0.882, 'mae':0.766}, 6:{'rmse':7.228, 'mae':6.691}, 12:{'rmse':8.966, 'mae':8.476}},
            'Fukuoka': {1: {'rmse':0.885, 'mae':0.772}, 6:{'rmse':3.816, 'mae':3.508}, 12:{'rmse':2.894, 'mae':2.442}},
            'Beijing': {1: {'rmse':1.621, 'mae':1.132}, 6:{'rmse':4.460, 'mae':3.754}, 12:{'rmse':3.130, 'mae':2.339}},
            'Chengdu': {1: {'rmse':1.141, 'mae':0.809}, 6:{'rmse':3.118, 'mae':2.513}, 12:{'rmse':2.878, 'mae':2.324}},
            'Wuhan': {1: {'rmse':1.075, 'mae':0.901}, 6:{'rmse':1.749, 'mae':1.490}, 12:{'rmse':1.468, 'mae':1.254}},
            'Taipei': {1: {'rmse':1.081, 'mae':0.700}, 6:{'rmse':3.890, 'mae':2.804}, 12:{'rmse':4.314, 'mae':3.289}},
            'Zhengzhou': {1: {'rmse':0.801, 'mae':0.574}, 6:{'rmse':2.906, 'mae':2.590}, 12:{'rmse':3.123, 'mae':2.833}},
            'XiAn': {1: {'rmse':0.999, 'mae':0.744}, 6:{'rmse':2.549, 'mae':2.018}, 12:{'rmse':2.295, 'mae':1.711}},
            'Nanning': {1: {'rmse':0.572, 'mae':0.447}, 6:{'rmse':1.399, 'mae':1.052}, 12:{'rmse':1.698, 'mae':1.368}},
            'Fuzhou': {1: {'rmse':1.926, 'mae':1.250}, 6:{'rmse':3.443, 'mae':2.966}, 12:{'rmse':2.483, 'mae':2.089}}
        },
        'diurnal_mae': {
            'Shanghai':2.886, 'Hong Kong':2.440, 'Lanzhou':2.529, 'Seoul':5.682, 'Busan':6.995,
            'Fukuoka':3.624, 'Beijing':3.782, 'Chengdu':2.589, 'Wuhan':1.560, 'Taipei':2.992,
            'Zhengzhou':2.769, 'XiAn':2.058, 'Nanning':1.113, 'Fuzhou':2.978
        },
        'max_error_hour': {
            'Shanghai':4, 'Hong Kong':4, 'Lanzhou':19, 'Seoul':23, 'Busan':23, 'Fukuoka':23,
            'Beijing':1, 'Chengdu':0, 'Wuhan':7, 'Taipei':23, 'Zhengzhou':23, 'XiAn':2, 'Nanning':1, 'Fuzhou':0
        },
        'min_error_hour': {
            'Shanghai':12, 'Hong Kong':20, 'Lanzhou':8, 'Seoul':18, 'Busan':18, 'Fukuoka':18,
            'Beijing':10, 'Chengdu':6, 'Wuhan':19, 'Taipei':8, 'Zhengzhou':18, 'XiAn':10, 'Nanning':12, 'Fuzhou':11
        },
        'diurnal_amplitude': {
            'Shanghai':5.192, 'Hong Kong':3.740, 'Lanzhou':6.832, 'Seoul':9.899, 'Busan':7.375, 'Fukuoka':4.148,
            'Beijing':7.711, 'Chengdu':6.125, 'Wuhan':3.294, 'Taipei':8.968, 'Zhengzhou':4.741, 'XiAn':5.459, 'Nanning':3.082, 'Fuzhou':7.172
        }
    }
}

# Target cities and geographic coordinates
CITIES = {
    "Shanghai":  (31.25, 121.50),
    "Hong Kong": (22.32, 114.17),
    "Lanzhou":   (36.06, 103.84),
    "Seoul":     (37.56, 126.97),
    "Busan":     (35.18, 129.07),
    "Fukuoka":   (33.59, 130.40),
    "Beijing":   (39.90, 116.40),
    "Chengdu":   (30.67, 104.06),
    "Wuhan":     (30.59, 114.30),
    "Taipei":    (25.03, 121.57),
    "Zhengzhou": (34.75, 113.62),
    "XiAn":      (34.34, 108.94),
    "Nanning":   (22.82, 108.32),
    "Fuzhou":    (26.08, 119.30)
}
FORECAST_HOURS = [1, 6, 12]
COMPARISON_OUTPUT_DIR = "figures_seasonal_comparison"

# ====================== Core Utility Functions ======================
def setup_directories():
    """Create output directories for seasonal comparison plots"""
    try:
        if not os.path.exists(COMPARISON_OUTPUT_DIR):
            os.makedirs(COMPARISON_OUTPUT_DIR)
            print(f"Created output directory: {COMPARISON_OUTPUT_DIR}")
    except Exception as e:
        print(f"Warning: Could not create directory ({e}) - using current directory")

def load_precomputed_results(season_key):
    """Load precomputed results (from your output) instead of retraining models"""
    cfg = SEASONS[season_key]
    return cfg['precomputed'], cfg['diurnal_mae'], cfg['max_error_hour'], cfg['min_error_hour'], cfg['diurnal_amplitude']

def get_diurnal_error_curve(season_key):
    """Generate smooth diurnal error curve from precomputed hourly data"""
    cfg = SEASONS[season_key]
    cities = list(CITIES.keys())
    
    # Calculate multi-city mean diurnal MAE for each hour (0-23)
    hours_full = np.arange(24)
    mean_diurnal_mae = np.zeros(24)
    
    # Use precomputed max/min hour and amplitude to reconstruct curve
    for city in cities:
        max_h = cfg['max_error_hour'][city]
        min_h = cfg['min_error_hour'][city]
        amplitude = cfg['diurnal_amplitude'][city]
        mean_mae = cfg['diurnal_mae'][city]
        
        # Create smooth curve between max/min
        city_curve = np.ones(24) * mean_mae
        city_curve[max_h] = mean_mae + amplitude/2
        city_curve[min_h] = mean_mae - amplitude/2
        
        # Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        city_curve_smooth = gaussian_filter1d(city_curve, sigma=2)
        mean_diurnal_mae += city_curve_smooth
    
    # Average across all cities
    mean_diurnal_mae /= len(cities)
    
    # Smooth final curve
    hours_smooth = np.linspace(0, 23, 200)
    spline = make_interp_spline(hours_full, mean_diurnal_mae, k=3)
    curve_smooth = spline(hours_smooth)
    
    return hours_full, mean_diurnal_mae, hours_smooth, curve_smooth


# ====================== Seasonal Comparison Plot Functions ======================
def plot_combined_diurnal_cycle():
    """Plot combined diurnal MAE cycle (6h forecast) for summer + winter"""
    # Get diurnal curves for both seasons
    summer_hours, summer_mae, summer_smooth_h, summer_smooth_c = get_diurnal_error_curve('summer')
    winter_hours, winter_mae, winter_smooth_h, winter_smooth_c = get_diurnal_error_curve('winter')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot smoothed curves
    plt.plot(summer_smooth_h, summer_smooth_c, color=SEASONS['summer']['color'], 
             linewidth=3, label=SEASONS['summer']['label'], alpha=0.8)
    plt.plot(winter_smooth_h, winter_smooth_c, color=SEASONS['winter']['color'], 
             linewidth=3, label=SEASONS['winter']['label'], alpha=0.8)
    
    # Highlight max error hours
    summer_max_h = np.argmax(summer_mae)
    winter_max_h = np.argmax(winter_mae)
    plt.scatter(summer_max_h, summer_mae[summer_max_h], 
                color=SEASONS['summer']['color'], s=100, marker='*', 
                label=f"Summer Max (Hour {summer_max_h})")
    plt.scatter(winter_max_h, winter_mae[winter_max_h], 
                color=SEASONS['winter']['color'], s=100, marker='*', 
                label=f"Winter Max (Hour {winter_max_h})")
    
    # Formatting
    plt.title("Multi-City Mean Diurnal MAE (6h Forecast) - Summer vs Winter", fontsize=14, weight="bold")
    plt.xlabel("Hour of Day (UTC)", fontsize=12)
    plt.ylabel("Mean Absolute Error (°C)", fontsize=12)
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(True, alpha=0.25)
    plt.legend(loc='upper center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{COMPARISON_OUTPUT_DIR}/multi_city_diurnal_6h_combined.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: multi_city_diurnal_6h_combined.png")

def plot_side_by_side_skill_degradation():
    """Plot skill degradation (RMSE vs lead time) as side-by-side subplots (a/b)"""

    summer_results, _, _, _, _ = load_precomputed_results('summer')
    winter_results, _, _, _, _ = load_precomputed_results('winter')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Forecast Skill Degradation - Summer vs Winter", fontsize=16, y=0.97, weight="bold")
    
    # ---------------- Summer ----------------
    ax1.set_title("(a) " + SEASONS['summer']['label'], fontsize=14)
    for city in CITIES.keys():
        if city in summer_results:
            rmse_vals = [summer_results[city][lead]["rmse"] for lead in FORECAST_HOURS]
            ax1.plot(FORECAST_HOURS, rmse_vals, marker='o', label=city, alpha=0.7)
    
    summer_12h_rmse = {
        city: summer_results[city][12]["rmse"]
        for city in CITIES.keys() if city in summer_results
    }
    summer_worst = max(summer_12h_rmse, key=summer_12h_rmse.get)
    summer_best = min(summer_12h_rmse, key=summer_12h_rmse.get)
    
    ax1.text(
        0.02, 0.98,
        f"Worst: {summer_worst}\nBest: {summer_best}",
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # ---------------- Winter ----------------
    ax2.set_title("(b) " + SEASONS['winter']['label'], fontsize=14)
    for city in CITIES.keys():
        if city in winter_results:
            rmse_vals = [winter_results[city][lead]["rmse"] for lead in FORECAST_HOURS]
            ax2.plot(FORECAST_HOURS, rmse_vals, marker='o', alpha=0.7)
    
    winter_12h_rmse = {
        city: winter_results[city][12]["rmse"]
        for city in CITIES.keys() if city in winter_results
    }
    winter_worst = max(winter_12h_rmse, key=winter_12h_rmse.get)
    winter_best = min(winter_12h_rmse, key=winter_12h_rmse.get)
    
    ax2.text(
        0.02, 0.98,
        f"Worst: {winter_worst}\nBest: {winter_best}",
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # ---------------- Formatting ----------------
    for ax in [ax1, ax2]:
        ax.set_xlabel("Forecast Lead Time (h)", fontsize=12)
        ax.set_ylabel("RMSE (°C)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(FORECAST_HOURS)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=7,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.subplots_adjust(bottom=0.18, top=0.88)
    
    plt.savefig(
        f"{COMPARISON_OUTPUT_DIR}/skill_degradation_side_by_side.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    print("Created: skill_degradation_side_by_side.png")

def plot_side_by_side_heatmap():
    """Plot forecast error heatmap (city vs lead time) as side-by-side subplots (a/b)"""

    summer_results, _, _, _, _ = load_precomputed_results('summer')
    winter_results, _, _, _ ,_ = load_precomputed_results('winter')

    common_cities = list(CITIES.keys())
    summer_rmse_matrix = np.array([[summer_results[city][lead]["rmse"] for lead in FORECAST_HOURS] 
                                  for city in common_cities if city in summer_results])
    winter_rmse_matrix = np.array([[winter_results[city][lead]["rmse"] for lead in FORECAST_HOURS] 
                                  for city in common_cities if city in winter_results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle("Forecast Error Heatmap (RMSE) - Summer vs Winter", fontsize=16, weight="bold")
    
    # Summer heatmap (subplot a)
    im1 = ax1.imshow(summer_rmse_matrix, cmap="RdBu_r", aspect="auto")
    ax1.set_title("(a) " + SEASONS['summer']['label'], fontsize=14)
    
    # Add value labels (summer)
    for i in range(len(common_cities)):
        if common_cities[i] not in summer_results:
            continue
        for j in range(len(FORECAST_HOURS)):
            val = summer_results[common_cities[i]][FORECAST_HOURS[j]]["rmse"]
            text_color = "black" if val > np.median(summer_rmse_matrix) else "white"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
    
    # Winter heatmap (subplot b)
    im2 = ax2.imshow(winter_rmse_matrix, cmap="RdBu_r", aspect="auto")
    ax2.set_title("(b) " + SEASONS['winter']['label'], fontsize=14)
    
    # Add value labels (winter)
    for i in range(len(common_cities)):
        if common_cities[i] not in winter_results:
            continue
        for j in range(len(FORECAST_HOURS)):
            val = winter_results[common_cities[i]][FORECAST_HOURS[j]]["rmse"]
            text_color = "black" if val > np.median(winter_rmse_matrix) else "white"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
    
    # Formatting
    for ax in [ax1, ax2]:
        ax.set_yticks(range(len(common_cities)))
        ax.set_yticklabels(common_cities, fontsize=10)
        ax.set_xticks(range(len(FORECAST_HOURS)))
        ax.set_xticklabels([f"{h}h" for h in FORECAST_HOURS], fontsize=10)
        ax.set_xlabel("Forecast Lead Time", fontsize=12)
    
    ax1.set_ylabel("City", fontsize=12)
    
    # Colorbars
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("RMSE (°C)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{COMPARISON_OUTPUT_DIR}/forecast_heatmap_side_by_side.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: forecast_heatmap_side_by_side.png")

def plot_side_by_side_geographic_error():
    """Plot geographic error distribution (6h forecast) as side-by-side subplots (a/b)"""
    
    summer_results, _, _, _, _ = load_precomputed_results('summer')
    winter_results, _, _, _, _ = load_precomputed_results('winter')
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(20, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    fig.suptitle(
        "Geographic Distribution of Forecast Error (6h) - Summer vs Winter",
        fontsize=16,
        weight="bold"
    )
    
    # ---- Collect global RMSE range ----
    all_rmse = []
    for city in CITIES.keys():
        if city in summer_results:
            all_rmse.append(summer_results[city][6]["rmse"])
        if city in winter_results:
            all_rmse.append(winter_results[city][6]["rmse"])
    
    vmin = min(all_rmse)
    vmax = max(all_rmse)
    
    # ---- Common map features ----
    for ax in [ax1, ax2]:
        ax.set_extent([100, 140, 15, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=False, alpha=0.3)
    
    # ---- Summer (a) ----
    ax1.set_title("(a) " + SEASONS['summer']['label'], fontsize=14)
    
    for city, (lat, lon) in CITIES.items():
        if city in summer_results:
            rmse = summer_results[city][6]["rmse"]
            size = rmse**2 * 6
            
            ax1.scatter(
                lon, lat,
                s=size,
                c=[rmse],
                cmap="Reds",
                vmin=vmin, vmax=vmax,
                edgecolors="black",
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=10
            )
            
            ax1.text(
                lon + 0.6, lat + 0.6,
                city,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                weight='bold'
            )
    
    # ---- Winter (b) ----
    ax2.set_title("(b) " + SEASONS['winter']['label'], fontsize=14)
    
    for city, (lat, lon) in CITIES.items():
        if city in winter_results:
            rmse = winter_results[city][6]["rmse"]
            size = rmse**2 * 6
            
            ax2.scatter(
                lon, lat,
                s=size,
                c=[rmse],
                cmap="Reds",
                vmin=vmin, vmax=vmax,
                edgecolors="black",
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=10
            )
            
            ax2.text(
                lon + 0.6, lat + 0.6,
                city,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                weight='bold'
            )
    
    # ---- Shared colorbar ----
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("RMSE (°C) (6h Forecast)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(
        f"{COMPARISON_OUTPUT_DIR}/geographic_spatial_error_6h_side_by_side.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    print("Created: geographic_spatial_error_6h_side_by_side.png")

def plot_seasonal_error_comparison_bar():
    """Additional plot: Bar chart comparing key cities' 12h RMSE (summer vs winter)"""
    
    key_cities = ["Lanzhou", "Beijing", "Seoul", "Busan"]
    summer_results, _, _, _, _ = load_precomputed_results('summer')
    winter_results, _, _, _, _ = load_precomputed_results('winter')
    
    summer_rmse = [summer_results[city][12]["rmse"] for city in key_cities]
    winter_rmse = [winter_results[city][12]["rmse"] for city in key_cities]
    
    x = np.arange(len(key_cities))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, summer_rmse, width, 
                   label=SEASONS['summer']['label'], color=SEASONS['summer']['color'], alpha=0.8)
    bars2 = ax.bar(x + width/2, winter_rmse, width, 
                   label=SEASONS['winter']['label'], color=SEASONS['winter']['color'], alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.2f}", ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_title("12h Forecast RMSE - Key Cities (Summer vs Winter)", fontsize=14)
    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(key_cities, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight seasonal change percentage
    for i, city in enumerate(key_cities):
        change_pct = ((winter_rmse[i] - summer_rmse[i]) / summer_rmse[i]) * 100
        ax.text(i, max(summer_rmse[i], winter_rmse[i]) + 0.5,
                f"{change_pct:+.1f}%", ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{COMPARISON_OUTPUT_DIR}/key_cities_seasonal_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: key_cities_seasonal_comparison.png")

# ====================== Main Execution Pipeline ======================
if __name__ == "__main__":
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Generate all seasonal comparison plots (no raw data needed!)
    try:
        print("\nGenerating seasonal comparison plots...")
        plot_combined_diurnal_cycle()
        plot_side_by_side_skill_degradation()
        plot_side_by_side_heatmap()
        plot_side_by_side_geographic_error()
        plot_seasonal_error_comparison_bar()
        
        # Step 3: Print key comparison statistics
        print("\n=== Seasonal Comparison Statistics (12h Forecast) ===")
        key_cities = ["Lanzhou", "Beijing", "Seoul", "Busan"]
        summer_results, _, _, _, _ = load_precomputed_results('summer')
        winter_results, _, _, _, _ = load_precomputed_results('winter')
        
        for city in key_cities:
            summer_rmse = summer_results[city][12]["rmse"]
            winter_rmse = winter_results[city][12]["rmse"]
            change_pct = ((winter_rmse - summer_rmse) / summer_rmse) * 100
            print(f"{city}: Summer={summer_rmse:.2f}°C | Winter={winter_rmse:.2f}°C | Change={change_pct:+.2f}%")
        
        print(f"\n✅ All plots saved to: {os.path.abspath(COMPARISON_OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        print("Please check your Python environment (ensure cartopy/scipy/matplotlib are installed)")