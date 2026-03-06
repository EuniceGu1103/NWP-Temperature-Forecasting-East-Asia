# ============================================
# ERA5-Based Winter Temperature Forecasting
# Multi-Step + Multi-City + Physical Diagnosis
# Author: Yunrong (Eunice) Gu
# Course: MATH406 Mathematical Modeling
# University: Duke Kunshan University
# ============================================

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import make_interp_spline

# ====================== Global Configuration ======================
DATA_PATH = "era5_eastasia_january26.nc"
OUTPUT_DIR = "figures_january"
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
results = {}

# ====================== Core Functions ======================
def setup_directories():
    """Create output directory if not exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_and_validate_data():
    """Load ERA5 dataset and filter cities within spatial coverage"""
    ds = xr.open_dataset(DATA_PATH)
    
    # Check spatial range
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    print("=== Dataset Spatial Coverage ===")
    print(f"Latitude: {lat_min} to {lat_max}")
    print(f"Longitude: {lon_min} to {lon_max}\n")
    
    # Filter valid cities
    valid_cities = {}
    for city, (lat, lon) in CITIES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            valid_cities[city] = (lat, lon)
        else:
            print(f"Warning: {city} is outside dataset coverage (skipped)")
    
    return ds, valid_cities

def prepare_dataframe(ds, lat, lon):
    """Extract and preprocess data for a single city (feature engineering)"""
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = point.to_dataframe().reset_index()
    
    # Unit conversion
    df["T"] = df["t2m"] - 273.15  # Kelvin to Celsius
    df["P"] = df["msl"] / 100     # Pascals to hectopascals
    
    # Physical features
    df["dTdt"] = df["T"].diff()                           # Temperature tendency
    df["Adv_T"] = df["u10"] * df["T"].diff() + df["v10"] * df["T"].diff()  # Advection proxy
    
    # Lag features
    df["T_lag1"] = df["T"].shift(1)
    df["T_lag2"] = df["T"].shift(2)
    df["U_lag1"] = df["u10"].shift(1)
    df["V_lag1"] = df["v10"].shift(1)
    df["P_lag1"] = df["P"].shift(1)
    
    return df.dropna().reset_index(drop=True)

def build_multistep_data(df, lead):
    """Build features/target arrays for multi-step forecasting"""
    features = ["T_lag1","T_lag2","U_lag1","V_lag1","P_lag1","dTdt","Adv_T"]
    X = df[features].values
    y = df["T"].shift(-lead).values
    valid = ~np.isnan(y)
    return X[valid], y[valid], features

def train_ridge_model(X, y):
    """Train Ridge regression with cross-validation for hyperparameter tuning"""
    # Train-test split (70/30)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Hyperparameter search
    alphas = np.logspace(-3, 2, 20)
    grid = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)
    
    # Model evaluation
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return mae, rmse, model, y_test, y_pred, grid.best_params_

# ====================== Visualization Functions ======================
def plot_forecast_vs_observed(city, lead, y_test, y_pred):
    """Plot observed vs predicted temperature for a single city/lead time"""
    plt.figure(figsize=(9,4))
    plt.plot(y_test, label="Observed")
    plt.plot(y_pred, "--", label="Predicted")
    plt.title(f"{city} | {lead}h Forecast")
    plt.xlabel("Time Index")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{city}_{lead}h.png", dpi=300)
    plt.close()

def plot_skill_degradation():
    """Plot RMSE degradation across forecast lead times for all cities"""
    plt.figure(figsize=(8,5))
    for city in results.keys():
        rmse_list = [results[city][lt]["rmse"] for lt in FORECAST_HOURS]
        plt.plot(FORECAST_HOURS, rmse_list, marker="o", label=city)
    
    plt.xlabel("Forecast Lead Time (h)")
    plt.ylabel("RMSE (°C)")
    plt.title("Forecast Skill Degradation")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/skill_degradation.png", dpi=300)
    plt.close()

def plot_performance_heatmap():
    """Plot heatmap of RMSE across cities/lead times (red/blue color scheme)"""
    cities = list(results.keys())
    leads = FORECAST_HOURS
    rmse_matrix = np.zeros((len(cities), len(leads)))
    
    for i, city in enumerate(cities):
        for j, lead in enumerate(leads):
            rmse_matrix[i, j] = results[city][lead]["rmse"]
    
    plt.figure(figsize=(9,6))
    plt.imshow(rmse_matrix, cmap="RdBu_r", aspect="auto")
    
    # Add value labels
    for i in range(len(cities)):
        for j in range(len(leads)):
            text_color = "white" if rmse_matrix[i,j] > np.median(rmse_matrix) else "black"
            plt.text(j, i, f"{rmse_matrix[i,j]:.2f}", ha="center", va="center", color=text_color)
    
    plt.xticks(range(len(leads)), [f"{l}h" for l in leads])
    plt.yticks(range(len(cities)), cities)
    plt.colorbar(label="RMSE (°C)")
    plt.title("Forecast Error Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/forecast_heatmap.png", dpi=300)
    plt.close()

def plot_density_scatter(city="Shanghai", lead=6):
    """Plot density scatter of observed vs predicted temperatures"""
    y_true = results[city][lead]["y_test"]
    y_pred = results[city][lead]["y_pred"]
    
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_true[idx], y_pred[idx], z[idx]
    
    plt.figure(figsize=(7,6))
    plt.scatter(x, y, c=z, s=20, cmap="Spectral_r", edgecolors="none")
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(limits, limits, "k--", label="Perfect Fit")
    
    plt.xlabel("Observed Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title(f"{city} {lead}h Density Scatter")
    plt.colorbar(label="Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{city}_density_scatter_{lead}h.png", dpi=300)
    plt.close()

def plot_spatial_error_with_map(target_lead=6):
    """Plot geographic distribution of forecast error with cartopy map"""
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([100, 140, 15, 45], crs=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.25)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Prepare data
    lats, lons, rmses, names = [], [], [], []
    for city, coords in CITIES.items():
        lats.append(coords[0])
        lons.append(coords[1])
        rmses.append(results[city][target_lead]["rmse"])
        names.append(city)
    
    # Plot error bubbles
    sc = ax.scatter(
        lons, lats,
        s=[r**2 * 20 for r in rmses],
        c=rmses,
        cmap="Reds",
        alpha=0.9,
        edgecolors="black",
        transform=ccrs.PlateCarree(),
        zorder=10
    )
    
    # Annotate city names
    for i, name in enumerate(names):
        ax.text(lons[i] + 0.5, lats[i] + 0.5, name, transform=ccrs.PlateCarree(), fontsize=10, weight='bold')
    
    cbar = plt.colorbar(sc, shrink=0.6, pad=0.05)
    cbar.set_label("RMSE (°C)")
    plt.title(f"Geographic Distribution of Forecast Error ({target_lead}h Lead)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/geographic_spatial_error_{target_lead}h.png", dpi=300)
    plt.close()

def plot_multi_city_diurnal_cycle(lead=6):
    """Plot smoothed diurnal error cycle with 95% CI (no individual city lines)"""
    plt.figure(figsize=(10,6), dpi=150)
    hourly_mae_all_cities = []
    hours_full = np.arange(24)
    
    print("\n=== Diurnal Error Summary (Per City) ===")
    for city in CITIES.keys():
        df = prepare_dataframe(ds, CITIES[city][0], CITIES[city][1])
        X, y, _ = build_multistep_data(df, lead)
        
        # Calculate errors
        y_pred = results[city][lead]["y_pred"]
        y_test = results[city][lead]["y_test"]
        errors = np.abs(y_pred - y_test)
        
        # Align time and extract hour
        time_col = [col for col in df.columns if 'time' in col.lower()][0]
        split_idx = int(0.7 * len(X))
        test_time = df[time_col].iloc[split_idx : split_idx + len(y_test)]
        
        df_error = pd.DataFrame({'error': errors, 'time': pd.to_datetime(test_time)})
        df_error['hour'] = df_error['time'].dt.hour
        
        # Calculate hourly MAE
        hourly_mae = df_error.groupby('hour')['error'].mean()
        hourly_mae_24h = pd.Series(index=hours_full, dtype=float)
        for h in hours_full:
            hourly_mae_24h[h] = hourly_mae.get(h, np.nan)
        
        hourly_mae_all_cities.append(hourly_mae_24h.values)
        
        # Print city-level stats
        print(f"{city}:")
        print(f"  Mean MAE: {hourly_mae_24h.mean():.3f}°C")
        print(f"  Max Error Hour: {hourly_mae_24h.idxmax() if not hourly_mae_24h.isna().all() else 'N/A'}")
        print(f"  Min Error Hour: {hourly_mae_24h.idxmin() if not hourly_mae_24h.isna().all() else 'N/A'}")
        print(f"  Diurnal Amplitude: {hourly_mae_24h.max() - hourly_mae_24h.min():.3f}°C" if not hourly_mae_24h.isna().all() else 'N/A')
    
    # Calculate statistical metrics
    hourly_mae_all_cities = np.array(hourly_mae_all_cities)
    mean_mae = np.nanmean(hourly_mae_all_cities, axis=0)
    std_mae = np.nanstd(hourly_mae_all_cities, axis=0)
    n_cities = np.sum(~np.isnan(hourly_mae_all_cities), axis=0)
    
    # Calculate 95% CI
    ci95 = []
    for i in range(24):
        if n_cities[i] >= 2:
            t_val = scipy.stats.t.ppf(0.975, n_cities[i]-1)
            ci = t_val * (std_mae[i] / np.sqrt(n_cities[i]))
        else:
            ci = 0
        ci95.append(ci)
    ci95 = np.array(ci95)
    
    # CI bounds
    ci_lower = np.maximum(mean_mae - ci95, 0)
    ci_upper = mean_mae + ci95
    
    # Smooth curve with spline interpolation
    hours_smooth = np.linspace(0, 23, 200)
    valid_idx = ~np.isnan(mean_mae)
    
    spline_mean = make_interp_spline(hours_full[valid_idx], mean_mae[valid_idx], k=3)
    spline_lower = make_interp_spline(hours_full[valid_idx], ci_lower[valid_idx], k=3)
    spline_upper = make_interp_spline(hours_full[valid_idx], ci_upper[valid_idx], k=3)
    
    mean_mae_smooth = spline_mean(hours_smooth)
    ci_lower_smooth = spline_lower(hours_smooth)
    ci_upper_smooth = spline_upper(hours_smooth)
    
    # Plot
    plt.plot(hours_smooth, mean_mae_smooth, color='#1f77b4', linewidth=3, label='Multi-City Mean MAE')
    plt.fill_between(hours_smooth, ci_lower_smooth, ci_upper_smooth, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f"Multi-City Diurnal Error Cycle ({lead}h Lead)")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Mean Absolute Error (°C)")
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(True, color='gray', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multi_city_diurnal_{lead}h.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print cross-city summary
    print("\n=== Cross-City Diurnal Error Summary ===")
    print(f"Overall Mean MAE: {np.nanmean(mean_mae):.3f}°C")
    print(f"Max Error Hour (Overall): {np.nanargmax(mean_mae)}")
    print(f"Min Error Hour (Overall): {np.nanargmin(mean_mae)}")
    print(f"Diurnal Amplitude: {np.nanmax(mean_mae) - np.nanmin(mean_mae):.3f}°C")

def plot_feature_importance(city="Shanghai", lead=6):
    """Plot Ridge regression coefficients (feature importance) with physical category coloring"""
    model = results[city][lead]["model"]
    features = results[city][lead]["features"]
    coefs = model.coef_
    
    # Print coefficient values (critical for paper)
    print(f"\n=== Feature Importance (Coefficients) - {city} ({lead}h) ===")
    for feat, coef in zip(features, coefs):
        print(f"{feat}: {coef:.4f}")
    
    # Color coding by feature type
    colors = []
    for f in features:
        if f in ["dTdt", "Adv_T"]:
            colors.append("#FF6B6B")  # Red = Physical terms
        elif "P" in f:
            colors.append("#4ECDC4")  # Teal = Pressure
        else:
            colors.append("#45B7D1")  # Blue = Lag/Inertia
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(features, coefs, color=colors)
    plt.title(f"Feature Importance (Ridge Coefficients) - {city}")
    plt.ylabel("Coefficient Value")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#FF6B6B', lw=4),
        Line2D([0], [0], color='#4ECDC4', lw=4),
        Line2D([0], [0], color='#45B7D1', lw=4)
    ]
    plt.legend(custom_lines, ['Physical Tendency', 'Pressure', 'Inertia/Lag'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{city}_feature_importance_{lead}h.png", dpi=300)
    plt.close()

def plot_residual_analysis(city="Shanghai", lead=6):
    """Plot residual histogram with normal fit (check for systematic bias)"""
    y_pred = results[city][lead]["y_pred"]
    y_test = results[city][lead]["y_test"]
    residuals = y_pred - y_test
    
    # Calculate residual stats (critical for paper)
    mu, std = norm.fit(residuals)
    print(f"\n=== Residual Analysis - {city} ({lead}h) ===")
    print(f"Mean Residual (Bias): {mu:.4f}°C")
    print(f"Residual Std Dev: {std:.4f}°C")
    print(f"Residual Range: {residuals.min():.4f} to {residuals.max():.4f}°C")
    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.4f}°C")
    
    # Plot
    plt.figure(figsize=(7, 5))
    n, bins, patches = plt.hist(residuals, bins=30, density=True, facecolor='#2A4C71', alpha=0.7, edgecolor='white')
    
    # Normal fit curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
    
    plt.title(f"Residual Analysis: {city} ({lead}h Lead)")
    plt.xlabel("Error (Predicted - Observed) [°C]")
    plt.ylabel("Probability Density")
    plt.axvline(0, color='red', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{city}_residuals_{lead}h.png", dpi=300)
    plt.close()

def print_cross_city_skill_summary():
    """Print comprehensive cross-city performance metrics"""
    print("\n=== Cross-City Forecast Skill Summary ===")
    for lead in FORECAST_HOURS:
        rmses = [results[city][lead]["rmse"] for city in CITIES.keys()]
        maes = [results[city][lead]["mae"] for city in CITIES.keys()]
        cities = list(CITIES.keys())
        
        # Calculate stats
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        best_city = cities[np.argmin(rmses)]
        worst_city = cities[np.argmax(rmses)]
        
        # Print
        print(f"\nLead Time: {lead}h")
        print(f"Mean RMSE: {mean_rmse:.3f}°C (Std: {std_rmse:.3f})")
        print(f"Mean MAE: {mean_mae:.3f}°C (Std: {std_mae:.3f})")
        print(f"Best City: {best_city} | RMSE = {min(rmses):.3f}°C | MAE = {maes[np.argmin(rmses)]:.3f}°C")
        print(f"Worst City: {worst_city} | RMSE = {max(rmses):.3f}°C | MAE = {maes[np.argmax(rmses)]:.3f}°C")

def plot_multi_city_density_scatter(lead=6):
    """Plot combined density scatter for all cities"""
    plt.figure(figsize=(7,7))
    all_true, all_pred = [], []
    
    for city in CITIES.keys():
        all_true.extend(results[city][lead]["y_test"])
        all_pred.extend(results[city][lead]["y_pred"])
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # Calculate global stats
    corr = np.corrcoef(all_true, all_pred)[0,1]
    bias = np.mean(all_pred - all_true)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    
    print(f"\n=== Global Model Performance ({lead}h) ===")
    print(f"Correlation Coefficient: {corr:.3f}")
    print(f"Mean Bias: {bias:.3f}°C")
    print(f"Global RMSE: {rmse:.3f}°C")
    
    # Plot
    plt.scatter(all_true, all_pred, alpha=0.3, s=10)
    limits = [min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())]
    plt.plot(limits, limits, 'k--', linewidth=2)
    
    plt.title(f"All-City Prediction Scatter ({lead}h Lead)")
    plt.xlabel("Observed (°C)")
    plt.ylabel("Predicted (°C)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multi_city_scatter_{lead}h.png", dpi=300)
    plt.close()

# ====================== Main Execution Pipeline ======================
if __name__ == "__main__":
    
    setup_directories()
    ds, CITIES = load_and_validate_data()
    
    for city, (lat, lon) in CITIES.items():
        print(f"\nProcessing: {city}")
        df = prepare_dataframe(ds, lat, lon)
        results[city] = {}
        
        for lead in FORECAST_HOURS:
            print(f"  Lead Time: {lead}h")
            X, y, features = build_multistep_data(df, lead)
            mae, rmse, model, y_test, y_pred, best_params = train_ridge_model(X, y)
            
            results[city][lead] = {
                "mae": mae,
                "rmse": rmse,
                "y_test": y_test,
                "y_pred": y_pred,
                "features": features,
                "model": model,
                "best_alpha": best_params["alpha"]
            }
            
            # Print training stats
            print(f"    MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | Best Alpha: {best_params['alpha']:.4f}")
            
            # Plot observed vs predicted
            plot_forecast_vs_observed(city, lead, y_test, y_pred)
    
    plot_skill_degradation()
    plot_performance_heatmap()
    plot_density_scatter("Shanghai", 6)
    
    plot_spatial_error_with_map(6)
    plot_multi_city_diurnal_cycle(6)
    plot_feature_importance("Shanghai", 6)
    plot_residual_analysis("Shanghai", 6)
    print_cross_city_skill_summary()
    plot_multi_city_density_scatter(6)
    
    print(f"Results and figures saved to: {OUTPUT_DIR}")