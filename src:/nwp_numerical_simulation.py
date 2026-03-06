"""
NWP Model with ERA5 Initialization
6‑hour forecast validation against ERA5
Author: Yunrong (Eunice) Gu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Parameter settings
# ============================================
DATA_PATH = "era5_pde.nc"
OUTPUT_DIR = "figures_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
g = 9.81
rho0 = 1.2  # reference air density (kg/m³)
H0 = 10000.0  # reference geopotential height (m)
kappa = 2e-6  # thermal diffusivity
nu = 2e-6  # viscosity
Omega = 7.292e-5  # Earth angular velocity
alpha = 3e-4  # thermal expansion coefficient (K^{-1})
tau = 12 * 3600  # Newtonian cooling time scale (s)

# Grid configuration
lon_min, lon_max = 100, 140
lat_min, lat_max = 15, 50
Nx = 120                                     # number of grid points in x
Ny = 120                                     # number of grid points in y
lon = np.linspace(lon_min, lon_max, Nx)
lat = np.linspace(lat_min, lat_max, Ny)
dx = (lon_max - lon_min) * 111e3 / Nx        # approximate grid spacing in metres
dy = (lat_max - lat_min) * 111e3 / Ny
Lon, Lat = np.meshgrid(lon, lat)

# Coriolis parameter
f = 2 * Omega * np.sin(np.deg2rad(Lat))

# Simulation parameters
start_date = "2025-07-01T00:00:00"            # start time (must match data format)
target_lead = 6                               # forecast lead time (hours)
Nt = 200                                      # target number of time steps (for CFL estimate)
save_every = 2                                # save a frame every `save_every` steps (for animation)

# ============================================
# Finite difference operators
# ============================================
def ddx(f):
    """Central difference in x‑direction (periodic boundary conditions)."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)

def ddy(f):
    """Central difference in y‑direction (periodic boundary conditions)."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dy)

def laplacian(f):
    """Laplacian operator (periodic boundaries)."""
    return (
        (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dx**2
      + (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dy**2
    )

# ============================================
# Load ERA5 data and interpolate
# ============================================
print("Loading ERA5 data...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

ds = xr.open_dataset(DATA_PATH)
print("Dataset coordinates:", list(ds.coords))
print("Dataset dimensions:", ds.dims)

# Automatically detect time coordinate name
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Construct time strings (assuming format '2025-07-01T00:00:00')
init_time_str = start_date
valid_time_str = f"2025-07-01T{target_lead:02d}:00:00"

# Extract initial fields
t2m_init = ds["t2m"].sel({time_coord: init_time_str}).values - 273.15
u10_init = ds["u10"].sel({time_coord: init_time_str}).values
v10_init = ds["v10"].sel({time_coord: init_time_str}).values
msl_init = ds["msl"].sel({time_coord: init_time_str}).values / 100   # Pa -> hPa
sp_init  = ds["sp"].sel({time_coord: init_time_str}).values          # Pa
z_init   = ds["z"].sel({time_coord: init_time_str}).values           # m²/s²

# Extract verification fields (truth)
t2m_valid = ds["t2m"].sel({time_coord: valid_time_str}).values - 273.15
u10_valid = ds["u10"].sel({time_coord: valid_time_str}).values
v10_valid = ds["v10"].sel({time_coord: valid_time_str}).values
msl_valid = ds["msl"].sel({time_coord: valid_time_str}).values / 100

# ERA5 lat/lon grids
era_lat = ds.latitude.values
era_lon = ds.longitude.values
ELon, ELat = np.meshgrid(era_lon, era_lat)
points = np.column_stack([ELon.ravel(), ELat.ravel()])

def interp(field):
    """Bilinear interpolation from ERA5 grid to model grid.
       Out‑of‑bounds points are filled with nearest neighbour, then NaN → 0."""
    result = griddata(points, field.ravel(), (Lon, Lat), method='linear')
    result[np.isnan(result)] = 0.0
    return result

# Interpolate to model grid
T = interp(t2m_init)
u = interp(u10_init)
v = interp(v10_init)

# Compute equivalent height h (m) using surface pressure and orography
# h = p/(ρg) + terrain height
sp_hPa = interp(sp_init / 100)             # surface pressure in hPa
z_height = interp(z_init) / g              # geopotential height (m)
h = sp_hPa * 100 / (rho0 * g) + z_height   # convert hPa back to Pa

# Reference temperature (for buoyancy)
T_ref = np.nanmean(T)
T_clim = T_ref

# Save initial fields for diagnostics
T_init = T.copy()
u_init = u.copy()
v_init = v.copy()
h_init = h.copy()

print("Initialization complete.")

# ============================================
# Adaptive time step (CFL condition)
# ============================================
def compute_dt(u, v, frame):
    vmax = np.nanmax(np.sqrt(u**2 + v**2))
    if not np.isfinite(vmax) or vmax < 1.0:
        vmax = 10.0
    cfl_safety = 0.6 + 0.2 * np.tanh(frame / 100)
    dt_adv = cfl_safety * min(dx, dy) / vmax
    dt_diff = 0.3 * min(dx, dy)**2 / max(nu, 1e-6)
    target_dt = (target_lead * 3600) / Nt   # aim for about Nt steps
    return min(dt_adv, dt_diff, target_dt, 120)

# ============================================
# Right‑hand side (thermally coupled Boussinesq)
# ============================================
def RHS(u, v, h, T, frame, dt):
    # Buoyancy terms
    T_anom = T - T_ref
    buoy_x = -g * alpha * T_anom * ddx(h)
    buoy_y = -g * alpha * T_anom * ddy(h)

    # Momentum equations
    du = -u * ddx(u) - v * ddy(u) - g * ddx(h) + f * v + nu * laplacian(u) + buoy_x
    dv = -u * ddx(v) - v * ddy(v) - g * ddy(h) - f * u + nu * laplacian(v) + buoy_y

    # Continuity equation
    dh = -(ddx(u * h) + ddy(v * h))

    # Temperature equation
    dT = -u * ddx(T) - v * ddy(T) + kappa * laplacian(T)

    # Newtonian cooling (long‑wave radiation parameterisation)
    dT += - (T - T_clim) / tau

    # Simplified short‑wave heating
    local_hour = (frame * dt / 3600) % 24
    solar_angle = np.cos(2 * np.pi * (local_hour - 12) / 24)
    solar_angle = np.maximum(solar_angle, 0.0)
    Q_solar = 0.003 * solar_angle * np.sin(np.pi * (Lat - 15) / 35)   # latitude dependence
    dT += Q_solar

    # Wind‑speed‑dependent turbulent mixing
    wind_speed = np.sqrt(u**2 + v**2)
    K_mix = 1e-5 * wind_speed
    dT += K_mix * laplacian(T)

    # Sponge layer (only near boundaries)
    damping = 0.005
    sponge = (np.exp(-((Lon - 105) / 3)**2) + np.exp(-((Lon - 135) / 3)**2) +
              np.exp(-((Lat - 20) / 3)**2) + np.exp(-((Lat - 45) / 3)**2))
    sponge = np.clip(sponge, 0, 1)
    du -= damping * sponge * u
    dv -= damping * sponge * v

    return du, dv, dh, dT

# ============================================
# RK4 integrator
# ============================================
def RK4(u, v, h, T, dt, frame):
    k1u, k1v, k1h, k1T = RHS(u, v, h, T, frame, dt)
    k2u, k2v, k2h, k2T = RHS(u + 0.5*dt*k1u, v + 0.5*dt*k1v, h + 0.5*dt*k1h, T + 0.5*dt*k1T, frame+0.5, dt)
    k3u, k3v, k3h, k3T = RHS(u + 0.5*dt*k2u, v + 0.5*dt*k2v, h + 0.5*dt*k2h, T + 0.5*dt*k2T, frame+0.5, dt)
    k4u, k4v, k4h, k4T = RHS(u + dt*k3u, v + dt*k3v, h + dt*k3h, T + dt*k3T, frame+1.0, dt)
    u += dt * (k1u + 2*k2u + 2*k3u + k4u) / 6
    v += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
    h += dt * (k1h + 2*k2h + 2*k3h + k4h) / 6
    T += dt * (k1T + 2*k2T + 2*k3T + k4T) / 6
    return u, v, h, T

# ============================================
# Main integration loop (target_lead hours)
# ============================================
print(f"Integrating for {target_lead} hours...")
dt = compute_dt(u, v, 0)   # initial time step
total_steps = int(target_lead * 3600 / dt)
print(f"Using dt = {dt:.1f} s, total steps = {total_steps}")

for step in range(total_steps):
    # (frames could be saved here if desired)
    u, v, h, T = RK4(u, v, h, T, dt, step)

# Final predicted fields
T_pred = T.copy()
u_pred = u.copy()
v_pred = v.copy()
h_pred = h.copy()

# ============================================
# Comparison with ERA5 truth
# ============================================
# Interpolate truth to model grid
T_true = interp(t2m_valid)
u_true = interp(u10_valid)
v_true = interp(v10_valid)

# RMSE and MAE functions
def rmse(pred, true):
    return np.sqrt(np.nanmean((pred - true)**2))

def mae(pred, true):
    return np.nanmean(np.abs(pred - true))

print(f"\n=== Validation at {target_lead}h ===")
print(f"Temperature RMSE: {rmse(T_pred, T_true):.3f}°C")
print(f"Temperature MAE : {mae(T_pred, T_true):.3f}°C")
print(f"U-wind RMSE     : {rmse(u_pred, u_true):.3f} m/s")
print(f"V-wind RMSE     : {rmse(v_pred, v_true):.3f} m/s")

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"6‑hour Simulation Result vs ERA5 (Initial: {start_date})", fontsize=16, y=0.98, weight="bold")

# Temperature
im1 = axes[0,0].imshow(T_pred, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdBu_r')
axes[0,0].set_title("Predicted Temperature (°C)")
plt.colorbar(im1, ax=axes[0,0], label='°C')

im2 = axes[0,1].imshow(T_true, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdBu_r')
axes[0,1].set_title("ERA5 Temperature (°C)")
plt.colorbar(im2, ax=axes[0,1], label='°C')

im3 = axes[0,2].imshow(T_pred - T_true, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdBu_r')
axes[0,2].set_title("Error (Pred - True) (°C)")
plt.colorbar(im3, ax=axes[0,2], label='°C')

# Wind
speed_pred = np.sqrt(u_pred**2 + v_pred**2)
speed_true = np.sqrt(u_true**2 + v_true**2)

cf1 = axes[1,0].contourf(Lon, Lat, speed_pred, levels=20, cmap='viridis')
axes[1,0].quiver(Lon[::3,::3], Lat[::3,::3], u_pred[::3,::3], v_pred[::3,::3], scale=300)
axes[1,0].set_title("Predicted Wind Speed (m/s)")
plt.colorbar(cf1, ax=axes[1,0], label='m/s')

cf2 = axes[1,1].contourf(Lon, Lat, speed_true, levels=20, cmap='viridis')
axes[1,1].quiver(Lon[::3,::3], Lat[::3,::3], u_true[::3,::3], v_true[::3,::3], scale=300)
axes[1,1].set_title("ERA5 Wind Speed (m/s)")
plt.colorbar(cf2, ax=axes[1,1], label='m/s')

error_speed = speed_pred - speed_true
cf3 = axes[1,2].contourf(Lon, Lat, error_speed, levels=20, cmap='RdBu_r')
axes[1,2].set_title("Wind Speed Error (m/s)")
plt.colorbar(cf3, ax=axes[1,2], label='m/s')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/validation_{target_lead}h.png", dpi=300)
print(f"Validation figure saved to {OUTPUT_DIR}")
plt.show()