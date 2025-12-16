#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== GENERAL PARAMETERS =====================
DT_PS_DEFAULT = 400.0
WINDOW_FRAMES = 1
MAX_LAG_FRAMES = 200

PB_DIR = Path("Pb")
HG_DIR = Path("Hg")
FNAME = "ocupacion_iones_metales_multi.csv"
OUTFIG = "fig_Pb_Hg_timeseries_crosscorr.png"

# Matplotlib style parameters
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})


# ===================== CROSS-CORRELATION FUNCTION =====================
def crosscorr(x, y, max_lag):
    """
    Compute the normalized cross-correlation between two time series.

    Parameters
    ----------
    x, y : array-like
        Input time series.
    max_lag : int
        Maximum lag (in frames) to compute.

    Returns
    -------
    lags : ndarray
        Array of lag values (from -max_lag to +max_lag).
    corr : ndarray
        Cross-correlation values for each lag.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Remove mean
    x = x - x.mean()
    y = y - y.mean()

    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        lags = np.arange(-max_lag, max_lag + 1)
        return lags, np.zeros_like(lags)

    # Normalize by standard deviation
    x /= sx
    y /= sy

    n = len(x)
    corr_full = np.correlate(x, y, mode="full") / n
    mid = n - 1

    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr_full[mid - max_lag : mid + max_lag + 1]
    return lags, corr


# ===================== CONDITION ANALYSIS =====================
def analyze_condition(folder: Path):
    """
    Analyze a simulation condition (Pb or Hg).

    The function:
    - loads ion/metal occupancy data,
    - infers the time step,
    - smooths time series,
    - computes cross-correlation per replica,
    - averages correlations across replicas.

    Parameters
    ----------
    folder : Path
        Directory containing the CSV file for the condition.

    Returns
    -------
    time_ns : ndarray
        Time axis in nanoseconds.
    M_smooth : ndarray
        Smoothed heavy-metal occupancy in the channel mouth.
    K_smooth : ndarray
        Smoothed K+ occupancy in the pore.
    lags_ns : ndarray
        Lag axis in nanoseconds.
    corr_mean : ndarray
        Mean cross-correlation across replicas.
    corr_std : ndarray
        Standard deviation of cross-correlation across replicas.
    """
    df = pd.read_csv(folder / FNAME)
    df = df.sort_values(["Replica", "Frame"]).reset_index(drop=True)

    # Infer dt from the first replica
    first_rep = df["Replica"].iloc[0]
    sub0 = df[df["Replica"] == first_rep].sort_values("Frame")
    if len(sub0) > 1:
        diffs = np.diff(sub0["Time_ps"].values)
        pos = diffs[diffs > 0]
        dt_ps = float(np.median(pos)) if len(pos) > 0 else DT_PS_DEFAULT
    else:
        dt_ps = DT_PS_DEFAULT

    # Build global time axis (concatenated replicas)
    df["time_ns"] = np.arange(len(df)) * dt_ps * 1e-3

    # Rolling average smoothing
    df["K_smooth"] = df["nK_in_pore"].rolling(
        window=WINDOW_FRAMES, center=True
    ).mean()
    df["M_smooth"] = df["nHeavy_in_mouth"].rolling(
        window=WINDOW_FRAMES, center=True
    ).mean()

    # Cross-correlation computed per replica
    replicas = df["Replica"].unique()
    corrs = []
    lags_frames = None

    for rep in replicas:
        sub = df[df["Replica"] == rep].sort_values("Frame")
        nK = sub["nK_in_pore"].values.astype(float)
        nM = sub["nHeavy_in_mouth"].values.astype(float)
        if len(nK) < 2:
            continue

        lags, corr = crosscorr(nK, nM, MAX_LAG_FRAMES)
        if lags_frames is None:
            lags_frames = lags
        corrs.append(corr)

    corrs = np.array(corrs)
    corr_mean = np.nanmean(corrs, axis=0)
    corr_std = np.nanstd(corrs, axis=0)

    # Convert lag axis to nanoseconds
    lags_ns = lags_frames * dt_ps * 1e-3

    return (
        df["time_ns"].values,
        df["M_smooth"].values,
        df["K_smooth"].values,
        lags_ns,
        corr_mean,
        corr_std,
    )


# ===================== LOAD Pb AND Hg DATA =====================
print("Processing Pb...")
t_pb, M_pb, K_pb, lag_pb, cmean_pb, cstd_pb = analyze_condition(PB_DIR)

print("Processing Hg...")
t_hg, M_hg, K_hg, lag_hg, cmean_hg, cstd_hg = analyze_condition(HG_DIR)


# ===================== FIGURE (NO SHARED AXES) =====================
fig, axes = plt.subplots(3, 2, figsize=(6, 4))

# Colors
color_M_pb = "orange"
color_K_pb = "lightcoral"
color_cc_pb = "orange"

color_M_hg = "0.6"
color_K_hg = "red"
color_cc_hg = "0.6"

# Panel labels
labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

# ---- Panel (a) Pb: metals in mouth ----
ax = axes[0, 0]
ax.plot(t_pb, M_pb, lw=1.5, color=color_M_pb)
ax.set_ylabel("⟨Metals in mouth⟩")
ax.text(0.02, 0.9, "a)", transform=ax.transAxes, fontweight="bold")

# ---- Panel (b) Hg: metals in mouth ----
ax = axes[0, 1]
ax.plot(t_hg, M_hg, lw=1.5, color=color_M_hg)
ax.text(0.02, 0.9, "b)", transform=ax.transAxes, fontweight="bold")

# ---- Panel (c) Pb: K+ in pore ----
ax = axes[1, 0]
ax.plot(t_pb, K_pb, lw=1.5, color=color_K_pb)
ax.set_ylabel("⟨K⁺ in pore⟩")
ax.text(0.02, 0.9, "c)", transform=ax.transAxes, fontweight="bold")

# ---- Panel (d) Hg: K+ in pore ----
ax = axes[1, 1]
ax.plot(t_hg, K_hg, lw=1.5, color=color_K_hg)
ax.text(0.02, 0.9, "d)", transform=ax.transAxes, fontweight="bold")

# ---- Panel (e) Pb: cross-correlation ----
ax = axes[2, 0]
ax.plot(lag_pb, cmean_pb, lw=1.8, color=color_cc_pb)
ax.fill_between(
    lag_pb,
    cmean_pb - cstd_pb,
    cmean_pb + cstd_pb,
    alpha=0.3,
    color=color_cc_pb,
)
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("Lag (ns)")
ax.set_ylabel("Cross-correlation")
ax.text(0.02, 0.9, "e)", transform=ax.transAxes, fontweight="bold")

# ---- Panel (f) Hg: cross-correlation ----
ax = axes[2, 1]
ax.plot(lag_hg, cmean_hg, lw=1.8, color=color_cc_hg)
ax.fill_between(
    lag_hg,
    cmean_hg - cstd_hg,
    cmean_hg + cstd_hg,
    alpha=0.3,
    color=color_cc_hg,
)
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("Lag (ns)")
ax.text(0.02, 0.9, "f)", transform=ax.transAxes, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
plt.show()
print(f"Figure saved as: {OUTFIG}")
