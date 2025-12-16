#!/usr/bin/env python3
"""
fe_2d_pbc.py

2D PMF (z vs r) using KDE + contourf, with:
- spherical radius,
- upper hemisphere only (z > center),
- PBC correction in x, y and z (minimum image convention).

Requirements: mdtraj, numpy, scipy, matplotlib.
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import csv
import sys

# ------------------- CONFIGURATION -------------------
xtc_file = "md-hg_each5.xtc"
topology_file = "md-hg.pdb"

ion_resname = "HG"   # residue name for ions

# atoms defining the cylinder (0-based indices)
cylinder_atoms_indices = [744, 2063, 3375, 4694]
radius_factor = 1.0   # safety factor for cylinder radius

# grid & KDE parameters
z_pad   = 10.0       # padding in z around the cylinder
DZ_BIN  = 0.5        # bin size in z (Å)
DR_BIN  = 0.5        # bin size in r (Å)
r_max_plot = 40.0    # maximum r to be displayed (Å)

# --- Geometric filters (applied frame by frame) ---
# Z_CUT = None  -> no additional z filter
# Z_CUT = 66.0  -> only ions with z_mic > 66 Å (in addition to hemisphere filter)
Z_CUT = None

# Radial filter on spherical radius ρ (None = no filter)
R_MIN = None   # e.g. 0.0
R_MAX = None   # e.g. 15.0

# KDE and PMF parameters
bw_factor  = 1.0      # 1.0 = Scott's rule; >1 increases smoothing
min_counts = 5        # bins with fewer counts are masked as NaN
T  = 300.0
kB = 0.0019872041     # kcal/(mol·K)
kT = kB * T
max_G_clip = 25.0     # upper saturation for plotting
epsilon    = 1e-12    # to avoid log(0)

# True = use all ions; False = only 1 ion per frame (minimum r)
count_all_ions = True

# ------------------- TRAJECTORY LOADING -------------------
print(f"Loading trajectory: {xtc_file}")
traj = md.load(xtc_file, top=topology_file)
traj_xyz_ang = traj.xyz * 10.0  # nm -> Å

n_frames = traj.n_frames
n_atoms  = traj.n_atoms
print(f"Frames: {n_frames}, atoms: {n_atoms}")

# box lengths for PBC (Å)
if traj.unitcell_lengths is None:
    print("ERROR: trajectory has no unitcell_lengths (no periodic box).")
    sys.exit(1)
box_lengths_ang = traj.unitcell_lengths * 10.0  # (frames, 3) in Å

# check cylinder atom indices
if max(cylinder_atoms_indices) >= n_atoms:
    print("ERROR: cylinder atom indices out of range.")
    sys.exit(1)

# ion indices
ion_indices = [
    a.index for a in traj.topology.atoms
    if a.residue.name.strip() == ion_resname
]
if not ion_indices:
    print(f"No atoms found with resname '{ion_resname}'.")
    sys.exit(1)
print(f"Found {len(ion_indices)} atoms with resname '{ion_resname}'. Example: {ion_indices[:6]}")

# ------------------- CYLINDER GEOMETRY -------------------
cil_pos = traj_xyz_ang[:, cylinder_atoms_indices, :]  # (frames, n_atoms, 3)

# center_x, center_y, center_z per frame (sphere / hemisphere center)
center_xyz = np.mean(cil_pos, axis=1)   # (frames, 3)
center_xy  = center_xyz[:, :2]

# maximum XY radius (informative)
dist_xy_per_frame = np.linalg.norm(
    cil_pos[:, :, :2] - center_xy[:, np.newaxis, :], axis=2
)
radius_per_frame = np.max(dist_xy_per_frame, axis=1)
cylinder_radius  = np.max(radius_per_frame) * radius_factor

z_min_cyl = np.min(cil_pos[:, :, 2])
z_max_cyl = np.max(cil_pos[:, :, 2])

print(f"Cylinder radius ≈ {cylinder_radius:.2f} Å")
print(f"Cylinder Z range: {z_min_cyl:.2f} – {z_max_cyl:.2f} Å")

# z limits based on cylinder only
z_min = z_min_cyl - z_pad
z_max = z_max_cyl + z_pad

# ------------------- DATA COLLECTION (z_mic, ρ) WITH HEMISPHERE + PBC -------------------
zs = []   # wrapped z (z_mic)
rs = []   # spherical radius ρ
frames_record = []
ions_record   = []

for frame in range(n_frames):
    cx, cy, cz = center_xyz[frame]       # center for this frame
    Lx, Ly, Lz = box_lengths_ang[frame]  # box lengths for this frame

    for ion in ion_indices:
        x, y, z = traj_xyz_ang[frame, ion, :]

        # raw differences
        dx = x - cx
        dy = y - cy
        dz = z - cz

        # --- minimum image convention in x, y and z ---
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        dz -= Lz * np.round(dz / Lz)
        # ------------------------------------------------

        # UPPER HEMISPHERE: only points with dz > 0 (after MIC)
        if dz <= 0:
            continue

        # spherical radius
        rho = np.sqrt(dx*dx + dy*dy + dz*dz)

        # wrapped z coordinate compatible with MIC
        z_mic = cz + dz

        # optional absolute z_mic filter
        if Z_CUT is not None and z_mic <= Z_CUT:
            continue

        # optional radial filter on ρ
        if R_MIN is not None and rho < R_MIN:
            continue
        if R_MAX is not None and rho > R_MAX:
            continue

        zs.append(z_mic)
        rs.append(rho)
        frames_record.append(frame)
        ions_record.append(ion)

zs = np.array(zs)
rs = np.array(rs)
frames_record = np.array(frames_record)
ions_record   = np.array(ions_record)

if len(zs) == 0:
    print("No points remain after filtering (hemisphere + PBC + cuts).")
    sys.exit(1)

print(f"Total points after geometric filters (hemisphere + PBC xyz): {len(zs)}")

if not count_all_ions:
    sel_mask = np.zeros_like(zs, dtype=bool)
    for f in range(n_frames):
        idxs = np.where(frames_record == f)[0]
        if idxs.size == 0:
            continue
        argmin = idxs[np.argmin(rs[idxs])]
        sel_mask[argmin] = True
    zs = zs[sel_mask]
    rs = rs[sel_mask]
    frames_record = frames_record[sel_mask]
    ions_record   = ions_record[sel_mask]
    print(f"After selecting 1 ion per frame: {len(zs)} points")

# ------------------- RAW HISTOGRAM -------------------
rbins_max = r_max_plot
zbins = np.arange(z_min, z_max + DZ_BIN, DZ_BIN)
rbins = np.arange(0.0, rbins_max + DR_BIN, DR_BIN)

# safety checks
if len(zbins) < 3:
    raise ValueError(
        f"Too few z bins: {len(zbins)-1}. "
        f"Decrease DZ_BIN (DZ_BIN={DZ_BIN}) or increase z-range."
    )
if len(rbins) < 3:
    raise ValueError(
        f"Too few r bins: {len(rbins)-1}. "
        f"Decrease DR_BIN (DR_BIN={DR_BIN}) or increase r_max_plot."
    )

H, z_edges, r_edges = np.histogram2d(zs, rs, bins=[zbins, rbins])
print(f"Raw histogram: {H.sum():.0f} counts; z bins={len(zbins)-1}, r bins={len(rbins)-1}")

# bin centers
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
Zmesh, Rmesh = np.meshgrid(z_centers, r_centers, indexing="ij")

# ------------------- KDE AND PMF -------------------
data = np.vstack([zs, rs])
print("Training 2D KDE (gaussian_kde)...")
kde = gaussian_kde(data, bw_method="scott")
if bw_factor != 1.0:
    kde.set_bandwidth(kde.factor * bw_factor)

points = np.vstack([Zmesh.ravel(), Rmesh.ravel()])
P = kde(points).reshape(Zmesh.shape)

# normalize KDE (with protection)
norm = np.trapz(np.trapz(P, r_centers, axis=1), z_centers, axis=0)
if norm <= 0 or not np.isfinite(norm):
    print(f"Warning: KDE integral over z–r is {norm}. Probability not normalized.")
else:
    P /= norm

with np.errstate(divide="ignore"):
    G = -kT * np.log(P + epsilon)

G_min   = np.nanmin(G)
G_shift = G - G_min
G_clip  = np.clip(G_shift, 0, max_G_clip)

mask_low = H < min_counts
G_plot   = np.where(mask_low, np.nan, G_clip)

# ------------------- SAVE CSV -------------------
with open("pmf_zr_kde_hemisphere_pbc_xyz.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["z_center_mic", "r_spherical", "G_kcalmol"])
    for i, zc in enumerate(z_centers):
        for j, rc in enumerate(r_centers):
            w.writerow([zc, rc, G_plot[i, j]])
print("Saved: pmf_zr_kde_hemisphere_pbc_xyz.csv")

# ------------------- LAYER-STYLE PLOT -------------------
fig, ax = plt.subplots(figsize=(5, 7))

vmin = np.nanmin(G_plot)
vmax = np.nanmax(G_plot)
levels = np.linspace(vmin, vmax, 15)

cmap = plt.colormaps["RdYlBu_r"]

cf = ax.contourf(
    Zmesh, Rmesh, G_plot,
    levels=levels,
    cmap=cmap,
    extend="neither"
)

cs = ax.contour(
    Zmesh, Rmesh, G_plot,
    levels=levels,
    colors="k",
    linewidths=0.4,
    alpha=0.7
)
ax.clabel(cs, fmt="%.1f", inline=True, fontsize=7)

cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Free energy (kcal mol$^{-1}$)", fontsize=16, fontweight="bold")
plt.setp(cbar.ax.get_yticklabels(), fontsize=10, fontweight="bold")

ax.set_xlabel("z (Å)", fontsize=16)
ax.set_ylabel("r (Å)", fontsize=16)

ax.tick_params(axis="both", labelsize=14, width=1.5)

# mark cylinder region (approximate reference in original z)
ax.axvline(z_min_cyl, color="white", linestyle="--", linewidth=1)
ax.axvline(z_max_cyl, color="white", linestyle="--", linewidth=1)
ax.text(
    (z_min_cyl + z_max_cyl) / 2, r_max_plot * 0.95,
    "Cylinder filter", color="white",
    ha="center", va="top", fontsize=8,
    bbox=dict(facecolor="black", alpha=0.3, pad=2)
)

ax.set_xlim(62, 75)
ax.set_ylim(0, r_max_plot)

fig.tight_layout()
fig.savefig("pmf_zr_kde_hemisphere_pbc_xyz_hg.pdf", dpi=300)
plt.show()

print("Saved: pmf_zr_kde_hemisphere_pbc_xyz_hg.pdf")

