#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from scipy.stats import gaussian_kde

# ==== Input parameters ====
top_file = "md-pb.pdb"
traj_file = "md-pb_each5.xtc"
selection = "name PB"        # ion selection
temp = 300                  # K
#kB = 0.0083144621           # kJ/(mol·K)
kB = 0.001989106             # kcal/(mol·K)
bw = 0.1                     # KDE bandwidth (nm)
n_points = 300               # number of points along z

# Atom indices defining S0 and Scav
# (indices as in the topology, starting from 1)
S0_atoms = [745, 2063, 3283, 4945]    # example
Scav_atoms = [694, 2013, 3332, 4650]  # example

# ==== Load system ====
u = mda.Universe(top_file, traj_file)

# Collect z positions of the ions
z_positions = []
for ts in u.trajectory:
    sel = u.select_atoms(selection)
    z_positions.extend(sel.positions[:, 2])
z_positions = np.array(z_positions)

# Compute average z coordinates for S0 and Scav
sel_S0 = u.select_atoms(f"index {' '.join(str(i-1) for i in S0_atoms)}")
sel_Scav = u.select_atoms(f"index {' '.join(str(i-1) for i in Scav_atoms)}")

z_S0 = np.mean(sel_S0.positions[:, 2])
z_Scav = np.mean(sel_Scav.positions[:, 2])

# Compute intermediate sites S1–S4
z_sites = np.linspace(z_S0, z_Scav, 6)  # S0 ... Scav
site_labels = ["S0", "S1", "S2", "S3", "S4", "Scav"]

# ==== KDE ====
kde = gaussian_kde(
    z_positions,
    bw_method=bw / np.std(z_positions, ddof=1)
)
z_grid = np.linspace(z_positions.min(), z_positions.max(), n_points)
P_kde = np.maximum(kde(z_grid), 1e-12)

free_energy_kde = -kB * temp * np.log(P_kde)
free_energy_kde -= free_energy_kde.min()

# ==== Plot ====
plt.figure(figsize=(6, 4))
plt.plot(z_grid, free_energy_kde, color="royalblue", lw=2)

# Add vertical lines and labels for binding sites
for z_val, label in zip(z_sites, site_labels):
    plt.axvline(z_val, color="red", linestyle="--", lw=1)
    plt.text(
        z_val, free_energy_kde.max() * 0.95, label,
        rotation=90, va="top", ha="center",
        color="red", fontsize=8
    )

plt.xlim(50, 70)
plt.ylim(0, 5)
plt.xlabel("z (nm)")
plt.ylabel("Free energy (kJ/mol)")
plt.title("Free energy profile (KDE) with binding sites")
plt.grid(True)
plt.tight_layout()
plt.savefig("free_energy_profile_kde_sites.png", dpi=300)
plt.show()
