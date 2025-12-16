#!/usr/bin/env python3
import mdtraj as md
import numpy as np
import csv
import sys
from math import sqrt

# ===================== GENERAL CONFIGURATION =====================

# List of replicas: (xtc, topology)  --> EDIT THIS
REPLICAS = [
    ("md-hg_1.xtc", "../md_hg.pdb"),
    ("md-hg_2.xtc", "../md_hg.pdb"),
    ("md-hg_3.xtc", "../md_hg.pdb"),
    ("md-hg_4.xtc", "../md_hg.pdb"),
    # add as many as needed
]

ion_resname = "K+"          # residue name for K+
cylinder_atom_indices = [709, 744, 2028, 2063, 3347, 3375, 4666, 4694]

# ps between frames if the trajectory does NOT contain timestep info
DEFAULT_DT_PS = 400.0

dV_mV = 200                 # applied potential (mV)
radius_factor = 1.0         # safety factor for cylinder radius

# Heavy metals
heavy_resnames = ["PB", "PB2", "HG", "HG2"]   # adjust according to topology
mouth_thickness_A = 5.0     # thickness (Å) above z_max defining the channel mouth

# ===================== CONSTANTS =====================
q_coulombs = 1.602e-19
dV_V = dV_mV / 1000.0

# ===================== AUXILIARY FUNCTIONS =====================

def get_state(x, y, z, frame, center_xy, z_min, z_max, cylinder_radius):
    """
    State of a K+ ion relative to the dynamic cylinder.
    0   = outside (below)
    1   = inside the cylinder (pore)
    2   = outside (above)
    0.5 = within Z-range but outside XY (ignored)
    """
    dx = x - center_xy[frame, 0]
    dy = y - center_xy[frame, 1]
    r = sqrt(dx*dx + dy*dy)
    inside_xy = (r <= cylinder_radius)
    inside_z = (z >= z_min) and (z <= z_max)

    if not inside_xy:
        if z < z_min:
            return 0
        elif z > z_max:
            return 2
        else:
            return 0.5
    else:
        if z < z_min:
            return 0
        elif z > z_max:
            return 2
        else:
            return 1

def in_channel_mouth(x, y, z, frame, center_xy,
                     z_mouth_min, z_mouth_max, cylinder_radius):
    dx = x - center_xy[frame, 0]
    dy = y - center_xy[frame, 1]
    r = sqrt(dx*dx + dy*dy)
    inside_xy = (r <= cylinder_radius)
    inside_mouth_z = (z >= z_mouth_min) and (z <= z_mouth_max)
    return inside_xy and inside_mouth_z

# ===================== PER-REPLICA PROCESSING =====================

def process_replica(rep_id, xtc_file, topology_file):
    print(f"\n===== Processing replica {rep_id}: {xtc_file} =====")

    try:
        traj = md.load(xtc_file, top=topology_file)
    except Exception as e:
        print(f"Error loading {xtc_file}: {e}")
        return {
            "rep_id": rep_id,
            "transitions": [],
            "nK_in_pore": np.array([]),
            "nHeavy_in_mouth": np.array([]),
            "dt_ps": 0.0,
            "n_frames": 0
        }

    if hasattr(traj, "timestep") and traj.timestep is not None and traj.timestep > 0:
        dt_ps = float(traj.timestep)
        print(f"Using trajectory timestep: {dt_ps} ps/frame")
    else:
        dt_ps = DEFAULT_DT_PS
        print(f"No timestep found in trajectory, using DEFAULT_DT_PS={dt_ps} ps/frame")

    n_frames = traj.n_frames
    traj_xyz_ang = traj.xyz * 10.0  # nm -> Å

    # K+ indices
    ion_indices = [a.index for a in traj.topology.atoms
                   if a.residue.name.strip() == ion_resname]
    if not ion_indices:
        print(f"[Replica {rep_id}] No atoms found with resname '{ion_resname}'.")
        return {
            "rep_id": rep_id,
            "transitions": [],
            "nK_in_pore": np.zeros(n_frames, dtype=int),
            "nHeavy_in_mouth": np.zeros(n_frames, dtype=int),
            "dt_ps": dt_ps,
            "n_frames": n_frames
        }

    print(f"[Replica {rep_id}] Detected {len(ion_indices)} K+ ions "
          f"(e.g.: {ion_indices[:10]}...)")

    # Heavy metals
    heavy_indices = [a.index for a in traj.topology.atoms
                     if a.residue.name.strip() in heavy_resnames]
    print(f"[Replica {rep_id}] Detected {len(heavy_indices)} heavy metals: {heavy_resnames}")

    # Define the cylinder from reference atoms
    cylinder_pos = traj_xyz_ang[:, cylinder_atom_indices, :]
    center_xy = np.mean(cylinder_pos[:, :, :2], axis=1)

    dist_xy = np.linalg.norm(
        cylinder_pos[:, :, :2] - center_xy[:, np.newaxis, :], axis=2
    )
    radius_per_frame = np.max(dist_xy, axis=1)
    cylinder_radius = np.max(radius_per_frame) * radius_factor

    z_min = np.min(cylinder_pos[:, :, 2])
    z_max = np.max(cylinder_pos[:, :, 2])

    z_mouth_min = z_max
    z_mouth_max = z_max + mouth_thickness_A

    print(f"[Replica {rep_id}] Cylinder radius: {cylinder_radius:.3f} Å")
    print(f"[Replica {rep_id}] Cylinder Z-range: {z_min:.3f} -> {z_max:.3f} Å")
    print(f"[Replica {rep_id}] Mouth (metals): Z {z_mouth_min:.3f} -> {z_mouth_max:.3f} Å")

    # Per-ion tracking
    prev_state = {i: None for i in ion_indices}
    transition_progress = {i: 0 for i in ion_indices}
    entry_frame = {i: None for i in ion_indices}
    transitions = []

    # Occupancies
    nK_in_pore = np.zeros(n_frames, dtype=int)
    nHeavy_in_mouth = np.zeros(n_frames, dtype=int)

    print(f"[Replica {rep_id}] Analyzing {n_frames} frames...")

    for frame in range(n_frames):
        # K+: transitions + occupancy
        for ion in ion_indices:
            x, y, z = traj_xyz_ang[frame, ion, :]
            state = get_state(x, y, z, frame,
                              center_xy, z_min, z_max, cylinder_radius)
            prev = prev_state[ion]

            if state == 1:
                nK_in_pore[frame] += 1

            if prev is None:
                prev_state[ion] = state
                continue

            if state == 0.5:
                transition_progress[ion] = 0
                entry_frame[ion] = None
                prev_state[ion] = state
                continue

            if transition_progress[ion] == 0:
                if prev == 0 and state == 1:
                    transition_progress[ion] = 1
                    entry_frame[ion] = frame
            elif transition_progress[ion] == 1:
                if prev == 1 and state == 2:
                    exit_frame = frame
                    entry = entry_frame[ion]
                    delta_frames = exit_frame - entry
                    delta_t_ps = delta_frames * dt_ps
                    if delta_t_ps > 0:
                        delta_t_s = delta_t_ps * 1e-12
                        I_A = q_coulombs / delta_t_s
                        G_S = I_A / dV_V
                        transitions.append({
                            "replica": rep_id,
                            "ion": ion,
                            "entry": entry,
                            "exit": exit_frame,
                            "delta_t_ps": delta_t_ps,
                            "I_A": I_A,
                            "G_S": G_S
                        })
                    transition_progress[ion] = 0
                    entry_frame[ion] = None
                elif state == 0:
                    transition_progress[ion] = 0
                    entry_frame[ion] = None

            prev_state[ion] = state

        # Heavy metals in channel mouth
        if heavy_indices:
            for h in heavy_indices:
                xh, yh, zh = traj_xyz_ang[frame, h, :]
                if in_channel_mouth(
                    xh, yh, zh, frame,
                    center_xy, z_mouth_min, z_mouth_max, cylinder_radius
                ):
                    nHeavy_in_mouth[frame] += 1

    print(f"[Replica {rep_id}] Detected events: {len(transitions)}")

    return {
        "rep_id": rep_id,
        "transitions": transitions,
        "nK_in_pore": nK_in_pore,
        "nHeavy_in_mouth": nHeavy_in_mouth,
        "dt_ps": dt_ps,
        "n_frames": n_frames
    }

# ===================== MAIN MULTI-REPLICA LOOP =====================

all_transitions = []
total_time_s = 0.0

# Combined occupancy with replica labels
occupancy_rows = []   # (replica, frame, time_ps, nK, nHeavy)

for i, (xtc, top) in enumerate(REPLICAS, start=1):
    rep_id = f"rep{i}"
    res = process_replica(rep_id, xtc, top)

    dt_ps = res["dt_ps"]
    n_frames = res["n_frames"]

    if n_frames == 0 or dt_ps == 0:
        continue

    # Simulation time for this replica
    time_rep_s = n_frames * dt_ps * 1e-12
    total_time_s += time_rep_s

    # Accumulate transitions
    all_transitions.extend(res["transitions"])

    # Frame-wise occupancy
    for frame in range(n_frames):
        time_ps = frame * dt_ps
        nK = res["nK_in_pore"][frame]
        nH = res["nHeavy_in_mouth"][frame]
        occupancy_rows.append((rep_id, frame, time_ps, nK, nH))

# ===================== OUTPUT WRITING =====================

# Individual transitions
csv_trans = "transitions_conductance_multi.csv"
with open(csv_trans, mode="w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Replica", "Ion", "Entry_Frame", "Exit_Frame",
                "Delta_t_ps", "I_A", "G_S", "G_pS"])
    for t in all_transitions:
        w.writerow([
            t["replica"], t["ion"], t["entry"], t["exit"],
            f"{t['delta_t_ps']:.3f}",
            f"{t['I_A']:.6e}",
            f"{t['G_S']:.6e}",
            f"{t['G_S']*1e12:.6f}"
        ])
print(f"\nSaved {len(all_transitions)} events to '{csv_trans}'.")

# Temporal occupancy of K+ / heavy metals
csv_occ = "ion_metal_occupancy_multi.csv"
with open(csv_occ, mode="w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Replica", "Frame", "Time_ps", "nK_in_pore", "nHeavy_in_mouth"])
    for row in occupancy_rows:
        w.writerow(row)
print(f"Saved temporal occupancy to '{csv_occ}'.")

# ===================== GLOBAL STATISTICS =====================

total_events = len(all_transitions)

if total_events > 0 and total_time_s > 0:
    event_rate_s = total_events / total_time_s
    I_total_A = event_rate_s * q_coulombs
    G_total_S = I_total_A / dV_V

    G_vals_S = np.array([t["G_S"] for t in all_transitions])
    G_mean_pS = np.mean(G_vals_S) * 1e12
    G_std_pS = np.std(G_vals_S, ddof=1) * 1e12 if total_events > 1 else 0.0

    print("\n===== GLOBAL SUMMARY (all replicas) =====")
    print(f"Analyzed replicas: {len(REPLICAS)}")
    print(f"Total events: {total_events}")
    print(f"Total simulation time: {total_time_s:.6e} s "
          f"({total_time_s*1e6:.3f} µs)")
    print(f"Event rate: {event_rate_s:.6e} s^-1 "
          f"({event_rate_s*1e-6:.4f} events/µs)")
    print(f"Average total current: {I_total_A:.3e} A "
          f"({I_total_A*1e12:.4f} pA)")
    print(f"Average total conductance: {G_total_S:.4e} S "
          f"({G_total_S*1e12:.4f} pS)")
    print(f"Mean conductance per event: {G_mean_pS:.4f} pS "
          f"(std = {G_std_pS:.4f} pS)")
else:
    print("\nNo events detected or total time is zero; global conductance cannot be computed.")

