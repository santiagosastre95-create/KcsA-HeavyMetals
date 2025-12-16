# KcsA–Heavy Metals  
**Supporting code for:**  
*Heavy metal directly blocks potassium channels: An experimental and theoretical approach*  
(Submitted to the *Journal of Chemical Information and Modeling*)

---

## Purpose of this repository

This repository contains **analysis scripts and processed data** supporting the
theoretical and computational results reported in the manuscript:

> **Heavy metal directly blocks potassium channels:  
> An experimental and theoretical approach**

submitted to the *Journal of Chemical Information and Modeling (JCIM)*.

The repository is provided to ensure **methodological transparency** and to
facilitate inspection of the computational analyses used in the study.

---

## Scope and limitations

- All molecular dynamics (MD) simulations were performed using **AMBER**.
- This repository **does not include raw MD trajectories**
  due to size constraints and data management best practices.
- Only **analysis scripts, processed outputs, and figures** are tracked.
- The repository is **not intended as a general-purpose software package**.

---

## Repository structure

```
.
├── Hg/            # Analyses related to Hg-bound simulations
├── Pb/            # Analyses related to Pb-bound simulations
├── K_only/        # Control simulations (K+ only)
├── scripts/       # Python analysis scripts used in the manuscript
├── .gitignore
└── README.md
```

Each system directory contains processed outputs (CSV files and figures)
derived from the corresponding MD simulations.

---

## Molecular dynamics simulations

- MD engine: **AMBER**
- Systems: KcsA potassium channel in a membrane environment
- Periodic boundary conditions: enabled
- Temperature: 300 K
- Trajectory formats: converted to analysis-friendly formats (e.g. `.xtc`)

The full simulation protocols, force fields, and system preparation details
are described in the **Methods section of the manuscript**.

---

## Analysis overview

### Ion permeation and conductance analysis

Ion translocation events were identified by tracking K⁺ ions through a
dynamically defined pore region:

- The pore was modeled as a cylinder defined by reference atoms.
- Complete permeation events were defined as bottom → pore → top transitions.
- Per-event ionic currents and conductances were estimated under an applied
  transmembrane potential.
- Heavy-metal occupancy at the channel mouth was monitored concurrently.
- Multiple independent replicas were analyzed and combined statistically.

Outputs include:
- per-event conductance tables,
- time-resolved ion and metal occupancies.

---

### Time-lagged cross-correlation analysis

To investigate the dynamical coupling between ion conduction and heavy-metal
binding, a time-lagged cross-correlation analysis was performed between:

- the number of K⁺ ions occupying the pore, and
- the number of heavy-metal ions located at the channel mouth.

The analysis was carried out independently for each replica and subsequently
averaged across replicas. Cross-correlation functions were computed over a
range of positive and negative time lags, allowing identification of temporal
ordering between metal binding and changes in ion occupancy.

This analysis provides direct evidence for a dynamical relationship between
heavy-metal binding near the pore entrance and modulation of K⁺ conduction,
supporting a direct blocking mechanism.

---

### One-dimensional PMF along the pore axis

One-dimensional potentials of mean force (PMFs) along the **z-axis** were
computed using kernel density estimation (KDE):

- Probability densities were estimated from ion positional distributions.
- Free-energy profiles were shifted to a zero minimum.
- Binding sites (S0–S4 and Scav) were defined from reference atom positions.

---

### Two-dimensional PMF (z vs r)

Two-dimensional PMFs were computed in **z–r space** using KDE, with:

- a spherical radial coordinate centered on the pore axis,
- explicit treatment of periodic boundary conditions using the
  minimum image convention in x, y, and z,
- restriction to the upper hemisphere (z > pore center),
- masking of low-statistics regions.

These analyses were used to characterize ion localization and blocking
mechanisms induced by heavy metals.

---

## Software requirements

The analysis scripts require Python 3 and the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `mdtraj`
- `MDAnalysis`

Example installation:

```bash
pip install numpy scipy matplotlib mdtraj MDAnalysis
```

---

## Units and conventions

- Length: Ångström (Å)
- Energy: kcal·mol⁻¹
- Temperature: Kelvin (K)
- Periodic boundary conditions: treated using the minimum image convention

---

## Reproducibility statement

This repository provides sufficient information to:
- inspect the analysis methodology,
- reproduce the post-processing given access to the original trajectories,
- verify consistency with the results reported in the manuscript.

Complete reproduction of the simulations requires access to the original
AMBER trajectories and topologies, which are available from the authors
upon reasonable request.

---

## Citation

If this repository is used in whole or in part, please cite the associated
manuscript:

> *Heavy metal directly blocks potassium channels:  
> An experimental and theoretical approach*  
> Journal of Chemical Information and Modeling (submitted)

---

## Contact

For questions related to this repository or the manuscript, please contact:

**MSc. Santiago Sastre**  
ssastre@fcien.edu.uy  
CEINBIO – Universidad de la República (Uruguay)
