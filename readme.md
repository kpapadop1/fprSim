# fprSim - Food Penetrating Radar Simulation Toolbox
--------------------------------------------------
FDTD-based simulation framework for mm-wave/sub-THz MIMO radar-based foreign object detection in food products for inline NDT applications. Using the gprMax Framework (https://www.gprmax.com/) it simulates the electromagnetic propagation of signals for MIMO-based antenna configurations in reflection and transmission mode, determines metrics using the complex-valued MIMO matrix and evaluates how well low-density foreign objects (e.g. PVC or PP fragments) can be detected in low-hydrated confectionery products with varying recipes (local distributions of dielectric properties).

---

## Contents
1. [Background and scope](#background-and-scope)
2. [Method overview](#method-overview)
3. [Requirements and installation](#requirements-and-installation)
4. [Quick start](#quick-start)
5. [Usage reference](#usage-reference)
6. [Citation, license and acknowledgements](#citation-license-and-acknowledgements)

---

## Background and scope

Inline NDT with mm-wave and sub-THz radar is promising for detecting low-density foreign objects in food, yet sensor design remains largely empirical. This framework serves as a numerical testbed to assess antenna arrays and sensor configurations in terms of feasibility and detection reliability prior to hardware development, supporting both sensor developers and food producers in early decision-making.

---

## Method overview

parameter JSON -> gprMax .in file -> FDTD run (GPU) -> tangential E/H on closed Huygens surface -> near-to-far-field transform (exact dyadic Green's function) -> E at each RX -> MIMO channelmatrix H_tf = E_rx,proj / E_tx -> detectability metrics -> statistics / figures

1. **Model:** grid, PML, conveyor belt, sample (layered, rough, laminated or free-form geometry) and foreign object are written as a gprMax input file. Dielectric properties come from mixing rules over the ingredients (fat, starch, sugar, protein, water, air) using Debye models.
2. **Excitation:** each TX element is modelled as a plane wave with a Gaussian pulse whose spectrum covers the target band (not covered in gprMax 3.1.7).
3. **Field reconstruction:** gprMax records the tangential fields on a closed six-face Huygens surface around the sample. An equivalence-principle transform propagates them to the real RX positions.
4. **Channel matrix:** for every TX/RX pair and frequency bin in the band, H_tf is the co-polar RX field divided by the TX spectrum. One simulation serves both the reflection and the transmission side.
5. **Scoring:** each foreign-object scene is compared with its own baseline (same product without the object) using amplitude and phase metrics, an SNR sweep and a detection rule. The output is e.g. the minimum detectable foreign-object size or position range (cookie studies).

---

## Requirements and Installation

**Software**
- Python 3.12 or above
- Python packages: 'numpy', 'scipy', 'pandas', 'matplotlib', 'h5py', 'statsmodels' (see 'requirements.txt')
- gprMax <!-- 3.1.7 --> with GPU (CUDA) support, installed in its **own** virtual environment
- Windows: MSVC compiler for building gprMax (`MSVC_BIN_PATH` in the parameter files)

**Hardware (typical full study configuration)**
- NVIDIA GPU with about 20 GB VRAM
- 17 GB of temporary disk space per TX (approximately)

preflight resource checks are implemented to estimate memory and disk use before a run. It stops if the ressources are insufficient.

**Installation**
Create a virtual environment for Python applications. Install the dependencies using 'pip install -r requirements.txt'. Then, install gprMax in a separate environment following the [gprMax documentation](https://docs.gprmax.com/en/latest/include_readme.html) and set the path to its Python interpreter in 'EMSimulationSettings.py' ('GPRMAX_PYTHON').

---

## Quick start

All runs are started from 'EMSimulation.py'. 'MODE' defines what do. 'EMSimulationSettings.py' defines the basic parmeters.

1. **Simulate (MODE 0):** Set 'MODE = 0' and choose a parameter file in 'SINGLE_RUN_PARMS_FILE' (e.g. 'parameters/parms_empty_scene_transmission_closed.json' for a first test of the field reconstruction). Run 'python EMSimulation.py'. A preflight resource check is done first, then 'simulate()' writes the gprMax input files, runs gprMax for every TX, reconstructs the fields at the RX positions and saves the channel matrix H_tf as 'data*.pkl' together with the run protocol in a time-stamped folder under 'input_output_files/'. For a foreign-object case, simulate the same scene twice: once without the object (baseline) and once with it.
2. **Score (MODE 1):** Set 'MODE = 1', enter the run folder(s) in 'ANALYZE_RESULTS_FOLDERS' and the baseline run in 'ANALYZE_BASELINE_FOLDER', and run 'python EMSimulation.py' again. 'analyze()' loads H_tf of each run and plots magnitude, phase and group delay, as well as the RMS amplitude and phase deviation from the baseline per TX/RX channel. The deviations are compared with the detectability thresholds 'AMP_NOISE_FLOOR_DB' and 'PHASE_NOISE_FLOOR_DEG' from the parameter file. No new simulation is started, so MODE 1 can be repeated any time.

---

## Usage reference

### Code structure

'EMSimulation' is a class that is assembled from mixins:

| File | Content |
|---|---|
| EMSimulation.py | MODE dispatch|
| EMSimulationSettings.py | Run settings: MODE, parameter files, study folders, per-study options etc. |
| EMSimulationParameters.py | Module-level constants (plot styles, DOE/ANOVA coding, study defaults) |
| EMSimulationModel.py | Materials, grid, time step, waveform, antennas, gprMax input file definition |
| EMSimulationSolver.py | gprMax runs, nearfield-to-farfield conversion, transfer function, run protocols etc. |
| EMSimulationAnalysis.py | Scoring, ANOVA, scan imaging, camera model etc.|
| EMSimulationStudies.py | Cookie studies with separate simulation and evaluation, moving-sample scan |
| EMSimulationPlots.py | Plots, figure Export etc. |
| EMSimulationHelpers.py | Shared helpers |
| EMSimulationExports.py | Module-level functions export |

In 'EMSimulationSettings.py' defines case to be run, while parameter are defined as JSON files in 'parameters/' (scene, sensor etc.) and physical constants in 'constants/consts.json'.

### MODE overview

| MODE | Purpose |
|---|---|
| 0 | Single simulation + analysis |
| 1 | Re-analysis of existing results against a baseline |
| 2 | DOE batch (one run per row of the DOE table) |
| 3 | DOE scoring, Type-II ANOVA, main-effect/interaction figures |
| 4 | MIMO channel-count sensitivity (reuses MODE 2/3 data) |
| 5 | Check of the Huygens source-array subsampling |
| 6 / 7 | Cookie detectability vs. foreign-object size (1–10 mm): simulation / evaluation |
| 8 / 9 | Cookie detectability vs. foreign-object position: simulation / evaluation |
| 10 / 11 | Cookie detectability vs. frequency band: simulation / evaluation |
| 12 | Moving-sample scan based on transmission principle |
| 13 | Images of the scan |
| 14 | THz line-camera model on the scan |
| 15 | Additional figures: excitation pulse, MIMO array geometry |

### Output structure

```
studies/<study>/
├── input_output_files/   Raw gprMax output, data*.pkl per run, manifests etc.
└── results_analysis/     CSV results, ANOVA tables, PNG images
```
Each run folder also contains a **run protocol**: 'protocol.txt' (all positions, angles, grid snapping), 'protocol_parms.json' (complete parameter set), plus a 'run_protocol.csv' per study. All simulation modes keep a manifest, so an interrupted study resumes and only retries those runs that have failed.


## Citation, license and acknowledgements

  **License:** This project is licensed under the MIT License – see [LICENSE](LICENSE).
  gprMax is a separate program under GPLv3 and is not part of this repository.

**Acknowledgements:** This work was carried out at the Cologne Laboratory of Artificial Intelligence and Smart Automation (CAISA), TH Köln – University of Applied Sciences. Simulations use [gprMax](https://github.com/gprMax/gprMax) (Warren, Giannopoulos & Giannakis, *Computer Physics Communications*, 2016).

**Contact:** Konstantinos Papadopoulos, ORCID [0009-0001-4446-3870](https://orcid.org/0009-0001-4446-3870), konstantinos.papadopoulos@th-koeln.de