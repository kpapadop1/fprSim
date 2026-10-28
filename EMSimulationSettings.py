"""Run settings of EMSimulation.py: MODE selection, parameter/constants files, study folders and per-study options.

Imported by the __main__ block of EMSimulation.py; edit the values here.
"""

# Mode definition 
#   0: Simulate and analyze
#   1: Analyze existing 
#   2: DOE batch
#   3: Score DOE batch (+ paper figures/tables of both full-factorial studies)
#   4: MIMO channel sensitivity
#   5: Source-stride check
#   6/7: Cookie FO-size simulation / evaluation
#   8/9: FO-position simulation / evaluation
#   10/11: Frequency simulation / evaluation
#   12: Moving-sample scan (MSLA) simulation
#   13: Moving-sample scan (MSLA) evaluation
#   14: Moving-sample scan (MSLA) realistic (approx) camera model evaluation
#   15: Paper figures to exports/: Gaussian excitation pulse (time + frequency domain) and MIMO TX/RX array geometry
MODE = 14

# Which parameter/constants file set to use for this run (in modes 0 and 1).
STUDY = 'reflection'  # 'reflection' or 'transmission'

# Parameter file for the cookie detectability study (modes 6-11)
PARMS_FILE = {'reflection': 
              'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_reflection.json',
              'transmission': 
              'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_transmission.json',}[STUDY]

# Override parameter file for feasibility test
if MODE == 0 or MODE == 1:
    PARMS_FILE = 'parameters/for_feasibility_tests/parms_transmission_nofo_highmoisture.json'  

# Single run (mode 0): parameter file; analysis (mode 1): run folders and baseline folder
SINGLE_RUN_PARMS_FILE = 'parameters/parms_empty_scene_transmission_closed.json'
ANALYZE_RESULTS_FOLDERS = ['input_output_files//20260916_131504', 
                           'input_output_files//20260916_131504']
ANALYZE_BASELINE_FOLDER = 'input_output_files//20260916_131504'

CONSTS_FILE = 'constants/consts.json'

# Python interpreter of the gprMax environment (None = <parent of this folder>/EM Simulation/.venv/Scripts/python.exe)
GPRMAX_PYTHON = None

# Path to the DOE CSV whose rows define the parameter sets to simulate/score (Used in modes 2 and 3).
DOE_CSV_FILE = 'studies/food_sim_doe_64run.csv'

# results folder (data.pkl) used as the 'no anomaly' reference for every DOE row (under studies/) (only in mode 3)
_DOE_STUDY_DIR = {'reflection': 
                  'reflection-based full-factorial sensitivity study','transmission': 
                  'transmission-based full-factorial sensitivity study',}[STUDY]

# Baseline (no-anomaly reference) run folder
_DOE_BASELINE_RUN = {'reflection': '20260827_110644_base',
                     'transmission': '20260916_131504_base'}[STUDY]
DOE_BASELINE_FOLDER = (f'studies/{_DOE_STUDY_DIR}/'
                        f'input_output_files/{_DOE_BASELINE_RUN}')

# Manifest/output locations under "studies/"" (only in mode 3)
DOE_MANIFEST_FILE = (f'studies/{_DOE_STUDY_DIR}/'
                      'input_output_files/doe_manifest_food_sim_doe_64run.csv')
DOE_OUTPUT_CSV = (f'studies/{_DOE_STUDY_DIR}/'
                   'results_analysis/doe_results_food_sim_doe_64run.csv')

# Both full-factorial studies and the output folder of their paper figures/tables (only in mode 3)
DOE_PAPER_STUDY_DIRS = {'Reflection': 'studies/reflection-based full-factorial sensitivity study',
                        'Transmission': 'studies/transmission-based full-factorial sensitivity study'}
DOE_PAPER_ASSETS_DIR = 'studies/full-factorial sensitivity study'

# Retry FAILED manifest rows (True) or skip them. OK rows are always skipped. (only in mode 2)
RETRY_FAILED = True

# Restrict to these DOE 'run' ids (None = all rows), e.g. to test a fix on a few rows first (only in mode 2).
DOE_RUN_SUBSET = list(range(1, 65))

# Re-uses the DOE's collected 5x5-channel data (only in mode 4)
MIMO_STUDY_DIR = f'studies/{_DOE_STUDY_DIR}'
MIMO_OUTPUT_DIR = f'studies/mimo sensitivity study/{STUDY}'
MIMO_CHANNEL_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25]
MIMO_N_DRAWS = 200
MIMO_RESPONSE_METRIC = 'rms_dphase'
MIMO_SEED = 0

# Output folder of source_array_subsampling_check.png (only in mode 5; None: this file's directory).
SRC_STRIDE_CHECK_OUTPUT_DIR = 'studies/source-array subsampling check'

# Parameter file whose grid/band/TX positions are checked (only in mode 5; the cookie study file: production geometry)
SRC_STRIDE_CHECK_PARMS_FILE = 'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_both.json'

# One overlay file per product (COOKIE_NAME, SUT/LAYERS, optional GRID); the rest comes from PARMS_FILE (only in modes 6, 7, 8)
COOKIE_PARMS_FILES = ['parameters/cookie_types/cookie_crisp.json',
                      'parameters/cookie_types/cookie_chocolate_waffle.json',
                      'parameters/cookie_types/cookie_waffle.json',
                      'parameters/cookie_types/cookie_butter.json',]

# Cookie detectability study (modes 6-11): 
#   'both': reflection AND transmission from ONE simulation per run (parameter #           file with ALSO_MIRRORED_RX = true. Every run folder holds 
#           data_reflection.pkl + data_transmission.pkl and the evaluations 
#           score both sides at once); 
#   'reflection' / 'transmission': single-side parameter files as before. One
#                                  study folder per COOKIE_STUDY (manifest, 
#                                  CSVs, figures, input_output_files/)
COOKIE_STUDY = 'both'
COOKIE_PARMS_FILE = {'both': 
                     'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_both.json','reflection': 
                     'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_reflection.json',
                     'transmission': 
                     'parameters/parms_cylindric_cookie_pvc_7mm_rectangular_transmission.json',}[COOKIE_STUDY]
COOKIE_STUDY_DIR = f'studies/cookie detectability study/fo size/{COOKIE_STUDY}'
COOKIE_FO_MATERIAL = 'PVC'
COOKIE_RESPONSE_METRICS = ('rms_dphase_max', 
                           'rms_dmag_max')

# Detection rule (mode 7): 'envelope' or 'conservative'
COOKIE_DETECTION_RULE = 'envelope'

# Result selection (.pkl files; only in modes 7, 9, 11;  None = reflection and transmission (every side found), or e.g. ('reflection',)
COOKIE_RX_SIDES = None

# Assumed per-bin SNRs for cookie_detectability_fo_size_snr_sensitivity.csv ("()": skip)
COOKIE_SNR_SWEEP_DB = (20, 30, 40)

# Fixed (assuming: already detectable)FO size in mm swept from center to near the edge (only in mode 8) along with number of steps, axis of displacement, and edge clearance. The runs are reused (manifest) when set back to False.
COOKIE_FO_POSITION_SIZE_MM = 5.0
COOKIE_FO_POSITION_N_STEPS = 10
COOKIE_FO_POSITION_AXIS = 'x'
COOKIE_FO_POSITION_EDGE_CLEARANCE_MM = 1.0
COOKIE_POSITION_STUDY_DIR = f'studies/cookie detectability study/fo position/{COOKIE_STUDY}'

# Operating bands in GHz (the reference band's runs come from the fo-position study unless COOKIE_FREQUENCY_REUSE_POSITION_RUNS = False.; only in mode 10, 11)
COOKIE_FREQUENCY_BANDS_GHZ = [(22, 26), 
                              (59, 63), 
                              (77, 81), 
                              (96, 100), 
                              (120, 124)]
COOKIE_FREQUENCY_FO_SIZE_MM = 5.0
COOKIE_FREQUENCY_REUSE_POSITION_RUNS = False

# Pilot: True = MODE == 10 only runs the highest band for the first product (baseline + centered cube) to catch errors early. Its runs are reused (manifest) when set back to False.
COOKIE_FREQUENCY_PILOT = False
COOKIE_FREQUENCY_STUDY_DIR = f'studies/cookie detectability study/frequency/{COOKIE_STUDY}_huygens2'

# Parameter values set in every frequency-study run (2 Huygens samples per wavelength on caps and walls, so 120-124 GHz fits on the GPU; None = as in the parms file)
COOKIE_FREQUENCY_PARMS_OVERRIDES = {'RX_HUYGENS_SAMPLES_PER_WAVELENGTH': 2,
                                    'RX_HUYGENS_WALL_SAMPLES_PER_WAVELENGTH': 2}

# Wideband: True = one run per product and case over all bands (mesh of the highest band; pilot = first product), results in <study dir>/wideband; MODE == 11 cuts out each band. False = one run per band.
COOKIE_FREQUENCY_WIDEBAND = False

# Wideband PML thickness in free-space wavelengths at the lowest band edge (N_PML raised, inner domain unchanged; None = keep N_PML)
COOKIE_FREQUENCY_WIDEBAND_PML_WAVELENGTHS = 1.0

# Only used in modes 12, 13
#--------------------------
# The transmission parms file supplies band/belt/PML/GPU settings; the TX/RX geometry is set by the function itself.
MSLA_BASE_PARMS_FILE = 'parameters/parms_cookie_100ghz_transmission.json'
MSLA_COOKIE_PARMS_FILE = 'parameters/cookie_types/cookie_jelly_oval.json'

# FO size
MSLA_FO_SIZE_MM = 7.0

# Relative position inside the cookie (x_rel, y_rel, z_rel; reference: lower left, mid-height; 'RING_LEFT': FREEFORM cookie, left, on the belt)
MSLA_FO_POSITION_RELATIVE = 'RING_LEFT'

# FO lifted off the belt [mm] (sits at the ceiling of a hole drilled from below)
MSLA_FO_LIFT_MM = 0.5

# Air gap around the FO [mm]: hole from the belt up to the FO top, this much wider on each side (0 = no hole)
MSLA_FO_AIR_GAP_MM = 0.5

# Number of receivers (None = as many receivers as fit into the PML-free inner domain at the pitch below)
MSLA_N_RX = None

# RX spacing along y in mm
MSLA_RX_PITCH_MM = 1.5         

# Simulation domain width in y-direction (relative to the cookies width)
MSLA_DOMAIN_WIDTH_FACTOR = 1.0     

# RX gap below the belt's lower face (fixed in the domain)
MSLA_RX_GAP_BELOW_BELT_MM = 5.0    

# TX height above the belt
MSLA_TX_STANDOFF_M = 0.35     

# Conveyor belt increment in x--direction
MSLA_STEP_MM = 1.5           

# Symmetric travel path (None: as far as the cookie stays clear of the PML)
MSLA_SCAN_HALF_RANGE_MM = None     

# Simulation domain width in x-direction (relative to the cookies width)
MSLA_DOMAIN_LENGTH_FACTOR = 2.0

# Flag for quick run test (True: only the centered position (empty belt + baseline + FO = 3 runs); False: full scan)
MSLA_PILOT = False

# Optional reference runs per position (False: only the FO run; MODE 13 then aborts, MODE 14 still works)
MSLA_SIMULATE_BASELINE = False
MSLA_SIMULATE_EMPTY = False

# Flag for scene check (planning, memory estimate etc.)
MSLA_DRY_RUN = False

# Directory for storing study results
MSLA_STUDY_DIR = 'studies/moving sample line array study'

# Only in mode 14
#----------------
# (for analyze_moving_sample_camera(): detector model of a THz line camera on the; for data by mode 12).

# Paper sheets in the beam for damping (columns '0p', '10p', '20p')
CAMERA_SHEETS = (0, 10, 20)

# Power damping of one paper sheet in dB
CAMERA_ATTEN_DB_PER_SHEET = 0.15

# Frequency to use (None = band-mean power of the simulated band; else nearest bin)
CAMERA_FREQ_GHZ = 100.0

# Pixel aperture in mm (Gaussian Full Swith at Half Maximum - FWHM; along the RX line)
CAMERA_APERTURE_MM = 2.0 # 4.0

# Smearing due to conveyor movement in mm
CAMERA_SMEAR_X_MM = 0.2

# Spacing of camera pixels in y-direction in mm (None: simulated RX pitch)
CAMERA_PITCH_MM = 1.5

# Step along x-direction in mm (None: simulated conveyor step)
CAMERA_STEP_MM = None

# Interpolation method for camera image resampling ('cubic' | 'linear'; only used if pitch/step differ from the simulation)
CAMERA_INTERP = 'cubic'

# Temporal noise (std in full-scale units) 
CAMERA_NOISE_REL = 0.02

# Standard deviation of gain of every detector pixel
CAMERA_GAIN_SPREAD = 0.05

# Offset of gain of every detector pixel
CAMERA_OFFSET_REL = 0.01

# ADC bits of the detector pixels
CAMERA_ADC_BITS = 12

# Largest value the detector can show (empty beam (0p) just reaches full scale, which corresponds to a saturated red background)
CAMERA_FULL_SCALE = 0.35

# Random seed for the camera noise
CAMERA_SEED = 1

# Color scaling ('fixed': 0 .. full scale for every image; 'auto': each image to its own range)
CAMERA_DISPLAY = 'fixed'

# Reference power for the image units (band-mean |H|^2 of the transfer function, e.g. a measured empty-belt level; None = automatic)
CAMERA_REFERENCE_POWER = None

# Same reference as a level in dB (10*log10 of the |H|^2 power above); set at most one of the two
CAMERA_REFERENCE_POWER_DB = None