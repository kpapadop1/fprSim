"""Here we define module-level parameters for the EMSimulation, including physical constants, plot styles, DOE/ANOVA coding, and study defaults.
"""

# Vacuum permittivity [F/m], which converts conductivity to ohmic loss (-j*sigma/(omega*eps0)).
_VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12

# Sentinel default for analyze()'s noise-floor args
_NOISE_FLOOR_UNSET = object()

# Shared color/label/axis scheme for response metrics in analyze() Figure 5 and analyze_doe_batch()
_RESPONSE_COLOR_PRIMARY = 'steelblue'    # contrast_real / 1st factor level
_RESPONSE_COLOR_SECONDARY = 'indianred'  # contrast_complex_mag / 2nd factor level

# Base metric name
_RESPONSE_LABELS = {
    'contrast_real': "contrast_real ((eps'_fo - eps'_sut) / eps'_sut)",
    'contrast_complex_mag': "contrast_complex_mag (|eps*_fo - eps*_sut| / |eps*_sut|)",
    'mean_dtau': 'mean delta group delay (ns)',
    'rms_dtau': 'RMS delta group delay (ns)',
    'mean_dmag': 'mean delta |H_tf| (dB)',
    'rms_dmag': 'RMS delta |H_tf| (dB)',
    'mean_dphase': 'mean delta phase (deg)',
    'rms_dphase': 'RMS delta phase (deg)',
    'n_exceed': 'n_exceed (channels over threshold)',
    'weighted_score': 'weighted exceedance score',}

# Metrics with a meaningful sign get a dashed zero line and symmetric y-limits
_RESPONSE_SIGNED_METRICS = {'contrast_real', 
                            'mean_dtau', 
                            'mean_dmag', 
                            'mean_dphase'}

# Colors _draw_scene() cycles through when the SUT is layered (top layer
# first)
_SUT_LAYER_COLORS = ['saddlebrown', 
                     'sandybrown', 
                     'peru', 
                     'tan']

# Whitelist of consts.json MATERIAL_PROPERTIES keys
FO_MATERIALS_ALLOWED = {'PVC', 
                        'PP'}

# Position -> x-fraction of POSITION_RELATIVE (y stays 0.5). 0.80 keeps the foreign object inside the SUT boundary
POSITION_X_FRACTION = {'Center': 0.5, 
                       'Edge': 0.80}

# Target WATER fraction for moisture == 'High'. 'Low' reuses the base parms value.
WATER_FRACTION_HIGH = 0.15

# Target AIR fraction for 'Porous/Layered'
POROUS_AIR_FRACTION = 0.15

# For cookie studies (MODE == 6-11)
#----------------------------------
# Foreign-object cube edge lengths swept for every product  in mm.
COOKIE_FO_SIZES_MM = [10, 8, 6, 5, 4, 3, 2, 1.5, 1]

# One hue per product plus a distinct marker
_COOKIE_COLORS = ['#2a78d6',
                   '#eb6834',
                     '#1baf7a',
                       '#eda100']
_COOKIE_MARKERS = ['o', 's', '^', 'D']

# Solid densities in g/cm3 of the consts.json ingredient classes for mass-to-volume fraction conversion
_INGREDIENT_DENSITY_G_CM3 = {'WATER': 1.00, 
                             'FAT': 0.92, 
                             'PROTEIN': 1.35,
                             'STARCH': 1.50, 
                             'SUGAR': 1.59, 
                             'SALT': 2.17,}

# ANOVA-related constants and mappings
#-------------------------------------
# The 6 two-level DOE factors, ±1 coded (orthogonal coding for the balanced design)
ANOVA_FACTOR_CODES = ['A_code', 
                      'B_code', 
                      'C_code', 
                      'D_code', 
                      'E_code', 
                      'F_code']

ANOVA_CODE_TO_FACTOR = {'A_code': 'diameter_mm',
                        'B_code': 'depth_mm',
                        'C_code': 'position',
                        'D_code': 'material',
                        'E_code': 'heterogeneity',
                        'F_code': 'moisture',}

# Non-response columns of analyze_doe_batch()'s results table
_ANOVA_NON_RESPONSE_COLUMNS = frozenset(['run', 'response_status'] + ANOVA_FACTOR_CODES + list(ANOVA_CODE_TO_FACTOR.values()))

_ANOVA_SIG_LEVEL = 0.05

# Line colors of combined contrast_real / rms_dphase_max overlay plots
_COMBINED_CONTRAST_COLOR_DARK = 'navy'
_COMBINED_CONTRAST_COLOR_LIGHT = 'cornflowerblue'
_COMBINED_RMSDPHASE_COLOR_DARK = 'darkred'
_COMBINED_RMSDPHASE_COLOR_LIGHT = 'lightcoral'

# Metric denominatinon of MIMO channel-count sensitivity analysis (MODE == 4)
MIMO_METRIC_NAMES = ('mean_dtau', 
                     'rms_dtau', 
                     'mean_dmag', 
                     'rms_dmag',
                     'mean_dphase', 
                     'rms_dphase')
MIMO_RESPONSE_CHOICES = list(MIMO_METRIC_NAMES) + ['n_exceed', 'weighted_score']

# Source-array subsampling check (MODE == 5)
_SRC_STRIDE_CHECK_PARMS = {'SIMULATE_BEFORE_ANALYSIS': False,
                            'L_X_GRID': 0.05, 
                            'L_Y_GRID': 0.05, 
                            'L_Z_GRID': 0.05,
                            'CELLS_PER_WAVELENGTH': 10,
                            'FREQUENCY_RANGE': (77e9, 81e9),
                            'SWEEPING_TIME': 1e-4,
                            'APERTURE': 2e-3,
                            'TX_POS_RANGE': [[0.0, -0.025], 0.0, 0.35],
                            'N_TX': 3,
                            'TX_AMPLITUDE': 1.0,
                            'TX_POLARIZATION': 'x',
                            'RX_POS_RANGE': [[0.025, 0.0], 0.0, 0.35],
                            'N_RX': 3,
                            'RX_POLARIZATION': 'x',
                            'N_PML': 20,
                            'N_PML_MARGIN': 15,
                            'BELT': {'MATERIAL': 'PE', 
                                    'THICKNESS': 1e-3},
                            'SUT': {'INGREDIENTS_FRACTIONS':
                                    {'WATER': 0.04, 
                                    'PROTEIN': 0.06, 
                                    'STARCH': 0.29,
                                    'FAT': 0.25, 
                                    'SUGAR': 0.35, 
                                    'SALT': 0.01},
                                    'SHAPE': 'CYLINDER',
                                    'SIZE': {'DIAMETER': 0.03, 
                                            'THICKNESS': 0.01}},
                            'FOREIGN_OBJECT': {'SHAPE': 'RECTANGULAR', 
                                            'SIZE': [7e-3, 7e-3, 7e-3],
                                            'POSITION_RELATIVE': [0.5, 0.5, 0.5], 
                                            'MATERIAL': 'PVC'},
                            'INSECT': {'SHAPE': 'ELLIPSOID', 
                                    'SIZE': [8e-3, 4.5e-3, 4e-3],
                                    'POSITION_RELATIVE': [0.5, 0.5, 0.5], 
                                    'MATERIAL': 'INSECT',
                                    'INGREDIENTS_FRACTIONS': {'WATER': 0.60, 
                                                            'PROTEIN': 0.20, 
                                                            'FAT': 0.10,
                                                            'CHITIN': 0.06, 
                                                            'GLYCOGEN': 0.02, 
                                                            'SALT': 0.02}},
                            'PATH_INPUT_OUTPUT_FILES': 'input_output_files',}

_SRC_STRIDE_CHECK_CONSTS = {'MATERIAL_PROPERTIES': {
                            'PVC': {'PERMITTIVITY_RELATIVE': [2.9, 3.4], 
                                    'CONDUCTIVITY': [1e-2, 5e-2], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'PE': {'PERMITTIVITY_RELATIVE': [2.25, 2.35], 
                                'CONDUCTIVITY': [1e-4, 1e-3], 
                                'PERMEABILITY_RELATIVE': 1.0, 
                                'MAGNETIC_LOSS': 0.0},
                            'PP': {'PERMITTIVITY_RELATIVE': [2.2, 2.3], 
                                'CONDUCTIVITY': [1e-4, 1e-3], 
                                'PERMEABILITY_RELATIVE': 1.0, 
                                'MAGNETIC_LOSS': 0.0},
                            'PS': {'PERMITTIVITY_RELATIVE': [2.5, 2.6], 
                                'CONDUCTIVITY': [1e-4, 1e-3], 
                                'PERMEABILITY_RELATIVE': 1.0, 
                                'MAGNETIC_LOSS': 0.0},
                            'AIR': {'PERMITTIVITY_RELATIVE': [1.0005, 1.0007], 
                                    'CONDUCTIVITY': [0.0, 1e-14], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'WATER': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 78.0, 
                                    'EPS_INF': 4.9, 
                                    'TAU': 8.3e-12, 
                                    'CONDUCTIVITY': [1e-4, 1e-2], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'FAT': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 2.8, 
                                    'EPS_INF': 2.3, 
                                    'TAU': 2e-11, 
                                    'CONDUCTIVITY': [1e-8, 1e-6], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'PROTEIN': {'DISPERSION_MODEL': 'DEBYE', 
                                        'EPS_STATIC': 4.5, 
                                        'EPS_INF': 2.8, 
                                        'TAU': 1e-11, 
                                        'CONDUCTIVITY': [1e-4, 1e-2], 
                                        'PERMEABILITY_RELATIVE': 1.0, 
                                        'MAGNETIC_LOSS': 0.0},
                            'STARCH': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 3.8, 
                                    'EPS_INF': 2.6, 
                                    'TAU': 1e-11, 
                                    'CONDUCTIVITY': [1e-6, 1e-4], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'SUGAR': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 3.3, 
                                    'EPS_INF': 2.9, 
                                    'TAU': 5e-12, 
                                    'CONDUCTIVITY': [1e-8, 1e-6], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'SALT': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 6.0, 
                                    'EPS_INF': 5.0, 
                                    'TAU': 1e-12, 
                                    'CONDUCTIVITY': [1e-10, 1e-8], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'CHITIN': {'DISPERSION_MODEL': 'DEBYE', 
                                    'EPS_STATIC': 4.0, 
                                    'EPS_INF': 2.7, 
                                    'TAU': 1e-11, 
                                    'CONDUCTIVITY': [1e-7, 1e-5], 
                                    'PERMEABILITY_RELATIVE': 1.0, 
                                    'MAGNETIC_LOSS': 0.0},
                            'GLYCOGEN': {'DISPERSION_MODEL': 'DEBYE', 
                                        'EPS_STATIC': 4.0, 
                                        'EPS_INF': 2.8, 
                                        'TAU': 1e-11, 
                                        'CONDUCTIVITY': [1e-7, 1e-5], 
                                        'PERMEABILITY_RELATIVE': 1.0, 
                                        'MAGNETIC_LOSS': 0.0},}}
