import sys
import json
from pathlib import Path

if __name__ == '__main__':
    sys.modules.setdefault('EMSimulation', sys.modules[__name__])

# Imports
from EMSimulationParameters import COOKIE_FO_SIZES_MM
from EMSimulationModel import ModelMixin
from EMSimulationSolver import SolverMixin
from EMSimulationAnalysis import AnalysisMixin
from EMSimulationPlots import PlotsMixin
from EMSimulationStudies import StudiesMixin

# Get module level functions of the sub-modules
from EMSimulationExports import *
from EMSimulationExports import __all__ as _EXPORTS

__all__ = ['EMSimulation', *_EXPORTS]

print("Current debugger:", sys.executable)


class EMSimulation(PlotsMixin, ModelMixin, SolverMixin, AnalysisMixin, StudiesMixin):
    """Electromagnetic (gprMax / FMCW radar) simulation of foreign-object detection in food samples on conveyor belts. It sets up the model, runs gprMax simulation to gather the electromagnetic fields, reconstructs the MIMO channel matrix, analyzes and plots the results. Furthermore, it runs studies (DOE, cookie detectability, MIMO channel sensitivity, source-array subsampling, moving-sample scan etc.), applies analysis and creates plots.
    """

if __name__ == "__main__":

    # Run settings (MODE, parameter files, study folders, per-study options) live in EMSimulationSettings.py.
    from EMSimulationSettings import *

    # This mode runs a single simulation with the specified parameter file and constant file.
    if MODE == 0:

        EMSimulation.preflight_resource_check(SINGLE_RUN_PARMS_FILE, 
                                              CONSTS_FILE, 
                                              'input_output_files')

        em_simulation_1 = EMSimulation(
            parms=SINGLE_RUN_PARMS_FILE, 
            consts=CONSTS_FILE)
        
        em_simulation_1.simulate()

    # This mode analyzes single simulation results after running MODE 0 using baseline data for comparison.
    elif MODE == 1:

        em_simulation_1 = EMSimulation(parms=PARMS_FILE, 
                                       consts=CONSTS_FILE)
        
        em_simulation_1.analyze(
            results_folder=ANALYZE_RESULTS_FOLDERS,
            baseline_folder=ANALYZE_BASELINE_FOLDER)

    # This mode runs a batch of simulations based on the DOE CSV file.
    elif MODE == 2:

        run_doe_batch(
            DOE_CSV_FILE, 
            PARMS_FILE, 
            CONSTS_FILE,
            retry_failed=RETRY_FAILED, 
            run_subset=DOE_RUN_SUBSET)

    # This mode analyzes the results of a DOE batch after running MODE 2.
    elif MODE == 3:

        # Analyze the DOE batch results using the specified manifest and baseline data.
        em_simulation_1 = EMSimulation(parms=PARMS_FILE, 
                                       consts=CONSTS_FILE)
        
        em_simulation_1.analyze_doe_batch(
            doe_csv=DOE_CSV_FILE,
            baseline_folder=DOE_BASELINE_FOLDER,
            manifest=DOE_MANIFEST_FILE,
            output_csv=DOE_OUTPUT_CSV)

        # Export paper-ready figures and tables for the full-factorial DOE studies.
        export_full_factorial_paper_assets(
            study_dirs=DOE_PAPER_STUDY_DIRS, 
            output_dir=DOE_PAPER_ASSETS_DIR)

    # This mode performs MIMO channel-count sensitivity analysis using pre-collected DOE data.
    elif MODE == 4:

        # Analyze MIMO channel-count sensitivity on the DOE's collected 5x5-channel data
        analyze_mimo_channel_sensitivity(
            study_dir=MIMO_STUDY_DIR,
            output_dir=MIMO_OUTPUT_DIR,
            channel_counts=MIMO_CHANNEL_COUNTS,
            n_draws=MIMO_N_DRAWS,
            response_metric=MIMO_RESPONSE_METRIC,
            seed=MIMO_SEED)

        # Analyze the effect of the specific random channel subset on the response metric, while holding the draw axis fixed across DOE runs.
        analyze_mimo_channel_draw_surface(
            study_dir=MIMO_STUDY_DIR,
            output_dir=MIMO_OUTPUT_DIR,
            channel_counts=MIMO_CHANNEL_COUNTS,
            n_draws=MIMO_N_DRAWS,
            response_metric=MIMO_RESPONSE_METRIC,
            seed=MIMO_SEED)

        # Analyze the similarity of the per-channel phase deviations to understand why the coherent average depends on N.
        analyze_mimo_channel_similarity(
            study_dir=MIMO_STUDY_DIR, 
            output_dir=MIMO_OUTPUT_DIR)

    # Verify source-array subsampling stride (grating-lobe criterion)
    elif MODE == 5:

        verify_source_array_subsampling(
            output_dir=SRC_STRIDE_CHECK_OUTPUT_DIR, parms_file=SRC_STRIDE_CHECK_PARMS_FILE,
            consts_file=CONSTS_FILE)

    # FO-size study: simulation and evaluation of cookie detectability with varying focal-object sizes.
    elif MODE == 6:

        EMSimulation.run_cookie_detectability_fo_size_simulation(
            base_parms_file=COOKIE_PARMS_FILE, 
            consts_file=CONSTS_FILE,
            cookie_parms_files=COOKIE_PARMS_FILES,
            fo_sizes_mm=COOKIE_FO_SIZES_MM, 
            fo_material=COOKIE_FO_MATERIAL,
            output_dir=COOKIE_STUDY_DIR, 
            retry_failed=RETRY_FAILED)

    # FO-size study: evaluation of cookie detectability with varying focal-object sizes.
    elif MODE == 7:

        EMSimulation.show_cookie_detectability_fo_size_results(
            base_parms_file=COOKIE_PARMS_FILE, 
            output_dir=COOKIE_STUDY_DIR,
            response_metrics=COOKIE_RESPONSE_METRICS,
            detection_rule=COOKIE_DETECTION_RULE, snr_db_sweep=COOKIE_SNR_SWEEP_DB, 
            rx_sides=COOKIE_RX_SIDES,
            cookie_parms_files=COOKIE_PARMS_FILES)

    # FO-position study: simulation and evaluation of cookie detectability with varying focal-object positions.
    elif MODE == 8:

        EMSimulation.run_cookie_detectability_fo_position_simulation(
            base_parms_file=COOKIE_PARMS_FILE, 
            consts_file=CONSTS_FILE,
            cookie_parms_files=COOKIE_PARMS_FILES,
            fo_size_mm=COOKIE_FO_POSITION_SIZE_MM,
            n_positions=COOKIE_FO_POSITION_N_STEPS,
            axis=COOKIE_FO_POSITION_AXIS,
            edge_clearance_mm=COOKIE_FO_POSITION_EDGE_CLEARANCE_MM,
            fo_material=COOKIE_FO_MATERIAL, output_dir=COOKIE_POSITION_STUDY_DIR,
            retry_failed=RETRY_FAILED)

    # FO-position study: evaluation of cookie detectability with varying focal-object positions.
    elif MODE == 9:

        EMSimulation.show_cookie_detectability_fo_position_results(
            base_parms_file=COOKIE_PARMS_FILE, output_dir=COOKIE_POSITION_STUDY_DIR,
            response_metrics=COOKIE_RESPONSE_METRICS,
            detection_rule=COOKIE_DETECTION_RULE, snr_db_sweep=COOKIE_SNR_SWEEP_DB, 
            rx_sides=COOKIE_RX_SIDES)

    # Frequency study: simulation of cookie detectability with varying frequency bands.
    elif MODE == 10:

        EMSimulation.run_cookie_detectability_frequency_simulation(
            base_parms_file=COOKIE_PARMS_FILE, 
            consts_file=CONSTS_FILE,
            cookie_parms_files=(COOKIE_PARMS_FILES[:1] if COOKIE_FREQUENCY_PILOT
                                else COOKIE_PARMS_FILES),
            frequency_bands_ghz=(sorted(COOKIE_FREQUENCY_BANDS_GHZ)[-1:]
                                 if COOKIE_FREQUENCY_PILOT and not COOKIE_FREQUENCY_WIDEBAND else COOKIE_FREQUENCY_BANDS_GHZ),
            fo_size_mm=COOKIE_FREQUENCY_FO_SIZE_MM,
            fo_material=COOKIE_FO_MATERIAL, output_dir=COOKIE_FREQUENCY_STUDY_DIR + ('/wideband' if COOKIE_FREQUENCY_WIDEBAND else ''),
            reuse_reference_study_dir=(COOKIE_POSITION_STUDY_DIR
                                       if COOKIE_FREQUENCY_REUSE_POSITION_RUNS else None),
            retry_failed=RETRY_FAILED, 
            wideband=COOKIE_FREQUENCY_WIDEBAND,
            pml_wavelengths=COOKIE_FREQUENCY_WIDEBAND_PML_WAVELENGTHS, parms_overrides=COOKIE_FREQUENCY_PARMS_OVERRIDES)

    # Frequency study: evaluation of cookie detectability with varying frequency bands.
    elif MODE == 11:

        EMSimulation.show_cookie_detectability_frequency_results(
            base_parms_file=COOKIE_PARMS_FILE, output_dir=COOKIE_FREQUENCY_STUDY_DIR + ('/wideband' if COOKIE_FREQUENCY_WIDEBAND else ''),
            response_metrics=COOKIE_RESPONSE_METRICS, snr_db_sweep=COOKIE_SNR_SWEEP_DB, 
            rx_sides=COOKIE_RX_SIDES)

    # Moving sample line array study: simulation of cookie detectability with a moving sample and line array receiver.
    elif MODE == 12:

        EMSimulation.run_moving_sample_line_array_simulation(
            base_parms_file=MSLA_BASE_PARMS_FILE, 
            consts_file=CONSTS_FILE,
            cookie_parms_file=MSLA_COOKIE_PARMS_FILE,
            fo_size_mm=MSLA_FO_SIZE_MM, fo_position_fraction=MSLA_FO_POSITION_RELATIVE,
            fo_material=COOKIE_FO_MATERIAL, 
            n_rx=MSLA_N_RX, rx_pitch_mm=MSLA_RX_PITCH_MM,
            domain_width_factor=MSLA_DOMAIN_WIDTH_FACTOR,
            rx_gap_below_belt_mm=MSLA_RX_GAP_BELOW_BELT_MM,
            standoff_m=MSLA_TX_STANDOFF_M, 
            step_mm=MSLA_STEP_MM,
            scan_half_range_mm=MSLA_SCAN_HALF_RANGE_MM,
            domain_length_factor=MSLA_DOMAIN_LENGTH_FACTOR,
            output_dir=MSLA_STUDY_DIR, 
            retry_failed=RETRY_FAILED,
            pilot=MSLA_PILOT, 
            dry_run=MSLA_DRY_RUN,
            simulate_baseline=MSLA_SIMULATE_BASELINE, simulate_empty=MSLA_SIMULATE_EMPTY,
            fo_lift_mm=MSLA_FO_LIFT_MM, 
            fo_air_gap_mm=MSLA_FO_AIR_GAP_MM)

    # Moving sample line array study: evaluation of cookie detectability with a moving sample and line array receiver.
    elif MODE == 13:

        analyze_moving_sample_line_array(
            study_dir=MSLA_STUDY_DIR, 
            show=True)

    # THz line camera study: evaluation of cookie detectability with a moving sample and line camera receiver.
    elif MODE == 14:

        # Simulates the moving sample camera with the specified parameters.
        analyze_moving_sample_camera(
            study_dir=MSLA_STUDY_DIR, 
            sheets=CAMERA_SHEETS, 
            atten_db_per_sheet=CAMERA_ATTEN_DB_PER_SHEET,
            freq_ghz=CAMERA_FREQ_GHZ, 
            aperture_mm=CAMERA_APERTURE_MM, 
            smear_x_mm=CAMERA_SMEAR_X_MM,
            pitch_mm=CAMERA_PITCH_MM, 
            step_mm=CAMERA_STEP_MM, 
            interp=CAMERA_INTERP,
            noise_rel=CAMERA_NOISE_REL, 
            gain_spread=CAMERA_GAIN_SPREAD, 
            offset_rel=CAMERA_OFFSET_REL,
            adc_bits=CAMERA_ADC_BITS, 
            full_scale=CAMERA_FULL_SCALE, 
            seed=CAMERA_SEED,
            display=CAMERA_DISPLAY, 
            reference_power=CAMERA_REFERENCE_POWER,
            reference_power_db=CAMERA_REFERENCE_POWER_DB, 
            show=True)
        
        # Compares the measured reference images with the simulated 'with object' images for visual inspection.
        plot_camera_measured_vs_simulated(
            study_dir=MSLA_STUDY_DIR, 
            sheets=CAMERA_SHEETS, 
            full_scale=CAMERA_FULL_SCALE, 
            display=CAMERA_DISPLAY,
            out_path=Path(__file__).resolve().parent / MSLA_STUDY_DIR / 'results_analysis' / 'camera_measured_vs_simulated.png', 
            show=True)

    # This mode generates paper figures for the Gaussian excitation pulse and MIMO TX/RX array geometry.
    elif MODE == 15:

        with open(PARMS_FILE) as f:
            _parms = json.load(f)

        # Gaussian pulse
        plot_gaussian_pulse(
            frequency_range=tuple(_parms['FREQUENCY_RANGE']), 
            amplitude=_parms['TX_AMPLITUDE'],
            out_path=Path(__file__).resolve().parent / 'exports' / 'gaussian_pulse_time_frequency.png', 
            show=True)

        # MIMO array geometry
        plot_mimo_tx_rx_geometry(
            _parms, 
            out_path=Path(__file__).resolve().parent / 'exports' / 'mimo_tx_rx_geometry.png',
            show=True)

    pass
