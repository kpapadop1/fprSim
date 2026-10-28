import copy
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from EMSimulationHelpers import (
    _cookie_metric_short_name, _cookie_short_name, _format_duration, _freeform_fo_position_fraction, _freeform_outline,
    _freeform_region_outline, _get_sut_thickness_m, _rescale_fractions, _resolve_cookie_run_folder)
from EMSimulationModel import ModelMixin
from EMSimulationParameters import (
    _INGREDIENT_DENSITY_G_CM3, COOKIE_FO_SIZES_MM, FO_MATERIALS_ALLOWED, POROUS_AIR_FRACTION, POSITION_X_FRACTION,
    WATER_FRACTION_HIGH)
from EMSimulationPlots import (
    _plot_cookie_detectability_comparison_journal, _plot_cookie_detectability_frequency, _plot_cookie_detectability_journal,
    _plot_cookie_detectability_position, _plot_cookie_detectability_position_journal, _plot_cookie_mds_side_comparison,
    _plot_cookie_position_comparison_journal, plot_freeform_cookie_geometry)

_PACKING_CACHE = {}

# region StudiesMixin

class StudiesMixin:
    """This mixin class provides study drivers for EMSimulation, including DOE batch, cookie studies, moving sample studies, parameter building, sphere packing, scoring, and paper tables."""

    @classmethod
    def run_cookie_detectability_fo_size_simulation(cls, 
            base_parms_file: str, 
            consts_file: str, 
            cookie_parms_files: list, 
            fo_sizes_mm: list = None, 
            fo_material: str = 'PVC',
            output_dir: str = 'studies/cookie detectability study/fo size/transmission',
            retry_failed: bool = True, 
            homogenize_layers: bool = False,
            stop_if_first_run_fails: bool = True) -> 'pd.DataFrame':
        """This function runs the foreign-object size study for cookie detectability (MODE 6). For every product, it executes one no-anomaly baseline run plus one run per centered foreign-object cube size.

        Args:
            base_parms_file (str): 
                PARMS_FILE; everything except SUT/FOREIGN_OBJECT (and a product's optional GRID override) is used unchanged.
            consts_file (str): 
                CONSTS_FILE.
            cookie_parms_files (list): 
                Overlay files parameters/cookie_types/*.json (COOKIE_NAME, SUT block, optional GRID).
            fo_sizes_mm (list): 
                Cube edge lengths in mm; default COOKIE_FO_SIZES_MM (10 -> 1 mm).
            fo_material (str): 
                Foreign-object material key (default: 'PVC').
            output_dir (str): 
                Study folder (created if missing).
            retry_failed (bool): 
                Retry FAILED manifest rows.
            homogenize_layers (bool): 
                Flatten layered products into one mixture, for comparison only (default False)
            stop_if_first_run_fails (bool): 
                Raise a RuntimeError if the first run of this call fails, instead of continuing (default: True).

        Returns:
            pandas.DataFrame: The manifest (one row per run).

        Raises:
            RuntimeError: If the first run fails and stop_if_first_run_fails is True.
        """
        # Determine the foreign-object sizes to be used for this study.
        fo_sizes_mm = list(COOKIE_FO_SIZES_MM if fo_sizes_mm is None else fo_sizes_mm)
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load the base parameters and prepare the study directory.
        base_parms = cls._load_config(base_parms_file)
        if isinstance(base_parms.get('FREQUENCY_RANGE'), list):
            base_parms['FREQUENCY_RANGE'] = tuple(base_parms['FREQUENCY_RANGE'])
        base_parms['PATH_INPUT_OUTPUT_FILES'] = str(study_dir / 'input_output_files')

        # Perform a preflight check to ensure sufficient disk and GPU memory is available.
        cls.preflight_resource_check(base_parms, consts_file, study_dir / 'input_output_files')

        # Load the overlay configurations for each cookie type.
        overlays = [cls._load_config(p) for p in cookie_parms_files]
        overlay_files = {ov['COOKIE_NAME']: p for ov, p in zip(overlays, cookie_parms_files)}

        # Build the list of jobs to run: one per cookie type and foreign-object size. Skip sizes that do not fit.
        jobs = []
        for ov in overlays:
            thickness_mm = _cookie_total_thickness_mm(ov['SUT'])
            fitting = [s for s in fo_sizes_mm if s <= thickness_mm + 1e-9]
            skipped = [s for s in fo_sizes_mm if s not in fitting]
            if skipped:
                print(f"run_cookie_detectability_fo_size_simulation(): {ov['COOKIE_NAME']} is {thickness_mm:g} mm thick - skipping FO size(s) {skipped} mm (cube would not fit).")
            fo_clear_mm = max(fitting) if fitting else min(fo_sizes_mm)
            jobs.append(dict(cookie=ov['COOKIE_NAME'], 
                             overlay=ov, 
                             fo_size_mm='base',
                             baseline=True, 
                             thickness_mm=thickness_mm,
                             fo_clear_mm=fo_clear_mm,
                             build_size=min(fitting) if fitting else min(fo_sizes_mm)))
            for s in sorted(fitting, reverse=True):
                jobs.append(dict(cookie=ov['COOKIE_NAME'], 
                                 overlay=ov, 
                                 fo_size_mm=s,
                                 baseline=False, 
                                 thickness_mm=thickness_mm,
                                 fo_clear_mm=fo_clear_mm, 
                                 build_size=s))

        # Only process jobs that are baseline and have discrete inclusions.
        for job in jobs:
            if not job['baseline'] or not any('DISCRETE' in c for c in
                                               job['overlay']['SUT'].get('COMPONENTS', [])):
                continue

            # Build parameters for the current cookie run and extract inclusions info.
            inc = _build_parms_for_cookie_run(base_parms, 
                                              job['overlay'], 
                                              job['build_size'], 
                                              fo_material, 
                                              baseline=True, 
                                              fo_clear_mm=job['fo_clear_mm'])['SUT']['INCLUSIONS']
            info = inc['INFO']
            print(f"cookie study: {job['cookie']}: {info['n_placed']} of {info['n_target']} "
                  f"{info['NAME']} spheres ({1e3 * inc['DIAMETER']:g} mm) placed = "
                  f"{100 * info['share_in_spheres']:.1f} % of the volume (target "
                  f"{100 * info['share_target']:.0f} %; the other "
                  f"{100 * info['share_in_matrix_as_crumbs']:.1f} % stays in the matrix "
                  f"as fine crumbs); clearance for a {job['fo_clear_mm']:g} mm object.")
            slug = re.sub(r'[^A-Za-z0-9]+', '_', job['cookie']).strip('_').lower()

            # Save the inclusion centers to a CSV file for the current cookie run.
            pd.DataFrame(inc['CENTERS'], 
                         columns=['x_m', 'y_m', 'z_above_belt_m']).to_csv(
                out_dir / f'chip_layout_{slug}.csv', index=False)

        # Prepare the manifest file for resumable runs. Load existing manifest if it exists, otherwise create a new one.
        manifest_path = out_dir / 'manifest_cookie_size_sweep.csv'
        columns = ['cookie', 
                   'fo_size_mm', 
                   'baseline', 
                   'thickness_mm',
                   'results_folder', 
                   'status', 
                   'error']
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path, 
                                   dtype={'fo_size_mm': str, 
                                          'results_folder': str, 
                                          'error': str})
        else:
            manifest = pd.DataFrame(columns=columns)

        # Define a helper function to find a job in the manifest by cookie and FO size.
        def _find(cookie, size):
            hit = manifest[(manifest['cookie'] == cookie)
                           & (manifest['fo_size_mm'] == str(size))]
            
            return hit.index[0] if len(hit) else None

        # Loop over all jobs and process them according to the manifest status.
        t_done, n_done = 0.0, 0
        for j, job in enumerate(jobs):

            # Check if this job has already been processed in the manifest.
            idx = _find(job['cookie'], job['fo_size_mm'])
            status = manifest.loc[idx, 'status'] if idx is not None else None
            if status == 'OK' or (status == 'FAILED' and not retry_failed):
                continue
            print(f"\n=== cookie study run {j + 1}/{len(jobs)}: {job['cookie']}, FO = {'none (baseline)' if job['baseline'] else str(job['fo_size_mm']) + ' mm ' + fo_material} ===")

            t0 = time.time()
            row = dict(cookie=job['cookie'], 
                       fo_size_mm=str(job['fo_size_mm']),
                       baseline=job['baseline'], 
                       thickness_mm=job['thickness_mm'])
            
            # Try to build the parameters and run the simulation for this job.
            try:

                parms = _build_parms_for_cookie_run(
                                            base_parms, job['overlay'], 
                                            job['build_size'], 
                                            fo_material,
                                            baseline=job['baseline'], homogenize_layers=homogenize_layers,
                                            fo_clear_mm=job['fo_clear_mm'])
                em = cls(parms=parms, consts=consts_file)
                em.protocol_context = dict(
                                study='cookie detectability study, FO size', base_parms_file=base_parms_file,
                                cookie_parms_file=overlay_files[job['cookie']], consts_file=consts_file,
                                cookie=job['cookie'], 
                                fo_size_mm=job['fo_size_mm'], 
                                fo_material=fo_material,
                                baseline=job['baseline'])
                em.simulate()

                row.update(results_folder=Path(os.path.relpath(em.path_folder, study_dir)).as_posix(), status='OK', error='')
                
            except Exception as e:

                print(f"cookie study run FAILED - {e!r}")
                row.update(results_folder='', status='FAILED', error=repr(e))

            finally:
                plt.close('all')

            # Update the manifest with the results of this job and save it.
            if idx is None:
                manifest = pd.concat([manifest, 
                                      pd.DataFrame([row])], 
                                      ignore_index=True)
            else:
                for k, v in row.items():
                    manifest.loc[idx, k] = v
            manifest.to_csv(manifest_path, index=False)

            # If the first run failed and we are instructed to stop, raise an error.
            if row['status'] == 'FAILED' and n_done == 0 and stop_if_first_run_fails:
                raise RuntimeError(
                    f"run_cookie_detectability_fo_size_simulation(): the first run ({job['cookie']}, FO = {job['fo_size_mm']}) failed: {row['error']} - stopped before the rest of the batch (see the manifest in {out_dir}; pass stop_if_first_run_fails=False to continue anyway).")
            n_done += 1
            t_done += time.time() - t0
            left = sum(1 for jb in jobs[j + 1:]
                       if (lambda i: i is None or manifest.loc[i, 'status'] != 'OK')(
                           _find(jb['cookie'], jb['fo_size_mm'])))
            print(f"cookie study: {n_done} simulated this session, avg {_format_duration(t_done / n_done)}/run, {left} left - ETA {_format_duration(left * t_done / n_done)}.")


        # Check for completeness of the simulation results.
        ok = manifest[manifest['status'] == 'OK']

        # When ALSO_MIRRORED_RX, every run folder must hold data_reflection.pkl and data_transmission.pkl (and data.pkl) 
        need_files = (['data_reflection.pkl', 'data_transmission.pkl'] if base_parms.get('ALSO_MIRRORED_RX', False) else ['data.pkl'])
        
        # Identify OK runs that are missing or have failed.
        missing = [r for r in ok['results_folder']
                   if not all((_resolve_cookie_run_folder(study_dir, r) / fn).is_file() for fn in need_files)]
        n_failed = int((manifest['status'] == 'FAILED').sum())

        # Number of runs that are still open (not OK and not FAILED).
        n_open = len(jobs) - len(ok)

        print(f"\nrun_cookie_detectability_fo_size_simulation(): {len(ok) - len(missing)} of {len(jobs)} runs have a data.pkl; {n_failed} FAILED, {n_open} not OK yet{f', {len(missing)} OK rows lack data.pkl: {missing}' if missing else ''}.")

        # If there are no open runs and no missing files, all data has been collected.
        if not n_open and not missing:
            print("run_cookie_detectability_fo_size_simulation(): all data collected - run MODE == 7 (show_cookie_detectability_fo_size_results()) next.")

        return manifest

    @classmethod
    def show_cookie_detectability_fo_size_results(cls, 
            base_parms_file: str,
            output_dir: str = 'studies/cookie detectability study/fo size/transmission',
            response_metrics: tuple = ('rms_dphase_max', 'rms_dmag_max'),
            detection_rule: str = 'envelope',
            snr_db_sweep: tuple = (20, 30, 40),
            rx_sides: tuple = None,
            cookie_parms_files: list = None) -> 'pd.DataFrame':
        """This function scores the foreign-object runs against their respective baselines and generates the corresponding results and figures (MODE 7; results from MODE 6 required).

        Args:
            base_parms_file (str): 
                PARMS_FILE of the runs; only its AMP_NOISE_FLOOR_DB / PHASE_NOISE_FLOOR_DEG are used.
            output_dir (str): 
                Study folder (reads manifest_cookie_size_sweep.csv from results_analysis/, writes the outputs there).
            response_metrics (tuple): 
                Columns of _compute_differential_metrics() to plot/threshold, one panel each. Default ('rms_dphase_max', 'rms_dmag_max').
            detection_rule (str): 
                'envelope' (default) or 'conservative'
            snr_db_sweep (tuple): 
                Assumed per-bin SNRs [dB] for the sensitivity table; () skips it.
            rx_sides (tuple): 
                None = all sides found; or e.g. ('reflection',) / ('transmission',) / ('reflection', 'transmission').
            cookie_parms_files (list): 
                Product overlay files (parameters/cookie_types/*.json) for cookie_detectability_fo_size_products.csv; None = skip it.

        Returns:
            pandas.DataFrame: Per-side, per-product, per-size results (empty if scoring not possible); column rx_side.

        Raises:
            FileNotFoundError: If the manifest does not exist (run MODE 6 first).
            ValueError: If a response metric is not produced by _compute_differential_metrics().
        """
        response_metrics = list(response_metrics)

        # Ensure detection_rule is valid
        if detection_rule not in ('conservative', 'envelope'):
            raise ValueError(f"show_cookie_detectability_fo_size_results(): unknown detection_rule "
                             f"{detection_rule!r} (use 'conservative' or 'envelope').")

        # Prepare study and output directories, and read the manifest
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        manifest_path = out_dir / 'manifest_cookie_size_sweep.csv'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"show_cookie_detectability_fo_size_results(): no manifest at {manifest_path} - run MODE 6 (run_cookie_detectability_fo_size_simulation()) first.")
        manifest = pd.read_csv(manifest_path, 
                               dtype={'fo_size_mm': str, 
                                      'results_folder': str, 
                                      'error': str})

        # Load base parameters and compute noise floors for each response metric
        base_parms = cls._load_config(base_parms_file)
        floors = {m: _floor_for_metric(m, base_parms) for m in response_metrics}
        amp_floor = base_parms.get('AMP_NOISE_FLOOR_DB')
        phase_floor = base_parms.get('PHASE_NOISE_FLOOR_DEG')

        # Extract unique cookie names from the manifest in the order they appear
        cookie_names = list(dict.fromkeys(manifest['cookie']))

        # Define a helper function to determine which receiver sides have data in a given folder
        def _sides_in(folder):
            found = [sd for sd in ('reflection', 'transmission') if (Path(folder) / f'data_{sd}.pkl').is_file()]
            
            return found or (['legacy'] if (Path(folder) / 'data.pkl').is_file() else [])

        # Identify the baseline run(s) with 'OK' status.
        ok_base = manifest[(manifest['fo_size_mm'] == 'base') & (manifest['status'] == 'OK')]
        if ok_base.empty:
            print("show_cookie_detectability_fo_size_results(): no OK baseline run - nothing to score.")
            return pd.DataFrame()

        # Determine which receiver sides have data for the baseline run.
        detected = _sides_in(_resolve_cookie_run_folder(study_dir, 
                                                        ok_base['results_folder'].iloc[0]))
        sides = list(detected if rx_sides is None else rx_sides)
        for sd in sides:
            if sd not in detected:
                raise FileNotFoundError(f"show_cookie_detectability_fo_size_results(): rx_side {sd!r} requested but the baseline run folder only holds {detected}.")
        if not sides:
            print("show_cookie_detectability_fo_size_results(): no data.pkl / data_<side>.pkl found - nothing to score.")
            return pd.DataFrame()
        print(f"show_cookie_detectability_fo_size_results(): scoring receiver side(s): {sides}")

        # Define a helper function to load run data for a given folder and receiver side
        def _load(folder, sd):
            return cls._load_run_data(_resolve_cookie_run_folder(study_dir, folder), verbose=False, side=None if sd == 'legacy' else sd)

        # Iterate over each receiver side and then each cookie name to compute differential metrics
        records = []
        for sd in sides:
            for name in cookie_names:

                # Identify the baseline run for the current cookie name. Check if it exists and skip if not.
                b = manifest[(manifest['cookie'] == name) & (manifest['fo_size_mm'] == 'base') & (manifest['status'] == 'OK')]
                if b.empty:
                    print(f"show_cookie_detectability_fo_size_results(): no OK baseline for {name} - skipped.")
                    continue

                # Load the baseline run data for the current receiver side and cookie name
                H_b, f_b, *_ = _load(b['results_folder'].iloc[0], sd)
                runs = manifest[(manifest['cookie'] == name) & (manifest['fo_size_mm'] != 'base') & (manifest['status'] == 'OK')]

                # Iterate over each non-baseline run for the current cookie and receiver side
                for _, r in runs.iterrows():

                    # Load the current run data for the current receiver side and cookie name. Compute differential metrics relative to the baseline.
                    H, f, *_ = _load(r['results_folder'], sd)
                    metrics = cls._compute_differential_metrics(
                        H, f, H_b, f_b, 
                        amp_noise_floor_db=amp_floor,
                        phase_noise_floor_deg=phase_floor)

                    # Check that all requested response metrics are available in the computed metrics
                    for m in response_metrics:
                        if m not in metrics:
                            raise ValueError(
                                f"show_cookie_detectability_fo_size_results(): unknown response metric {m!r}; available: {sorted(metrics)}")
                    records.append(dict(rx_side=sd, 
                                        cookie=name, 
                                        fo_size_mm=float(r['fo_size_mm']),
                                        thickness_mm=r['thickness_mm'], **metrics))
        
        # If no records were collected, print a message and return an empty DataFrame
        if not records:
            print("show_cookie_detectability_fo_size_results(): nothing to score yet.")
            return pd.DataFrame()
        
        # Convert the collected records into a DataFrame and sort by receiver side, cookie, and field of view size
        results_all = pd.DataFrame(records)
        results_all['rx_side'] = pd.Categorical(results_all['rx_side'], 
                                                categories=sides, 
                                                ordered=True)
        results_all = results_all.sort_values(['rx_side', 
                                               'cookie', 
                                               'fo_size_mm'], 
                                               ascending=[True, True, False]).reset_index(drop=True)
        results_all['rx_side'] = results_all['rx_side'].astype(str)
        results_all.to_csv(out_dir / 'cookie_detectability_fo_size_results.csv', index=False)

        # Iterate over each receiver side to compute metrics and generate plots
        mds_parts, ratio_parts, sens_rows = [], [], []
        for sd in sides:

            # Filter results for the current receiver side
            suffix = '' if sd == 'legacy' else f'_{sd}'
            results = results_all[results_all['rx_side'] == sd].drop(columns='rx_side').reset_index(drop=True)
            if results.empty:
                continue

            # Compute the smallest detectable size (MDS) for the current receiver side
            mds_df = _cookie_score_mds(results, 
                                       response_metrics, 
                                       floors, 
                                       detection_rule)
            mds_parts.append(mds_df.assign(rx_side=sd))

            # Generate a plot for cookie detectability for the current receiver side
            _plot_cookie_detectability_journal(results, 
                                               mds_df, 
                                               response_metrics, 
                                               floors,
                                               out_dir / f'cookie_detectability_fo_size{suffix}.png')

            # Compute the ratio of response to floor for each product and size  
            ratio_df = results[['cookie', 'fo_size_mm']].copy()
            worst = np.zeros(len(results))
            for m in response_metrics:
                if floors[m] is None:
                    continue
                r = results[m].to_numpy() / floors[m]
                ratio_df[f'{_cookie_metric_short_name(m)}_over_floor'] = np.round(r, 3)
                worst = np.maximum(worst, r)
            ratio_df['combined_over_floor'] = np.round(worst, 3)
            
            # Include additional columns for exceedance counts if available
            for c in ('n_exceed_max', 'n_exceed_mean'):
                if c in results:
                    ratio_df[c] = results[c].to_numpy()
            ratio_parts.append(ratio_df.assign(rx_side=sd))

            # Sweep over assumed receiver SNR values to assess sensitivity of the limit
            for snr in snr_db_sweep:
                fl = {m: _crb_floor_for_metric(m, snr) for m in response_metrics}
                d = _cookie_score_mds(results, response_metrics, fl, detection_rule)
                for _, r in d.iterrows():
                    row = dict(rx_side=sd, snr_db=snr, cookie=r['cookie'])
                    for m in response_metrics:
                        row[f'floor_{_cookie_metric_short_name(m)}'] = (
                            None if fl[m] is None else round(fl[m], 4))
                    for k in list(response_metrics) + ['combined']:
                        row[f'mds_mm_{k}'] = r[f'mds_mm_{k}']
                        row[f'status_{k}'] = r[f'status_{k}']
                    sens_rows.append(row)

        # Combine and export results for all receiver sides
        mds_all = pd.concat(mds_parts, ignore_index=True)
        mds_all = mds_all[['rx_side'] + 
                          [c for c in mds_all.columns if c != 'rx_side']]
        mds_all.to_csv(out_dir / 'cookie_detectability_fo_size_mds_summary.csv', index=False)

        # Combine and export ratio results for all receiver sides
        ratio_all = pd.concat(ratio_parts, ignore_index=True)
        ratio_all = ratio_all[['rx_side'] + [c for c in ratio_all.columns if c != 'rx_side']]
        ratio_all.to_csv(out_dir / 'cookie_detectability_fo_size_ratio_by_size.csv', index=False)

        # Combine and export sensitivity results for all receiver sides
        if sens_rows:
            pd.DataFrame(sens_rows).to_csv(out_dir / 'cookie_detectability_fo_size_snr_sensitivity.csv', index=False)
        if len([sd for sd in sides if sd != 'legacy']) > 1:
            _plot_cookie_mds_side_comparison(mds_all, 
                                             response_metrics, out_dir / 'cookie_detectability_fo_size_mds_reflection_vs_transmission.png')

        # Prepare results for export and further analysis
        results = results_all
        mds_df = mds_all
        _export_cookie_size_paper_tables(mds_all, 
                                         ratio_all, 
                                         pd.DataFrame(sens_rows), 
                                         sides, 
                                         manifest, 
                                         cookie_parms_files, 
                                         out_dir)

        # Check if the product ranking is consistent across different response metrics. Iterate over each receiver side and compare rankings.
        for sd in sides:
            
            rankings = {}
            sub = mds_df[mds_df['rx_side'] == sd]

            # Compute the product ranking for each response metric based on the smallest detectable size.
            for m in response_metrics + ['combined']:
                col = sub.set_index('cookie')[f'mds_mm_{m}'].dropna().sort_values()
                rankings[m] = list(col.index)
            
            if len({tuple(r) for r in rankings.values()}) > 1:
                print(f"show_cookie_detectability_fo_size_results(): NOTE - [{sd}] product ranking by smallest detectable size differs between metrics:")
                for m, r in rankings.items():
                    print(f"    {m}: {' < '.join(r)}")
                    
        print(f"\nshow_cookie_detectability_fo_size_results(): done ({detection_rule} rule) - see {out_dir}")
        print(mds_df.to_string(index=False))
        
        return results

    @classmethod
    def run_cookie_detectability_fo_position_simulation(cls, 
            base_parms_file: str, 
            consts_file: str, 
            cookie_parms_files: list,
            fo_size_mm: float = 5.0, 
            n_positions: int = 10, 
            axis: str = 'x',
            edge_clearance_mm: float = 1.0, 
            fo_material: str = 'PVC',
            output_dir: str = 'studies/cookie detectability study/fo position/transmission',
            retry_failed: bool = True, 
            homogenize_layers: bool = False,
            stop_if_first_run_fails: bool = True) -> 'pd.DataFrame':
        """This function runs the foreign-object position simulation for the cookie detectability study (MODE 8). Size is fixed and the foreign-object position is swept along the specified axis (until clearance with the SUT boundary is reached). For every product, one run is performed per position and one baseline run at the center. Unfeasible positions or unfitting products are skipped.

        Args:
            base_parms_file (str): 
                Base parameter file
            consts_file (str): 
                Constants file
            cookie_parms_files (list): 
                Overlay files parameters/cookie_types/*.json (as in the size study).
            fo_size_mm (float): 
                Cube edge length in mm, fixed for the study
            n_positions (int): 
                Positions per product, including center and edge point (default 10).
            axis (str): 
                'x' (across belt width, default) or 'y': the swept in-plane POSITION_RELATIVE axis.
            edge_clearance_mm (float): 
                Minimum clearance in mm between the cube and the SUT boundary at the farthest position (default 1).
            fo_material (str): 
                Foreign-object material key (default 'PVC').
            output_dir (str): 
                Study folder (created if missing).
            retry_failed (bool): 
                Retry FAILED manifest rows.
            homogenize_layers (bool): 
                Flatten layered products into one mixture, for comparison only (default False).
            stop_if_first_run_fails (bool): 
                Raise a RuntimeError if the first run of this call fails (default True).

        Returns:
            pandas.DataFrame: The manifest (one row per run).

        Raises:
            ValueError: Unknown axis.
        """
        # Validate axis parameter and prepare study directories.
        if axis not in ('x', 'y'):
            raise ValueError(
                f"run_cookie_detectability_fo_position_simulation(): unknown axis "
                f"{axis!r} (use 'x' or 'y').")
        axis_idx = 0 if axis == 'x' else 1

        # Prepare study directories.
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load base parameters and convert frequency range to tuple if needed.
        base_parms = cls._load_config(base_parms_file)
        if isinstance(base_parms.get('FREQUENCY_RANGE'), list):
            base_parms['FREQUENCY_RANGE'] = tuple(base_parms['FREQUENCY_RANGE'])
        base_parms['PATH_INPUT_OUTPUT_FILES'] = str(study_dir / 'input_output_files')

        # Load overlay parameters for each cookie.
        overlays = [cls._load_config(p) for p in cookie_parms_files]
        overlay_files = {ov['COOKIE_NAME']: str(p) for ov, p in zip(overlays, cookie_parms_files)}

        # Determine the worst-case resource requirements among all cookies as a baseline for preflight checks.
        _worst = None
        for ov in overlays:
            if fo_size_mm > _cookie_total_thickness_mm(ov['SUT']) + 1e-9:
                continue
            p_ = _build_parms_for_cookie_run(base_parms, 
                                            ov, 
                                            fo_size_mm, fo_material, baseline=True,
                                            homogenize_layers=homogenize_layers,fo_clear_mm=fo_size_mm)
            _r = cls.preflight_resource_check(p_, consts_file, study_dir / 'input_output_files')
            if _worst is None or _r['n_rx_points'] > _worst['n_rx_points']:
                _worst = _r

        # Build the job list for each cookie, including baseline and various positions. Check if the FO fits within the cookie thickness.
        jobs = []
        for ov in overlays:

            # Get the total thickness of the cookie. Skip if the FO is too large.
            thickness_mm = _cookie_total_thickness_mm(ov['SUT'])
            if fo_size_mm > thickness_mm + 1e-9:
                print(f"run_cookie_detectability_fo_position_simulation(): {ov['COOKIE_NAME']} is {thickness_mm:g} mm thick - skipping (fo_size_mm={fo_size_mm:g} mm would not fit).")
                continue

            # Determine the possible positions for the FO within the cookie, considering edge clearance.
            fractions = _cookie_position_fractions(
                ov['SUT'], fo_size_mm, n_positions, axis_idx, edge_clearance_mm)
            extent_mm = _cookie_footprint_extent_mm(ov['SUT'], axis_idx)
            
            jobs.append(dict(cookie=ov['COOKIE_NAME'], 
                             overlay=ov,
                             position_index='base', 
                             baseline=True,
                             thickness_mm=thickness_mm, 
                             x_fraction=0.5,
                             offset_mm=0.0))
            for i, frac in enumerate(fractions):
                jobs.append(dict(cookie=ov['COOKIE_NAME'], 
                                 overlay=ov,
                                 position_index=i, 
                                 baseline=False,
                                 thickness_mm=thickness_mm, 
                                 x_fraction=frac,
                                 offset_mm=(frac - 0.5) * extent_mm))

        # Iterate over the jobs to process discrete-inclusion layouts.
        for job in jobs:

            if not job['baseline'] or not any('DISCRETE' in c for c in
                                               job['overlay']['SUT'].get('COMPONENTS', [])):
                continue

            inc = _build_parms_for_cookie_run(base_parms, 
                                              job['overlay'], 
                                              fo_size_mm, 
                                              fo_material,
                                              baseline=True, fo_clear_mm=fo_size_mm)['SUT']['INCLUSIONS']
            info = inc['INFO']

            print(f"fo-position study: {job['cookie']}: {info['n_placed']} of {info['n_target']} {info['NAME']} spheres ({1e3 * inc['DIAMETER']:g} mm) placed = {100 * info['share_in_spheres']:.1f} % of the volume (target {100 * info['share_target']:.0f} %; clearance for a {fo_size_mm:g} mm object at the SUT center only.")
            slug = re.sub(r'[^A-Za-z0-9]+', '_', job['cookie']).strip('_').lower()

            # Save the inclusion centers to a CSV file for this job.
            pd.DataFrame(inc['CENTERS'], 
                columns=['x_m', 'y_m', 'z_above_belt_m']).to_csv(
                out_dir / f'chip_layout_{slug}.csv', index=False)

        # Create or update the manifest file for resumable runs. Set the path and columns.
        manifest_path = out_dir / 'manifest_cookie_fo_position_sweep.csv'
        columns = ['cookie', 
                   'position_index', 
                   'x_fraction', 
                   'offset_mm', 
                   'axis',
                   'baseline', 
                   'thickness_mm', 
                   'fo_size_mm', 
                   'results_folder',
                   'status', 'error']
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path,
                                   dtype={'position_index': str, 'results_folder': str, 'error': str})
        else:
            manifest = pd.DataFrame(columns=columns)

        # Define a helper function to find the index of a job in the manifest.
        def _find(cookie, position_index):
            hit = manifest[(manifest['cookie'] == cookie)
                           & (manifest['position_index'] == str(position_index))]
            
            return hit.index[0] if len(hit) else None

        # Loop over all jobs and process them according to their status in the manifest.
        t_done, n_done = 0.0, 0
        for j, job in enumerate(jobs):

            # Find the index and status of the current job in the manifest.
            idx = _find(job['cookie'], job['position_index'])
            status = manifest.loc[idx, 'status'] if idx is not None else None
            if status == 'OK' or (status == 'FAILED' and not retry_failed):
                continue

            # Print a message indicating the start of processing for the current job.
            label = ('none (baseline)' if job['baseline'] else
                     f"x_fraction={job['x_fraction']:.4f}")
            print(f"\n=== cookie fo-position study run {j + 1}/{len(jobs)}: "
                  f"{job['cookie']}, FO position = {label} ===")
            
            t0 = time.time()
            row = dict(cookie=job['cookie'], 
                       position_index=str(job['position_index']),
                       x_fraction=job['x_fraction'], 
                       offset_mm=job['offset_mm'], 
                       axis=axis,
                       baseline=job['baseline'], 
                       thickness_mm=job['thickness_mm'],
                       fo_size_mm=fo_size_mm)

            # Run the simulation for the current job and handle any exceptions.
            try:

                # Set up the foreign object position for the simulation (relative to the cookie)
                fo_position_fraction = [0.5, 0.5, 0.5]
                fo_position_fraction[axis_idx] = job['x_fraction']

                # Build the parameters for the cookie run, including the foreign object position.
                parms = _build_parms_for_cookie_run(
                    base_parms, job['overlay'], fo_size_mm, fo_material,
                    baseline=job['baseline'], homogenize_layers=homogenize_layers,
                    fo_clear_mm=fo_size_mm, fo_position_fraction=fo_position_fraction)
                parms['FOREIGN_OBJECT']['ALLOW_INCLUSION_OVERLAP'] = True

                # Initialize the simulation object with the prepared parameters.
                em = cls(parms=parms, consts=consts_file)
                em.protocol_context = dict(
                            study='cookie detectability study, FO position', base_parms_file=base_parms_file,
                            cookie_parms_file=overlay_files[job['cookie']], consts_file=consts_file,
                            cookie=job['cookie'], 
                            position_index=job['position_index'], 
                            x_fraction=job['x_fraction'],
                            offset_mm=round(job['offset_mm'], 3), 
                            axis=axis, 
                            fo_size_mm=fo_size_mm, 
                            fo_material=fo_material,
                            baseline=job['baseline'])
                
                # Run the simulation for the current job.
                em.simulate()

                # Update the manifest row with the results of the simulation.
                row.update(results_folder=Path(os.path.relpath(em.path_folder, study_dir)).as_posix(),status='OK', error='')
            
            except Exception as e:

                print(f"cookie fo-position study run FAILED - {e!r}")
                row.update(results_folder='', status='FAILED', error=repr(e))

            finally:

                plt.close('all')

            # Append the current row to the manifest, either as a new row or updating an existing one.
            if idx is None:
                manifest = pd.concat([manifest, 
                                      pd.DataFrame([row])], 
                                      ignore_index=True)
            else:
                for k, v in row.items():
                    manifest.loc[idx, k] = v

            # Save the updated manifest to the CSV file.
            manifest.to_csv(manifest_path, index=False)

            # If the current run failed and it's the first run, optionally stop the batch.
            if row['status'] == 'FAILED' and n_done == 0 and stop_if_first_run_fails:
                raise RuntimeError(f"run_cookie_detectability_fo_position_simulation(): the first run ({job['cookie']}, position = {label}) failed: {row['error']} - stopped before the rest of the batch (see the manifest in {out_dir}; pass stop_if_first_run_fails=False to continue anyway).")
            n_done += 1
            t_done += time.time() - t0
            left = sum(1 for jb in jobs[j + 1:]
                       if (lambda i: i is None or manifest.loc[i, 'status'] != 'OK')(_find(jb['cookie'], jb['position_index'])))
            
            print(f"cookie fo-position study: {n_done} simulated this session, avg {_format_duration(t_done / n_done)}/run, {left} left - ETA {_format_duration(left * t_done / n_done)}.")

        # Perform completeness check. Identify runs that are marked as 'OK' in the manifest and missing required result files.
        ok = manifest[manifest['status'] == 'OK']
        need_files = _cookie_required_result_files(base_parms)
        missing = [r for r in ok['results_folder']
                   if not all((_resolve_cookie_run_folder(study_dir, r) / fn).is_file() for fn in need_files)]
        
        # Count the number of failed and open runs and report the status.
        n_failed = int((manifest['status'] == 'FAILED').sum())
        n_open = len(jobs) - len(ok)
        print(f"\nrun_cookie_detectability_fo_position_simulation(): {len(ok) - len(missing)} of {len(jobs)} runs have a data.pkl; {n_failed} FAILED, {n_open} not OK yet{f', {len(missing)} OK rows lack data.pkl: {missing}' if missing else ''}.")
        if not n_open and not missing:
            print("run_cookie_detectability_fo_position_simulation(): all data collected - run MODE == 9 (show_cookie_detectability_fo_position_results()) next.")
        
        return manifest

    @classmethod
    def show_cookie_detectability_fo_position_results(cls, 
            base_parms_file: str,
            output_dir: str = 'studies/cookie detectability study/fo position/transmission',
            response_metrics: tuple = ('rms_dphase_max', 'rms_dmag_max'),
            detection_rule: str = 'envelope',
            snr_db_sweep: tuple = (20, 30, 40),
            rx_sides: tuple = None) -> 'pd.DataFrame':
        """This function analyzes the results of the cookie detectability study for the field offset position, scoring each run against its baseline and generating summary tables and figures (MODE 9, results of MODE 8 required).

        Args:
            base_parms_file (str): 
                PARMS_FILE of the runs; only its AMP_NOISE_FLOOR_DB / PHASE_NOISE_FLOOR_DEG are used.
            output_dir (str): 
                Study folder (reads manifest_cookie_fo_position_sweep.csv from results_analysis/, writes the outputs there).
            response_metrics (tuple): 
                Columns of _compute_differential_metrics() to plot/threshold, one panel each (default as in the size study).
            detection_rule (str): 
                'envelope' (default) or 'conservative'; see _max_detectable_offset_mm() (whether the center itself must clear the floor).
            snr_db_sweep (tuple): 
                Assumed per-bin SNRs [dB] for the sensitivity table. () skips it.
            rx_sides (tuple): 
                None = all sides found. Or e.g. ('reflection',) / ('transmission',).

        Returns:
            pandas.DataFrame: Per-side, per-product, per-position results (empty if nothing could be scored) with column rx_side.

        Raises:
            FileNotFoundError: If the manifest does not exist (run MODE == 8 first).
            ValueError: If a response metric is not produced by _compute_differential_metrics().
        """
        response_metrics = list(response_metrics)

        # Validate detection rule
        if detection_rule not in ('conservative', 'envelope'):
            raise ValueError(f"show_cookie_detectability_fo_position_results(): unknown detection_rule {detection_rule!r} (use 'conservative' or 'envelope').")

        # Prepare study directories and manifest
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        manifest_path = out_dir / 'manifest_cookie_fo_position_sweep.csv'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"show_cookie_detectability_fo_position_results(): no manifest at {manifest_path} - run MODE 8 (run_cookie_detectability_fo_position_simulation()) first.")
        
        # Load the manifest CSV file containing the run information.
        manifest = pd.read_csv(manifest_path, 
                               dtype={'position_index': str, 
                                      'results_folder': str, 
                                      'error': str})

        # Load the base parameters from the configuration file and compute the floors for each response metric.
        base_parms = cls._load_config(base_parms_file)
        floors = {m: _floor_for_metric(m, base_parms) for m in response_metrics}
        amp_floor = base_parms.get('AMP_NOISE_FLOOR_DB')
        phase_floor = base_parms.get('PHASE_NOISE_FLOOR_DEG')

        # Get unique cookie names from the manifest, preserving order.
        cookie_names = list(dict.fromkeys(manifest['cookie']))

        # Identify the baseline runs that are marked as 'OK' for each cookie.
        ok_base = manifest[(manifest['position_index'] == 'base') & (manifest['status'] == 'OK')]
        if ok_base.empty:
            print("show_cookie_detectability_fo_position_results(): no OK baseline run - nothing to score.")
            return pd.DataFrame()

        # Determine the receiver sides to be scored based on the first OK baseline run.
        sides = _cookie_select_rx_sides(study_dir, 
                                        ok_base['results_folder'].iloc[0], rx_sides,
                                        'show_cookie_detectability_fo_position_results()')
        if not sides:
            print("show_cookie_detectability_fo_position_results(): no result files found - nothing to score.")
            return pd.DataFrame()
        
        print(f"show_cookie_detectability_fo_position_results(): scoring receiver side(s): {sides}")

        # Define a helper function to load run data for a given folder and receiver side.
        def _load(folder, sd):
            return cls._load_run_data(_resolve_cookie_run_folder(study_dir, folder), verbose=False, side=None if sd == 'legacy' else sd)

        # Iterate over each receiver side and then over each cookie to compute differential metrics.
        records = []
        for sd in sides:
            for name in cookie_names:

                # Identify the baseline run for the current cookie and receiver side. Skip if none found.
                b = manifest[(manifest['cookie'] == name) & (manifest['position_index'] == 'base') & (manifest['status'] == 'OK')]
                if b.empty:
                    print(f"show_cookie_detectability_fo_position_results(): no OK baseline "
                          f"for {name} - skipped.")
                    continue

                # Load the baseline run data for the current cookie and receiver side.
                H_b, f_b, *_ = _load(b['results_folder'].iloc[0], sd)
                runs = manifest[(manifest['cookie'] == name) & (manifest['position_index'] != 'base')
                                & (manifest['status'] == 'OK')]

                # Iterate over each non-baseline run for the current cookie and receiver side.
                for _, r in runs.iterrows():
                    
                    # Load the current run data for the current receiver side and cookie. Compute differential metrics relative to the baseline.
                    H, f, *_ = _load(r['results_folder'], sd)
                    metrics = cls._compute_differential_metrics(
                        H, f, H_b, f_b, 
                        amp_noise_floor_db=amp_floor,
                        phase_noise_floor_deg=phase_floor)
                    
                    # Check that all requested response metrics are available in the computed metrics.
                    for m in response_metrics:
                        if m not in metrics:
                            raise ValueError(
                                f"show_cookie_detectability_fo_position_results(): unknown response metric {m!r}; available: {sorted(metrics)}")
                    records.append(dict(rx_side=sd, 
                                        cookie=name, 
                                        position_index=r['position_index'],
                                        x_fraction=float(r['x_fraction']),
                                        offset_mm=float(r['offset_mm']), 
                                        axis=r['axis'],
                                        thickness_mm=r['thickness_mm'],
                                        fo_size_mm=float(r['fo_size_mm']), **metrics))
        
        # If no records were collected, print a message and return an empty DataFrame.
        if not records:
            print("show_cookie_detectability_fo_position_results(): nothing to score yet.")
            return pd.DataFrame()

        # Convert the collected records into a DataFrame and sort them for further analysis.
        results_all = pd.DataFrame(records)
        results_all['rx_side'] = pd.Categorical(results_all['rx_side'], 
                                                categories=sides, 
                                                ordered=True)
        results_all = results_all.sort_values(['rx_side', 
                                               'cookie', 
                                               'offset_mm']).reset_index(drop=True)
        results_all['rx_side'] = results_all['rx_side'].astype(str)
        results_all.to_csv(out_dir / 'cookie_detectability_fo_position_results.csv', index=False)

        # Iterate over each receiver side to compute metrics and generate plots
        range_parts, ratio_parts, sens_rows = [], [], []
        for sd in sides:

            # Filter results for the current receiver side
            suffix = '' if sd == 'legacy' else f'_{sd}'
            results = results_all[results_all['rx_side'] == sd].drop(columns='rx_side').reset_index(drop=True)
            if results.empty:
                continue

            # Compute the range of detectable offsets for the current receiver side
            range_df = _cookie_score_position_range(results, 
                                                    response_metrics, 
                                                    floors, 
                                                    detection_rule)
            range_parts.append(range_df.assign(rx_side=sd))

            # Generate overview and journal plots for the current receiver side
            _plot_cookie_detectability_position(results, 
                                                range_df, 
                                                response_metrics, 
                                                floors, out_dir / f'cookie_detectability_fo_position_overview{suffix}.png', detection_rule=detection_rule)
            
            # Generate a plot for cookie detectability for the current receiver side
            _plot_cookie_detectability_position_journal(results, 
                                                        range_df, response_metrics, floors, out_dir / f'cookie_detectability_fo_position{suffix}.png')

            # Compute the ratio of the response to the floor for each metric and the combined worst-case ratio
            ratio_df = results[['cookie', 'offset_mm', 'x_fraction']].copy()
            worst = np.zeros(len(results))
            for m in response_metrics:
                if floors[m] is None:
                    continue
                r = results[m].to_numpy() / floors[m]
                ratio_df[f'{_cookie_metric_short_name(m)}_over_floor'] = np.round(r, 3)
                worst = np.maximum(worst, r)
            ratio_df['combined_over_floor'] = np.round(worst, 3)
            
            # Include additional columns for the number of exceedances if available
            for c in ('n_exceed_max', 'n_exceed_mean'):
                if c in results:
                    ratio_df[c] = results[c].to_numpy()
            ratio_parts.append(ratio_df.assign(rx_side=sd))

            # Sweep over different assumed receiver SNR values to assess sensitivity of the limit
            for snr in snr_db_sweep:
                fl = {m: _crb_floor_for_metric(m, snr) for m in response_metrics}
                d = _cookie_score_position_range(results, response_metrics, fl, detection_rule)
                for _, r in d.iterrows():
                    row = dict(rx_side=sd, snr_db=snr, cookie=r['cookie'])
                    for m in response_metrics:
                        row[f'floor_{_cookie_metric_short_name(m)}'] = (
                            None if fl[m] is None else round(fl[m], 4))
                    for k in list(response_metrics) + ['combined']:
                        row[f'max_offset_mm_{k}'] = r[f'max_offset_mm_{k}']
                        row[f'status_{k}'] = r[f'status_{k}']
                    sens_rows.append(row)

        # Combine and export the results for all receiver sides
        range_all = pd.concat(range_parts, ignore_index=True)
        range_all = range_all[['rx_side'] + 
                              [c for c in range_all.columns if c != 'rx_side']]
        range_all.to_csv(out_dir / 'cookie_detectability_fo_position_range_summary.csv', index=False)
        
        # Combine and export the ratio results for all receiver sides
        ratio_all = pd.concat(ratio_parts, ignore_index=True)
        ratio_all = ratio_all[['rx_side'] + [c for c in ratio_all.columns if c != 'rx_side']]
        ratio_all.to_csv(out_dir / 'cookie_detectability_fo_position_ratio_by_offset.csv', index=False)
        
        # Combine and export the sensitivity results for all receiver sides
        if sens_rows:
            pd.DataFrame(sens_rows).to_csv(out_dir / 'cookie_detectability_fo_position_snr_sensitivity.csv', index=False)
        if len([sd for sd in sides if sd != 'legacy']) > 1:
            _plot_cookie_mds_side_comparison(range_all, 
                                             response_metrics, out_dir / 'cookie_detectability_fo_position_range_reflection_vs_transmission.png',
                                             prefix='max_offset_mm_', ylabel='Largest detectable offset / mm')

        # Prepare results for export and further analysis
        results = results_all
        range_df = range_all
        _export_cookie_position_paper_tables(results_all, 
                                             range_all, 
                                             pd.DataFrame(sens_rows), 
                                             sides, 
                                             response_metrics, 
                                             floors,
                                             snr_db_sweep, 
                                             out_dir)

        # Check if the product ranking is consistent across different response metrics for each receiver side
        for sd in sides:

            rankings = {}
            sub = range_df[range_df['rx_side'] == sd]

            # Compute the product ranking for each response metric based on the largest detectable offset
            for m in response_metrics + ['combined']:
                col = sub.set_index('cookie')[f'max_offset_mm_{m}'].dropna().sort_values()
                rankings[m] = list(col.index)

            if len({tuple(r) for r in rankings.values()}) > 1:
                print(f"show_cookie_detectability_fo_position_results(): NOTE - [{sd}] product ranking by largest detectable offset differs between metrics:")
                for m, r in rankings.items():
                    print(f"    {m}: {' < '.join(r)}")

        print(f"\nshow_cookie_detectability_fo_position_results(): done "
              f"({detection_rule} rule) - see {out_dir}")
        print(range_df.to_string(index=False))

        return results

    @classmethod
    def run_cookie_detectability_frequency_simulation(cls, 
            base_parms_file: str, 
            consts_file: str, 
            cookie_parms_files: list,
            frequency_bands_ghz: tuple = ((22, 26), (58, 62), (77, 81), (92, 96), (120, 124)),
            fo_size_mm: float = 5.0, 
            fo_material: str = 'PVC',
            output_dir: str = 'studies/cookie detectability study/frequency/transmission',
            reuse_reference_study_dir: str = None, 
            retry_failed: bool = True, 
            homogenize_layers: bool = False, 
            stop_if_first_run_fails: bool = True, 
            wideband: bool = False, 
            pml_wavelengths: float = None, 
            parms_overrides: dict = None) -> 'pd.DataFrame':
        """This function runs the frequency sweep simulation for cookie detectability studies (MODE 10). The centered FO of fixed size is placed in the array and the frequency bands are swept.

        Args:
            base_parms_file (str): 
                Base parameter file
            consts_file (str): 
                Constants file
            cookie_parms_files (list): 
                Paths of parameters/cookie_types/*.json.
            frequency_bands_ghz (sequence): 
                (f_low, f_high) pairs in GHz, one band each (default 22-26, 58-62, 77-81, 92-96, 120-124). Simulated from the highest band down.
            fo_size_mm (float): 
                Cube edge in mm, centered, fixed (default 5).
            fo_material (str): 
                Foreign-object material key (default 'PVC').
            output_dir (str): 
                Study folder (created if missing).
            reuse_reference_study_dir (str): 
                FO-position study folder to take the reference band's runs from; None simulates every band (default).
            retry_failed (bool): 
                Retry FAILED manifest rows.
            homogenize_layers (bool): 
                Flatten layered products (comparison only; default False).
            stop_if_first_run_fails (bool): 
                Raise a RuntimeError if the first run of this call fails (default True).
            wideband (bool): 
                One run over all bands instead of one run per band (default False).
            pml_wavelengths (float): 
                Wideband only: PML at least this many free-space wavelengths at the lowest band edge thick (N_PML raised, inner domain unchanged); None keeps N_PML (default).
            parms_overrides (dict): 
                Parameter values set in every run of this study, e.g. RX_HUYGENS_(WALL_)SAMPLES_PER_WAVELENGTH. Disables the reuse of the FO-position runs (default None).

        Returns:
            pandas.DataFrame: The manifest (one row per run).

        Raises:
            ValueError: No band given, or a band without 0 < f_low < f_high.
        """
        # Dependencies
        import contextlib
        import io

        # Sort and validate the frequency bands.
        bands = sorted((float(lo), float(hi)) for lo, hi in frequency_bands_ghz)
        if not bands:
            raise ValueError("run_cookie_detectability_frequency_simulation(): "
                             "frequency_bands_ghz is empty.")
        
        for lo, hi in bands:
            if not 0 < lo < hi:
                raise ValueError(
                    f"run_cookie_detectability_frequency_simulation(): band ({lo}, {hi}) "
                    f"GHz is invalid (need 0 < f_low < f_high).")

        # Determine the simulation bands: either wideband (one run over all bands) or per-band.
        sim_bands = [(bands[0][0], max(hi for _, hi in bands))] if wideband else bands

        # Create the output directories.
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load the base parameters and overlay configurations.
        base_parms = cls._load_config(base_parms_file)
        if isinstance(base_parms.get('FREQUENCY_RANGE'), list):
            base_parms['FREQUENCY_RANGE'] = tuple(base_parms['FREQUENCY_RANGE'])
        base_parms['PATH_INPUT_OUTPUT_FILES'] = str(study_dir / 'input_output_files')

        # Determine the reference band from the base parameters.
        ref_band = tuple(f / 1e9 for f in base_parms['FREQUENCY_RANGE'])

        # Load the overlay configurations for the cookie parameters.
        overlays = [cls._load_config(p) for p in cookie_parms_files]

        # Define helper function for labeling and reference band checking as well as  if a band is the reference band.
        def _label(lo, hi):
            return f"{lo:g}-{hi:g}"

        def _is_ref(lo, hi):
            return abs(lo - ref_band[0]) < 1e-6 and abs(hi - ref_band[1]) < 1e-6

        # Define helper function for determining the mesh step for a given band.
        def _dx_for_band(lo, hi):
            p = copy.deepcopy(base_parms)
            p['FREQUENCY_RANGE'] = (lo * 1e9, hi * 1e9)
            with contextlib.redirect_stdout(io.StringIO()):
                em = cls(parms=p, consts=consts_file)
                em._determine_number_of_elements()
            return em.dx

        # Determine the mesh step for the reference band and all simulation bands.
        dx_ref = _dx_for_band(*ref_band)
        dx_band = {b: (dx_ref if _is_ref(*b) else _dx_for_band(*b)) for b in sim_bands}

        # Determine the number of PML cells for the reference band and all simulation bands.
        n_layer = int(base_parms['N_PML']) + int(base_parms['N_PML_MARGIN'])
        n_pml_band = {b: int(base_parms['N_PML']) for b in sim_bands}

        # Adjust the number of PML cells for wideband simulations if necessary.
        if wideband and pml_wavelengths:
            for b in sim_bands:
                n_pml_band[b] = max(n_pml_band[b], 
                                    int(np.ceil(pml_wavelengths * sp.constants.c / (b[0] * 1e9) / dx_band[b])))
                print(f"run_cookie_detectability_frequency_simulation(): wideband PML {n_pml_band[b]} cells ({1e3 * n_pml_band[b] * dx_band[b]:.1f} mm = {pml_wavelengths:g} lambda0 at {b[0]:g} GHz; N_PML was {base_parms['N_PML']}).")

        # Define helper function for computing the grid (domain) lengths for a given band.
        def _grid_lengths(overlay, lo, hi):
            out = {}
            for key in ('L_X_GRID', 'L_Y_GRID', 'L_Z_GRID'):
                l_ref = overlay.get('GRID', {}).get(key, base_parms[key])
                if _is_ref(lo, hi):
                    out[key] = l_ref
                    continue
                dx = dx_band[(lo, hi)]
                inner = l_ref - 2 * n_layer * dx_ref
                n_cells = int(np.ceil((inner + 2 * (n_pml_band[(lo, hi)] + 
                                                    int(base_parms['N_PML_MARGIN'])) * dx) / dx - 1e-9))
                out[key] = (n_cells + 0.02) * dx
            return out

        # Reference-band runs of the FO-position study are optionally reused if available.
        ref_manifest, ref_dir = None, None
        if reuse_reference_study_dir is not None and parms_overrides:

            print("run_cookie_detectability_frequency_simulation(): parms_overrides given - the FO-position runs are not reused.")

        elif reuse_reference_study_dir is not None:

            ref_dir = Path(reuse_reference_study_dir)
            ref_path = ref_dir / 'results_analysis' / 'manifest_cookie_fo_position_sweep.csv'
            if ref_path.is_file():
                ref_manifest = pd.read_csv(ref_path, 
                                 dtype={'position_index': str,
                                        'results_folder': str,
                                        'error': str})
            else:
                print(f"run_cookie_detectability_frequency_simulation(): no manifest at {ref_path} - the reference band will be simulated.")

        # Define helper function to get reference run folder for a given cookie and kind
        def _reference_run(cookie, kind):
            if ref_manifest is None:
                return None
            pos = 'base' if kind == 'base' else '0'
            hit = ref_manifest[(ref_manifest['cookie'] == cookie)
                               & (ref_manifest['position_index'] == pos)
                               & (ref_manifest['status'] == 'OK')]
            if hit.empty or abs(float(hit['fo_size_mm'].iloc[0]) - fo_size_mm) > 1e-9:
                return None
            
            folder = _resolve_cookie_run_folder(ref_dir, hit['results_folder'].iloc[0])

            if not all((folder / fn).is_file() for fn in _cookie_required_result_files(base_parms)):
                return None
            try:
                return Path(os.path.relpath(folder, study_dir)).as_posix()
            except ValueError:
                return Path(folder).as_posix()

        # Iterate over simulation bands and overlays to determine which jobs need to be run (band by band)
        jobs, skipped = [], set()
        for (lo, hi) in sim_bands[::-1]:
            for ov in overlays:

                # Compute total thickness of the overlay's food sample (SUT)
                thickness_mm = _cookie_total_thickness_mm(ov['SUT'])

                # Skip overlays that are too thin
                if fo_size_mm > thickness_mm + 1e-9:

                    if ov['COOKIE_NAME'] not in skipped:
                        print(f"run_cookie_detectability_frequency_simulation(): {ov['COOKIE_NAME']} is {thickness_mm:g} mm thick - skipping (fo_size_mm={fo_size_mm:g} mm would not fit).")
                        skipped.add(ov['COOKIE_NAME'])
                    continue

                for kind in ('base', 'fo'):

                    # Determine if we can reuse a previous simulation for this overlay and band
                    reuse = _reference_run(ov['COOKIE_NAME'], kind) if _is_ref(lo, hi) else None
                    jobs.append(dict(cookie=ov['COOKIE_NAME'], 
                                     overlay=ov, 
                                     band=(lo, hi),
                                     kind=kind, 
                                     thickness_mm=thickness_mm, 
                                     reuse=reuse,
                                     out_bands=(bands if wideband else [(lo, hi)])))

        print("run_cookie_detectability_frequency_simulation(): bands and meshes (same physical scene in every band; cost per run ~ nx*ny*nz/dx):")

        # Print grid and mesh information for each simulation band
        for (lo, hi) in sim_bands:
            g = _grid_lengths(overlays[0], lo, hi)
            print(f"    {_label(lo, hi):>8s} GHz: dx = {1e3 * dx_band[(lo, hi)]:.4f} mm, domain {1e3 * g['L_X_GRID']:.2f} x {1e3 * g['L_Y_GRID']:.2f} x {1e3 * g['L_Z_GRID']:.2f} mm ({g['L_X_GRID'] / dx_band[(lo, hi)]:.0f} x {g['L_Y_GRID'] / dx_band[(lo, hi)]:.0f} x {g['L_Z_GRID'] / dx_band[(lo, hi)]:.0f} cells){' [reference band]' if _is_ref(lo, hi) else ''}")

        # Check resource availability for each pending band before starting simulations (receiver buffers grow with frequency, finer cells)
        pending_bands = sorted({j['band'] for j in jobs if j['reuse'] is None})
        problems = []
        for (lo, hi) in pending_bands:
            need = None
            for ov in overlays:
                
                if ov['COOKIE_NAME'] in skipped:
                    continue
                
                p_ = _build_parms_for_cookie_run(base_parms, 
                                                 ov, 
                                                 fo_size_mm, 
                                                 fo_material, 
                                                 baseline=True,
                                                 homogenize_layers=homogenize_layers, 
                                                 fo_clear_mm=fo_size_mm)
                p_['FREQUENCY_RANGE'] = (lo * 1e9, hi * 1e9)
                p_.update(parms_overrides or {})
                if wideband:
                    p_['N_PML'] = n_pml_band[(lo, hi)]
                p_.update(_grid_lengths(ov, lo, hi))

                # Check if resources are available for this band and overlay before proceeding
                try:
                    cls.preflight_resource_check(p_, consts_file, 
                                                 study_dir / 'input_output_files')
                except RuntimeError as e:
                    problems.append(f"{_label(lo, hi)} GHz ({ov['COOKIE_NAME']}): {str(e).split('): ', 1)[-1]}")
                    break

        # If there were any problems with resource availability, raise an error before proceeding
        if problems:
            raise RuntimeError("run_cookie_detectability_frequency_simulation(): not enough resources for one run in " + str(len(problems))
            + " band(s) - nothing was started. " + " | ".join(problems)
            + " Remove the band(s) from COOKIE_FREQUENCY_BANDS_GHZ or lower RX_HUYGENS(_WALL)_SAMPLES_PER_WAVELENGTH.")

        # Process discrete-inclusion layouts
        for ov in overlays:

            if ov['COOKIE_NAME'] in skipped or not any(
                    'DISCRETE' in c for c in ov['SUT'].get('COMPONENTS', [])):
                continue

            inc = _build_parms_for_cookie_run(base_parms, 
                                              ov, 
                                              fo_size_mm, 
                                              fo_material, 
                                              baseline=True,fo_clear_mm=fo_size_mm)['SUT']['INCLUSIONS']
            
            info = inc['INFO']
            print(f"frequency study: {ov['COOKIE_NAME']}: {info['n_placed']} of {info['n_target']} {info['NAME']} spheres ({1e3 * inc['DIAMETER']:g} mm) placed = {100 * info['share_in_spheres']:.1f} % of the volume; the layout is the same in every band.")

            # Save the discrete-inclusion layout to a CSV file
            slug = re.sub(r'[^A-Za-z0-9]+', '_', ov['COOKIE_NAME']).strip('_').lower()
            pd.DataFrame(inc['CENTERS'], 
                         columns=['x_m', 'y_m', 'z_above_belt_m']).to_csv(out_dir / f'chip_layout_{slug}.csv', index=False)

        # Load or initialize the manifest for resumable runs and define the columns for the manifest CSV file
        manifest_path = out_dir / 'manifest_cookie_frequency_sweep.csv'
        columns = ['cookie', 
                   'band', 
                   'f_low_ghz', 
                   'f_high_ghz', 
                   'f_center_ghz', 
                   'run_kind',
                   'thickness_mm', 
                   'fo_size_mm', 
                   'dx_mm', 
                   'results_folder', 
                   'status', 
                   'error']
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path, 
                                   dtype={'band': str, 
                                          'results_folder': str,
                                          'error': str})
        else:
            manifest = pd.DataFrame(columns=columns)

        # Define helper functions to find and check the status of jobs in the manifest
        def _find(job):
            hit = manifest[(manifest['cookie'] == job['cookie'])
                           & (manifest['band'] == _label(*job['out_bands'][0]))
                           & (manifest['run_kind'] == job['kind'])]
            return hit.index[0] if len(hit) else None

        # Define helper function to check if a job is open (not completed successfully)
        def _is_open(job):
            i = _find(job)
            return i is None or manifest.loc[i, 'status'] != 'OK'

        # Loop over all jobs and process them according to their status in the manifest
        t_done, n_done = 0.0, 0
        for j, job in enumerate(jobs):

            # Skip jobs that are already completed successfully or should not be retried if failed
            idx = _find(job)
            status = manifest.loc[idx, 'status'] if idx is not None else None
            if status == 'OK' or (status == 'FAILED' and not retry_failed):
                continue

            # Extract the lower and upper frequency bounds for the current job
            lo, hi = job['band']
            what = ('no foreign object (baseline)' 
                    if job['kind'] == 'base' 
                    else f"{fo_size_mm:g} mm {fo_material} cube, centered")
            print(f"\n=== cookie frequency study run {j + 1}/{len(jobs)}: {job['cookie']}, {_label(lo, hi)} GHz, {what} ===")
            
            # Prepare the row dictionary with job and frequency band information
            t0 = time.time()
            row = dict(cookie=job['cookie'], 
                       band=_label(lo, hi), 
                       f_low_ghz=lo, 
                       f_high_ghz=hi,
                       f_center_ghz=0.5 * (lo + hi), 
                       run_kind=job['kind'],
                       thickness_mm=job['thickness_mm'], 
                       fo_size_mm=fo_size_mm,
                       dx_mm=1e3 * dx_band[(lo, hi)])

            # Attempt the simulation and handle any exceptions to keep the batch alive
            simulated = False
            try:

                if job['reuse'] is not None:
                    
                    print(f"    reusing the fo-position study's run {job['reuse']}")
                    row.update(results_folder=job['reuse'], status='OK',
                               error='reused from the fo-position study')
                    
                else:

                    # Build the simulation parameters for the current cookie run
                    parms = _build_parms_for_cookie_run(
                        base_parms, 
                        job['overlay'], 
                        fo_size_mm, 
                        fo_material,
                        baseline=(job['kind'] == 'base'), homogenize_layers=homogenize_layers,
                        fo_clear_mm=fo_size_mm)

                    # Set the frequency range for the simulation
                    parms['FREQUENCY_RANGE'] = (lo * 1e9, hi * 1e9)
                    parms.update(parms_overrides or {})
                    if wideband:
                        parms['N_PML'] = n_pml_band[(lo, hi)]
                        parms['FREQUENCY_BANDS'] = [[b_lo * 1e9, b_hi * 1e9] for b_lo, b_hi in job['out_bands']]
                    parms.update(_grid_lengths(job['overlay'], lo, hi))
                    
                    # Initialize the EM simulation with the built parameters
                    em = cls(parms=parms, consts=consts_file)
                    em.simulate()
                    simulated = True
                    row.update(results_folder=Path(os.path.relpath(
                        em.path_folder, study_dir)).as_posix(), status='OK', error='')
                    
            except Exception as e:
                
                print(f"cookie frequency study run FAILED - {e!r}")
                row.update(results_folder='', status='FAILED', error=repr(e))
            
            finally:

                plt.close('all')

            # Update the manifest for each output band of the current job
            for b_lo, b_hi in job['out_bands']:
                row_b = dict(row, 
                             band=_label(b_lo, b_hi), 
                             f_low_ghz=b_lo, 
                             f_high_ghz=b_hi, 
                             f_center_ghz=0.5 * (b_lo + b_hi))
                idx_b = _find(dict(job, out_bands=[(b_lo, b_hi)]))
                if idx_b is None:
                    manifest = pd.concat([manifest, pd.DataFrame([row_b])], ignore_index=True)
                else:
                    for k, v in row_b.items():
                        manifest.loc[idx_b, k] = v
            manifest.to_csv(manifest_path, index=False)

            # Stop if the first run failed and the flag is set
            if row['status'] == 'FAILED' and n_done == 0 and stop_if_first_run_fails:
                raise RuntimeError(
                    f"run_cookie_detectability_frequency_simulation(): the first run ({job['cookie']}, {_label(lo, hi)} GHz, {job['kind']}) failed: {row['error']} - stopped before the rest of the batch (see the manifest in {out_dir}; pass stop_if_first_run_fails=False to continue anyway).")
            
            # Update counters and print progress if the current job was simulated
            if simulated:
                n_done += 1
                t_done += time.time() - t0
                left = sum(1 for jb in jobs[j + 1:] if jb['reuse'] is None and _is_open(jb))
                print(f"cookie frequency study: {n_done} simulated this session, avg {_format_duration(t_done / n_done)}/run, {left} left - ETA {_format_duration(left * t_done / n_done)}.")

        # Identify OK runs and check for missing result files
        ok = manifest[manifest['status'] == 'OK']
        need_files = _cookie_required_result_files(base_parms)
        missing = [r for r in ok['results_folder']
                   if not all((_resolve_cookie_run_folder(study_dir, r) / fn).is_file() for fn in need_files)]
        
        # Count failed and open jobs for summary
        n_failed = int((manifest['status'] == 'FAILED').sum())
        n_open = sum(1 for jb in jobs if _is_open(jb))
        print(f"\nrun_cookie_detectability_frequency_simulation(): {len(jobs) - n_open - len(set(missing))} of {len(jobs)} runs have a data.pkl; {n_failed} FAILED, {n_open} not OK yet{f', {len(missing)} OK rows lack data.pkl: {missing}' if missing else ''}.")
        if not n_open and not missing:
            print("run_cookie_detectability_frequency_simulation(): all data collected - run MODE == 11 (show_cookie_detectability_frequency_results()) next.")
        
        return manifest

    @classmethod
    def show_cookie_detectability_frequency_results(cls, 
            base_parms_file: str,
            output_dir: str = 'studies/cookie detectability study/frequency/transmission',
            response_metrics: tuple = ('rms_dphase_max', 'rms_dmag_max'),
            snr_db_sweep: tuple = (20, 30, 40),
            rx_sides: tuple = None) -> 'pd.DataFrame':
        """This function analyzes the results of the cookie detectability frequency simulation runs and generates summary tables and plots (MODE 11, requires results from MODE 10).

        Args:
            base_parms_file (str): 
                PARMS_FILE of the runs; only its AMP_NOISE_FLOOR_DB / PHASE_NOISE_FLOOR_DEG are used.
            output_dir (str): 
                Study folder (reads manifest_cookie_frequency_sweep.csv from results_analysis/, writes the outputs there).
            response_metrics (tuple):
                Columns of _compute_differential_metrics() to plot/threshold, one panel each.
            snr_db_sweep (tuple): 
                Assumed per-bin SNRs in dB for the sensitivity table; () skips it.
            rx_sides (tuple): 
                None = all sides found; or e.g. ('reflection',) / ('transmission',).

        Returns:
            pandas.DataFrame: Per-side, per-product, per-band results (empty if nothing could be scored); column rx_side.

        Raises:
            FileNotFoundError: If the manifest does not exist (run MODE == 10 first).
            ValueError: If a response metric is not produced by _compute_differential_metrics().
        """
        response_metrics = list(response_metrics)

        # Prepare study and output directories, and read the manifest.
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'
        
        # Path to the manifest file for the cookie frequency sweep results.
        manifest_path = out_dir / 'manifest_cookie_frequency_sweep.csv'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"show_cookie_detectability_frequency_results(): no manifest at {manifest_path} - run MODE == 10 (run_cookie_detectability_frequency_simulation()) first.")
        manifest = pd.read_csv(manifest_path, 
                               dtype={'band': str, 
                                      'results_folder': str,
                                      'error': str})

        # Load the base parameters and compute noise floors for the response metrics.
        base_parms = cls._load_config(base_parms_file)
        floors = {m: _floor_for_metric(m, base_parms) for m in response_metrics}
        amp_floor = base_parms.get('AMP_NOISE_FLOOR_DB')
        phase_floor = base_parms.get('PHASE_NOISE_FLOOR_DEG')

        # Get the unique cookie names
        cookie_names = sorted(dict.fromkeys(manifest['cookie']), key=_cookie_short_name)
        c0 = sp.constants.c

        # Define a helper function to pick the appropriate row from the manifest based on cookie, band, and run kind.
        def _pick(cookie, band, kind):
            hit = manifest[(manifest['cookie'] == cookie) & (manifest['band'] == band) & (manifest['run_kind'] == kind) & (manifest['status'] == 'OK')]
            return None if hit.empty else hit.iloc[0]

        # Filter the manifest to only include rows with OK status for the base run kind.
        ok_base = manifest[(manifest['run_kind'] == 'base') & 
                           (manifest['status'] == 'OK')]
        if ok_base.empty:
            print("show_cookie_detectability_frequency_results(): no OK baseline run - nothing to score.")
            return pd.DataFrame()
        
        found = [p for p in ok_base['results_folder'] if _cookie_sides_in_run_folder(_resolve_cookie_run_folder(study_dir, p))]
        sides = _cookie_select_rx_sides(study_dir, found[0] if found else ok_base['results_folder'].iloc[0], rx_sides,
                'show_cookie_detectability_frequency_results()')
        if not sides:
            print("show_cookie_detectability_frequency_results(): no result files found, therefore nothing to score.")
            return pd.DataFrame()
        print(f"show_cookie_detectability_frequency_results(): scoring receiver side(s): {sides}")

        # Define a helper function to load run data for a given row and side.
        def _load(row, sd):
            return cls._load_run_data(_resolve_cookie_run_folder(study_dir, 
                                    row['results_folder']), 
                                    verbose=False,
                                    side=None if sd == 'legacy' else sd)

        # Iterate over each receiver side and then cookie to compute metrics.
        records = []
        for sd in sides:
            for name in cookie_names:

                # Select the frequency bands for the current cookie.
                bands = manifest[manifest['cookie'] == name].drop_duplicates('band')

                # Compute metrics for each frequency band of the current cookie.
                for _, b in bands.iterrows():

                    # Pick the baseline and foreign-object runs for the current cookie and band.
                    r_base, r_fo = _pick(name, b['band'], 'base'), _pick(name, b['band'], 'fo')

                    if r_base is None or r_fo is None:
                        print(f"show_cookie_detectability_frequency_results(): {name}, {b['band']} GHz: baseline or foreign-object run missing - skipped.")
                        continue

                    # Check if the result files for the current cookie and band exist.
                    missing = [r['results_folder'] for r in (r_base, r_fo)
                               if not _cookie_sides_in_run_folder(_resolve_cookie_run_folder(study_dir, r['results_folder']))]
                    if missing:
                        print(f"show_cookie_detectability_frequency_results(): {name}, {b['band']} GHz: no result files in {missing} - skipped.")
                        continue

                    # Load the baseline data and the foreign-object data for the current cookie and band.
                    H_b, f_b, *_ = _load(r_base, sd)
                    H, f, *_ = _load(r_fo, sd)

                    # Determine the frequency range for the current band.
                    lo_hz, hi_hz = float(r_fo['f_low_ghz']) * 1e9, float(r_fo['f_high_ghz']) * 1e9
                    in_b, in_f = (f_b >= lo_hz) & (f_b <= hi_hz), (f >= lo_hz) & (f <= hi_hz)

                    # Identify the frequency bins that fall within the current band's range (at least two bins are required for group delay calculation)
                    if in_b.sum() < 2 or in_f.sum() < 2:
                        print(f"show_cookie_detectability_frequency_results(): {name}, {b['band']} GHz: fewer than 2 frequency bins in this band - skipped.")
                        continue

                    # If not all frequency bins are within the current band's range, trim the data accordingly.
                    if not (in_b.all() and in_f.all()):
                        H_b, f_b, H, f = H_b[..., in_b], f_b[in_b], H[..., in_f], f[in_f]
                    metrics = cls._compute_differential_metrics(
                        H, f, H_b, f_b, 
                        amp_noise_floor_db=amp_floor,
                        phase_noise_floor_deg=phase_floor)

                    # Check that all requested response metrics are available in the computed metrics.
                    for m in response_metrics:
                        if m not in metrics:
                            raise ValueError(
                                f"show_cookie_detectability_frequency_results(): unknown "
                                f"response metric {m!r}; available: {sorted(metrics)}")
                    f_c = float(r_fo['f_center_ghz'])
                    records.append(dict(
                        rx_side=sd, 
                        cookie=name, 
                        band=b['band'], 
                        f_low_ghz=float(r_fo['f_low_ghz']),
                        f_high_ghz=float(r_fo['f_high_ghz']), 
                        f_center_ghz=f_c,
                        thickness_mm=float(r_fo['thickness_mm']),
                        fo_size_mm=float(r_fo['fo_size_mm']),
                        fo_size_over_lambda0=float(r_fo['fo_size_mm']) * 1e-3 * f_c * 1e9 / c0,
                        **metrics))

        # If no records were collected, return an empty DataFrame
        if not records:
            print("show_cookie_detectability_frequency_results(): nothing to score yet.")
            return pd.DataFrame()

        # Convert the collected records into a DataFrame and sort them for further analysis
        results_all = pd.DataFrame(records)
        order = {n: i for i, n in enumerate(cookie_names)}
        side_order = {sd: i for i, sd in enumerate(sides)}
        results_all = results_all.sort_values(
            ['rx_side', 'cookie', 'f_center_ghz'],
            key=lambda sr: sr.map(order) if sr.name == 'cookie' else (sr.map(side_order) if sr.name == 'rx_side' else sr)
        ).reset_index(drop=True)

        # Compute the worst-case ratio over the floor for each record and add it to the DataFrame
        worst = np.zeros(len(results_all))
        for m in response_metrics:
            if floors[m] is None:
                continue
            r = results_all[m].to_numpy() / floors[m]
            results_all[f'{_cookie_metric_short_name(m)}_over_floor'] = np.round(r, 3)
            worst = np.maximum(worst, r)
        results_all['combined_over_floor'] = np.round(worst, 3)
        results_all.to_csv(out_dir / 'cookie_detectability_frequency_results.csv', index=False)

        # Define the columns to include in the ratio table for further analysis
        ratio_cols = ['rx_side', 
                      'cookie', 
                      'band', 
                      'f_center_ghz', 
                      'fo_size_over_lambda0'] + [
            c for c in results_all.columns if c.endswith('_over_floor')] + [
            c for c in ('n_exceed_max', 'n_exceed_mean') if c in results_all]
        results_all[ratio_cols].to_csv(out_dir / 'cookie_detectability_frequency_ratio_by_band.csv', index=False)

        # Loop over each side to generate per-side analysis (summary and sensitivity) for each cookie
        summary, sens = [], []
        for sd in sides:

            # Filter results for the current side and drop the 'rx_side' column
            suffix = '' if sd == 'legacy' else f'_{sd}'
            results = results_all[results_all['rx_side'] == sd].drop(columns='rx_side').reset_index(drop=True)

            # Skip processing if there are no results for the current side
            if results.empty:
                continue

            # Loop over each cookie to generate summary statistics for the current side
            for name in cookie_names:

                # Get results for the current cookie
                d = results[results['cookie'] == name]
                if d.empty:
                    continue
                above = d[d['combined_over_floor'] > 1.0]

                # Construct a summary row for the current cookie and side
                row = dict(rx_side=sd, 
                           cookie=name, 
                           n_bands=len(d), 
                           n_bands_above_floor=len(above),
                           bands_above_floor=' / '.join(above['band']) if len(above) else '',
                           strongest_band=d.loc[d['combined_over_floor'].idxmax(), 'band'],
                           weakest_band=d.loc[d['combined_over_floor'].idxmin(), 'band'],
                           min_ratio=d['combined_over_floor'].min(),
                           max_ratio=d['combined_over_floor'].max())

                # Compute log-log slope for each response metric
                for m in response_metrics:

                    vals = d[m].to_numpy(dtype=float)
                    ok = vals > 0
                    slope = (float(np.polyfit(np.log(d['f_center_ghz'].to_numpy()[ok]), np.log(vals[ok]), 1)[0]) if ok.sum() >= 3 else np.nan)
                    
                    row[f'loglog_slope_{_cookie_metric_short_name(m)}'] = slope
                
                summary.append(row)

            # Compute cookie detectability sensitivity for each SNR value
            for snr in snr_db_sweep:

                # Compute the floor values for each response metric at the current SNR
                fl = {m: _crb_floor_for_metric(m, snr) for m in response_metrics}

                # Iterate over each cookie to compute its detectability sensitivity at the current SNR
                for name in cookie_names:

                    d = results[results['cookie'] == name]
                    if d.empty:
                        continue

                    # Compute the ratio of each response metric to its floor value
                    ratio = np.zeros(len(d))
                    for m in response_metrics:
                        if fl[m] is not None:
                            ratio = np.maximum(ratio, d[m].to_numpy() / fl[m])
                    row = dict(rx_side=sd, 
                               snr_db=snr, 
                               cookie=name)
                    
                    for m in response_metrics:
                        row[f'floor_{_cookie_metric_short_name(m)}'] = (
                            None if fl[m] is None else round(fl[m], 4))
                    row['n_bands'] = len(d)
                    row['n_bands_above_floor'] = int((ratio > 1.0).sum())
                    row['bands_above_floor'] = ' / '.join(d['band'][ratio > 1.0])

                    sens.append(row)

            # Plot cookie detectability frequency overview and individual plots
            _plot_cookie_detectability_frequency(
                results, 
                response_metrics, 
                floors, 
                out_dir / f'cookie_detectability_frequency_overview{suffix}.png')
            
            _plot_cookie_detectability_frequency(
                results, 
                response_metrics, 
                floors, 
                out_dir / f'cookie_detectability_frequency{suffix}.png')

        # Export summary and sensitivity results to CSV files
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(out_dir / 'cookie_detectability_frequency_summary.csv', index=False)
        
        # Export sensitivity results if available
        if sens:
            pd.DataFrame(sens).to_csv(out_dir / 'cookie_detectability_frequency_snr_sensitivity.csv', index=False)
        results = results_all
        _export_cookie_frequency_paper_tables(results_all, 
                                              sides, 
                                              response_metrics, 
                                              floors, 
                                              snr_db_sweep, 
                                              out_dir)

        print(f"\nshow_cookie_detectability_frequency_results(): done - see {out_dir}")
        print(summary_df.to_string(index=False))

        return results

    @classmethod
    def run_moving_sample_line_array_simulation(cls, 
            base_parms_file: str, 
            consts_file: str, 
            cookie_parms_file: str, 
            fo_size_mm: float = 7.0, 
            fo_position_fraction=(0.3, 0.3, 0.5),
            fo_material: str = 'PVC', 
            n_rx: int = None, 
            rx_pitch_mm: float = 3.0,
            rx_gap_below_belt_mm: float = 5.0, 
            standoff_m: float = 0.35,
            step_mm: float = 3.0, 
            scan_half_range_mm: float = None,
            domain_length_factor: float = 2.0, 
            domain_width_factor: float = 1.0,
            edge_clearance_mm: float = 0.0,
            output_dir: str = 'studies/moving sample line array study',
            retry_failed: bool = True, 
            pilot: bool = False, 
            dry_run: bool = False,
            stop_if_first_run_fails: bool = True,
            homogenize_layers: bool = False, 
            simulate_baseline: bool = True,
            simulate_empty: bool = True, 
            fo_lift_mm: float = 0.0, 
            fo_air_gap_mm: float = 0.0) -> 'pd.DataFrame':
        """This function runs a moving sample line array simulation for a cookie with a foreign object (FO) moving along the x-axis under one fixed transmitter (TX) and a line array of receivers (RX) (MODE 12). MODE 13 and 14 analyze the resulting data from these simulations.

        Args:
            base_parms_file (str): 
                Parameter file giving band, belt, N_PML, GPU settings.
            consts_file (str): 
                Constants file
            cookie_parms_file (str): 
                Cookie overlay (parameters/cookie_types/*.json).
            fo_size_mm (float): 
                FO cube edge [mm] (default 7).
            fo_position_fraction (tuple or str): 
                FO POSITION_RELATIVE [x, y, z] in the cookie's bounding box (default (0.3, 0.3, 0.5)); 'RING_LEFT' (FREEFORM only) = left of the center between outer and jelly edge, on the belt (_freeform_fo_position_fraction()).
            fo_material (str): 
                Key of consts.json MATERIAL_PROPERTIES.
            n_rx (int): 
                Receivers along y; None (default) = as many as fit into the inner domain at rx_pitch_mm (centered on y).
            rx_pitch_mm (float): 
                Requested RX spacing in mm (default 3); snapped to a whole number of grid cells so receivers are equally spaced.
            rx_gap_below_belt_mm (float): 
                Distance of the RX line below the belt's lower face [mm] (default 5); snapped to whole grid cells.
            standoff_m (float): 
                TX height above the belt [m] (default 0.35); only sets the TX direction.
            step_mm (float): 
                Conveyor increment in mm (default 3).
            scan_half_range_mm (float): 
                Scan from -range to +range around the TX axis [mm]; None = as far as the sample stays clear of the PML.
            domain_length_factor (float): 
                Domain x-length relative to the overlay's L_X_GRID (default 2).
            domain_width_factor (float): 
                Domain y-width relative to the overlay's L_Y_GRID (default 1); raise it to fit more receivers.
            edge_clearance_mm (float): 
                Extra clearance to the PML in mm.
            output_dir (str): 
                Study folder.
            retry_failed (bool): 
                Retry FAILED manifest rows.
            fo_lift_mm (float): 
                FO raised by this much above its position [mm] (e.g. off the belt).
            fo_air_gap_mm (float): 
                Air hole drilled from below: from the belt to the FO top, this much wider per side in mm (FO runs only).
            pilot (bool): 
                Only the centered position (empty + baseline + FO = 3 runs).
            dry_run (bool): 
                Print the plan and memory estimate, run nothing.
            stop_if_first_run_fails (bool): 
                Stop if the first run of this call fails.
            homogenize_layers (bool): 
                Flatten a layered cookie (comparison only).
            simulate_baseline (bool): 
                Also simulate the 'baseline' run per position (needed by MODE 13, optional for MODE 14).
            simulate_empty (bool): 
                Also simulate the 'empty' run per position (needed by MODE 13, optional for MODE 14).

        Returns:
            pandas.DataFrame: The manifest (dry_run: the planned runs).

        Raises:
            ValueError: Scan range does not fit, the FO does not fit in the cookie, or n_rx receivers at rx_pitch_mm do not fit in y.
        """
        # Create study and output directories if they do not exist
        study_dir = Path(output_dir)
        out_dir = study_dir / 'results_analysis'

        # Load base parameters and overlay configuration
        base_parms = cls._load_config(base_parms_file)
        if isinstance(base_parms.get('FREQUENCY_RANGE'), list):
            base_parms['FREQUENCY_RANGE'] = tuple(base_parms['FREQUENCY_RANGE'])
        base_parms['PATH_INPUT_OUTPUT_FILES'] = str(study_dir / 'input_output_files')
        
        # Load inclusion configuration for the cookie parameters
        overlay = cls._load_config(cookie_parms_file)
        sut = overlay['SUT']

        # Determine cookie geometry and check dimensions
        thickness_mm = _cookie_total_thickness_mm(sut)
        if sut['SHAPE'] == 'CYLINDER':
            half_x = sut['SIZE']['DIAMETER'] / 2
            half_y = half_x
            extent_x, extent_y = 2 * half_x, 2 * half_x
        else:
            extent_x, extent_y = sut['SIZE'][0], sut['SIZE'][1]
            half_x, half_y = extent_x / 2, extent_y / 2
        
        fo_m = fo_size_mm / 1000.0
        if isinstance(fo_position_fraction, str):
            if fo_position_fraction != 'RING_LEFT' or sut['SHAPE'] != 'FREEFORM':
                raise ValueError(f"run_moving_sample_line_array_simulation(): fo_position_fraction {fo_position_fraction!r} is only defined as 'RING_LEFT' for a SHAPE 'FREEFORM' cookie.")
            fo_position_fraction = _freeform_fo_position_fraction(sut, fo_size_mm)

        # Compute FO position in the cookie frame and check if it fits within the cookie
        fx, fy, fz = fo_position_fraction
        fz += fo_lift_mm / thickness_mm
        fo_position_fraction = [fx, fy, fz]
        fo_cx_c = (fx - 0.5) * extent_x
        fo_cy_c = (fy - 0.5) * extent_y
        fo_cz = fz * thickness_mm / 1000.0

        # Determine moist shell (+ air gap)of drilling where FO sits in
        if sut.get('FREEFORM', {}).get('FO_SHELL'):
            h_ = fo_m / 2 + fo_air_gap_mm / 1000.0
            sut['FREEFORM']['FO_SHELL']['BOX'] = [fo_cx_c - h_, 
                                                  fo_cx_c + h_, 
                                                  fo_cy_c - h_, 
                                                  fo_cy_c + h_, fo_cz + fo_m / 2]

        # Check if FO fits within the cookie boundaries, raise an error if not
        if (abs(fo_cx_c) + fo_m / 2 > half_x + 1e-9 or abs(fo_cy_c) + fo_m / 2 > half_y + 1e-9 or fo_cz - fo_m / 2 < -1e-9 or fo_cz + fo_m / 2 > thickness_mm / 1000.0 + 1e-9):
            raise ValueError(
                f"run_moving_sample_line_array_simulation(): a {fo_size_mm:g} mm FO at position fraction {tuple(fo_position_fraction)} does not fit inside the {extent_x * 1e3:g} x {extent_y * 1e3:g} x {thickness_mm:g} mm cookie.")

        # Set up array geometry parameters for the simulation
        pitch_req_m = rx_pitch_mm / 1000.0
        pitch_m = pitch_req_m
        belt_thickness = base_parms['BELT']['THICKNESS']
        rx_z_rel = -(belt_thickness + rx_gap_below_belt_mm / 1000.0)
        array_parms = copy.deepcopy(base_parms)
        array_parms.update({'N_TX': 1,
                            'TX_POS_RANGE': [[0.0, 0.0], 0.0, standoff_m],
                            'N_RX': 2,
                            'RX_POS_RANGE': [[0.0, 0.0], 
                                             [-pitch_m / 2, pitch_m / 2], 
                                             rx_z_rel],
                            'RX_MODE': 'DIRECT',
                            'ALSO_MIRRORED_RX': False,
                            'RX_PROJECTION': 'POLARIZATION',
                            'PLANE_WAVE_ANGLES': 'CORRECT',})

        # Determine the grid lengths for the simulation domain based on overlay and scaling factors
        l_x = domain_length_factor * overlay.get('GRID', {}).get(
            'L_X_GRID', base_parms['L_X_GRID'])
        l_y = domain_width_factor * overlay.get('GRID', {}).get(
            'L_Y_GRID', base_parms['L_Y_GRID'])

        # Define a helper function to build the simulation parameters for each run (case = 'fo' | 'baseline' | 'empty')
        def _parms(offset_m: float, case: str) -> dict:

            # Build the parameters
            parms = _build_parms_for_cookie_run(array_parms, 
                                                overlay, 
                                                fo_size_mm, 
                                                fo_material, 
                                                baseline=(case != 'fo'),homogenize_layers=homogenize_layers, fo_clear_mm=fo_size_mm,
                fo_position_fraction=list(fo_position_fraction))
            
            # Adjust the parameters based on the case type ('empty' or 'fo')
            if case == 'empty':
                parms['SUT']['MODE'] = 'SURROUNDING'
                parms['FOREIGN_OBJECT']['MODE'] = 'MATERIAL'
                parms['FOREIGN_OBJECT']['MATERIAL'] = 'AIR'
            if case == 'fo' and fo_air_gap_mm > 0:
                parms['FOREIGN_OBJECT']['AIR_GAP'] = fo_air_gap_mm / 1000.0

            parms['L_X_GRID'] = l_x
            parms['L_Y_GRID'] = l_y
            parms['SUT_OFFSET'] = [float(offset_m), 0.0]
            return parms

        # Scan positions
        #---------------
        # Initialize the EM simulation with the baseline parameters (no foreign object)
        em0 = cls(parms=_parms(0.0, 'baseline'), consts=consts_file)
        em0._determine_number_of_elements()

        # Snap the RX pitch to a whole number of grid cells. This is required toensure equally spaced receivers.
        cells_per_pitch = max(1, int(round(pitch_req_m / em0.dx)))
        pitch_m = cells_per_pitch * em0.dx
        print(f"run_moving_sample_line_array_simulation(): RX pitch {rx_pitch_mm:g} mm requested = {pitch_req_m / em0.dx:.2f} grid cells (dx = {em0.dx * 1e3:.4f} mm) -> snapped to {cells_per_pitch} cells = {pitch_m * 1e3:.4f} mm so that all receivers are equally spaced.")
        
        # Snap the RX gap to whole cells below the belt's lower face as gprMax grids it
        i_belt = int(np.floor(round(em0.l_z_grid / 2 - belt_thickness, 4) / em0.dz + 0.5))
        cells_gap = max(1, int(round(rx_gap_below_belt_mm / 1000.0 / em0.dz)))
        rx_z_rel = (i_belt - cells_gap) * em0.dz - em0.l_z_grid / 2
        gap_mm = cells_gap * em0.dz * 1e3
        print(f"run_moving_sample_line_array_simulation(): RX gap {rx_gap_below_belt_mm:g} mm below the belt requested = {rx_gap_below_belt_mm / 1e3 / em0.dz:.2f} grid cells -> snapped to {cells_gap} cells = {gap_mm:.4f} mm.")
        
        # Determine the usable inner domain dimensions for placing the receivers in the y-direction. Determine the number of receivers that can fit.
        half_inner = l_x / 2 - (em0.n_pml + em0.n_pml_margin) * em0.dx
        half_inner_y = l_y / 2 - (em0.n_pml + em0.n_pml_margin) * em0.dx - 2 * em0.dx
        n_fit = int(np.floor(2 * half_inner_y / pitch_m + 1e-9)) + 1
        if n_rx is None:
            n_rx = n_fit if n_fit % 2 else n_fit - 1
        elif n_rx > n_fit:
            raise ValueError(
                f"run_moving_sample_line_array_simulation(): {n_rx} receivers at {pitch_m * 1e3:.4f} mm pitch span {(n_rx - 1) * pitch_m * 1e3:.2f} mm in y, but the PML-free inner domain ({l_y * 1e3:g} mm wide) holds only {n_fit} ({(n_fit - 1) * pitch_m * 1e3:.2f} mm). Use n_rx=None / <= {n_fit}, a smaller rx_pitch_mm, or widen the domain with domain_width_factor (>= {(1e3 * ((n_rx - 1) * pitch_m + 2 * (l_y / 2 - half_inner_y))) / (l_y * 1e3):.1f}).")
        
        # Determine the half span of the receiver array in the y-direction
        half_span = (n_rx - 1) * pitch_m / 2
        array_parms['N_RX'] = n_rx
        array_parms['RX_POS_RANGE'] = [[0.0, 0.0], 
                                       [-half_span, half_span], 
                                       rx_z_rel]

        # Determine the maximum allowed offset for the sample movement in the x-direction and step size
        max_offset = half_inner - half_x - edge_clearance_mm / 1000.0
        step_m = step_mm / 1000.0
        if scan_half_range_mm is None:
            n_side = int(np.floor(max_offset / step_m + 1e-9))
        else:
            n_side = int(round(scan_half_range_mm / step_mm))
            if n_side * step_m > max_offset + 1e-9:
                raise ValueError(
                    f"run_moving_sample_line_array_simulation(): scan_half_range_mm={scan_half_range_mm:g} puts the cookie edge into the PML; at most {1e3 * max_offset:.2f} mm fits ({l_x * 1e3:g} mm domain).")
        if n_side < 1:
            raise ValueError(
                "run_moving_sample_line_array_simulation(): the domain leaves no room to move the sample - raise domain_length_factor.")
        
        # Compute the array of sample offsets in meters and determine the number of positions
        offsets_m = step_m * np.arange(-n_side, n_side + 1)
        if pilot:
            offsets_m = np.array([0.0])
        n_pos = len(offsets_m)

        # Perform a fixed-position check: ensure that TX and RX coordinates remain constant for key sample offsets.
        ref = None
        for off in (offsets_m[0], offsets_m[len(offsets_m) // 2], offsets_m[-1]):
            # Instantiate the EM simulation object for the current sample offset and determine the antenna locations.
            em_c = cls(parms=_parms(float(off), 'fo'), 
                       consts=consts_file)
            em_c._determine_antenna_location_huygens()

            # Store the current TX, RX, and sample center positions for comparison.
            cur = (np.array(em_c.tx_pos), np.array(em_c.rx_pos), em_c.sut_cx)
            if ref is None:
                ref = cur
            elif not (np.array_equal(cur[0], ref[0]) and np.array_equal(cur[1], ref[1])):
                raise RuntimeError(
                    "run_moving_sample_line_array_simulation(): TX/RX coordinates change with the conveyor position - they must stay fixed.")
        
        print(f"run_moving_sample_line_array_simulation(): fixed-position check passed - TX at ({ref[0][0, 0] * 1e3:.2f}, {ref[0][1, 0] * 1e3:.2f}, {ref[0][2, 0] * 1e3:.2f}) mm and {n_rx} RX at x = {ref[1][0, 0] * 1e3:.2f} mm, y = {ref[1][1].min() * 1e3:.2f}..{ref[1][1].max() * 1e3:.2f} mm, z = {ref[1][2, 0] * 1e3:.2f} mm in every run; the cookie center moves from x = {(l_x / 2 + offsets_m[0]) * 1e3:.1f} to {(l_x / 2 + offsets_m[-1]) * 1e3:.1f} mm.")

        # Prepare the list of simulation cases to run based on the specified flags.
        cases = [c for c, on in (('empty', simulate_empty), 
                                 ('baseline', simulate_baseline), ('fo', True)) if on]
        case_label = ', '.join({'empty': 'empty belt', 'baseline': 'baseline', 'fo': 'FO'}[c] for c in cases)

        # If the sample under test has a freeform shape, plot its geometry
        keep_fig = None
        if sut['SHAPE'] == 'FREEFORM':
            keep_fig = plot_freeform_cookie_geometry(
                copy.deepcopy(sut), 
                em0.dx,
                fo=dict(center_mm=[1e3 * fo_cx_c, 1e3 * fo_cy_c, 1e3 * fo_cz], 
                        size_mm=fo_size_mm, 
                        material=fo_material,
                        air_gap_mm=fo_air_gap_mm),
                out_path=out_dir / f"cookie_geometry_{Path(cookie_parms_file).stem}.png")['fig_number']

        # Prepare the list of jobs for each combination of offset and simulation case.
        jobs = []
        for m, off in enumerate(offsets_m):
            for case in cases:
                jobs.append(dict(position_index=str(m), 
                                 offset_mm=1e3 * float(off), 
                                 case=case))

        print(f"run_moving_sample_line_array_simulation(): {overlay['COOKIE_NAME']}: {n_pos} position(s) from {1e3 * offsets_m[0]:+.1f} to {1e3 * offsets_m[-1]:+.1f} mm in {step_mm:g} mm steps x {len(cases)} case(s) ({case_label}) = {len(jobs)} runs; domain {l_x * 1e3:g} x {l_y * 1e3:g} mm (sample clear of the PML up to +-{1e3 * max_offset:.1f} mm); TX above the center (standoff {standoff_m:g} m), {n_rx} RX along y {gap_mm:.4f} mm below the belt, pitch {pitch_m * 1e3:.4f} mm (+-{half_span * 1e3:.1f} mm). FO {fo_size_mm:g} mm {fo_material} at cookie-frame (x, y) = ({fo_cx_c * 1e3:+.1f}, {fo_cy_c * 1e3:+.1f}) mm, {fo_cz * 1e3:.1f} mm above the belt.")

        # If dry_run is enabled, estimate memory requirements for each job and return early.
        if dry_run:
            def _estimate(parms_dict, label):
                em = cls(parms=parms_dict, consts=consts_file)
                em._determine_number_of_elements()
                em._determine_time_parameters()
                em._check_farfield_condition()
                em._determine_material_properties()
                em._determine_antenna_location_huygens()
                em._compute_receiver_geometry()
                est = len(em.rx_positions) * 6 * em.n_time * 4 / 1e9
                print(f"run_moving_sample_line_array_simulation(): [{label}] "
                      f"{len(em.rx_positions)} receivers, n_time = {em.n_time}, "
                      f"{int(em.band_mask.sum()) if hasattr(em, 'band_mask') else 'n/a'} band bins, "
                      f"gprMax receiver memory ~{est:.2f} GB per run.")
            _estimate(_parms(0.0, 'fo'), 'direct receivers')
            return pd.DataFrame(jobs)

        # Create the output directory if it doesn't exist.
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the scan configuration dictionary
        config = dict(
            cookie_name=overlay['COOKIE_NAME'], 
            cookie_parms_file=str(cookie_parms_file),
            sut_shape=sut['SHAPE'], 
            sut_size=sut['SIZE'],
            sut_thickness_mm=thickness_mm,
            fo_size_mm=fo_size_mm, 
            fo_material=fo_material,
            fo_position_fraction=list(fo_position_fraction),
            fo_center_cookie_frame_mm=[1e3 * fo_cx_c, 
                                       1e3 * fo_cy_c, 
                                       1e3 * fo_cz],
            fo_lift_mm=fo_lift_mm, 
            fo_air_gap_mm=fo_air_gap_mm,
            step_mm=step_mm, 
            offsets_mm=[1e3 * float(o) for o in offsets_m],
            n_rx=n_rx, rx_pitch_mm=pitch_m * 1e3, rx_pitch_requested_mm=rx_pitch_mm, 
            rx_gap_below_belt_mm=gap_mm, rx_gap_requested_mm=rx_gap_below_belt_mm,
            standoff_m=standoff_m, 
            rx_mode='DIRECT',
            has_empty=bool(simulate_empty), 
            has_baseline=bool(simulate_baseline), 
            rx_y_inner_half_mm=1e3 * half_inner_y,
            domain_m=[l_x, 
                      l_y, 
                      base_parms['L_Z_GRID']],
            frequency_range_hz=list(base_parms['FREQUENCY_RANGE']),
            pilot=bool(pilot))

        # If freeform, compute the outline and region outlines in mm
        if sut['SHAPE'] == 'FREEFORM':
            ox, oy = _freeform_outline(sut)
            config['sut_outline_mm'] = [[1e3 * float(a) for a in ox], [1e3 * float(b) for b in oy]]
            config['sut_region_outlines_mm'] = [
                [[1e3 * float(a) for a in rx_], 
                 [1e3 * float(b) for b in ry_]]
                for rx_, ry_ in (_freeform_region_outline(reg) for reg in sut.get('FREEFORM', {}).get('REGIONS', []))]
        
        # Write the scan configuration to a JSON file
        with open(out_dir / 'scan_config.json', 'w') as fh:
            json.dump(config, fh, indent=1)

        # Prepare the manifest file for resumable runs
        manifest_path = out_dir / 'manifest_moving_sample_line_array.csv'
        columns = ['position_index', 
                   'offset_mm', 
                   'case', 
                   'fo_size_mm',
                   'results_folder', 
                   'status', 
                   'error']

        # Load the existing manifest if it exists. Otherwise create a new one.
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path, 
                                   dtype={'position_index': str, 
                                          'case': str, 
                                          'results_folder': str, 
                                          'error': str})
        else:
            manifest = pd.DataFrame(columns=columns)

        # Define helper function to find a job in the manifest based on position index, case, and offset.
        def _find(position_index, case, offset_mm):
            hit = manifest[(manifest['position_index'] == str(position_index)) & (manifest['case'] == case)
            & np.isclose(pd.to_numeric(manifest['offset_mm']), float(offset_mm), atol=1e-6)]
            return hit.index[0] if len(hit) else None

        # Iterate over all jobs and process them if they are not already marked as 'OK' in the manifest.
        t_done, n_done = 0.0, 0
        for j, job in enumerate(jobs):

            # Check if the current job is already in the manifest and determine its status.
            idx = _find(job['position_index'], job['case'], job['offset_mm'])
            status = manifest.loc[idx, 'status'] if idx is not None else None
            if status == 'OK' or (status == 'FAILED' and not retry_failed):
                continue
            print(f"\n=== moving-sample run {j + 1}/{len(jobs)}: position {job['position_index']} (offset {job['offset_mm']:+.1f} mm), case '{job['case']}' ===")
            
            # Initialize a row dictionary for the manifest with the current job's details.
            t0 = time.time()
            row = dict(position_index=job['position_index'], 
                       offset_mm=job['offset_mm'],
                       case=job['case'], 
                       fo_size_mm=fo_size_mm)

            # Attempt to run the simulation for the current job and update the manifest accordingly.
            try:
                em = cls(parms=_parms(job['offset_mm'] / 1e3, job['case']),
                         consts=consts_file)
                em.protocol_context = dict(
                    study='moving sample line array (MODE 12)', 
                    case=job['case'],
                    position_index=job['position_index'], conveyor_offset_mm=round(job['offset_mm'], 3),
                    base_parms_file=str(base_parms_file), 
                    cookie_parms_file=str(cookie_parms_file),
                    consts_file=str(consts_file), 
                    fo_size_mm=fo_size_mm, 
                    fo_material=fo_material,
                    standoff_m=standoff_m, 
                    rx_pitch_mm=rx_pitch_mm, 
                    rx_pitch_used_mm=round(pitch_m * 1e3, 5), rx_gap_below_belt_mm=rx_gap_below_belt_mm,
                    rx_gap_used_mm=round(gap_mm, 5))
                
                em.simulate()

                row.update(results_folder=Path(os.path.relpath(em.path_folder, study_dir)).as_posix(), status='OK', error='')
                
            except Exception as e:

                print(f"moving-sample run FAILED - {e!r}")
                row.update(results_folder='', status='FAILED', error=repr(e))

            finally:

                for num in plt.get_fignums():
                    if num != keep_fig:
                        plt.close(num)

            # Update the manifest with the current row's status and results folder
            if idx is None:
                manifest = pd.concat([manifest, pd.DataFrame([row])], 
                                     ignore_index=True)
            else:
                for k, v in row.items():
                    manifest.loc[idx, k] = v

            # Save the updated manifest to the CSV file
            manifest.to_csv(manifest_path, index=False)
            
            # Stop the batch if the first run failed and the flag is set
            if row['status'] == 'FAILED' and n_done == 0 and stop_if_first_run_fails:
                raise RuntimeError(
                    f"run_moving_sample_line_array_simulation(): the first run (position {job['position_index']}, '{job['case']}') failed: {row['error']} - stopped before the rest of the batch (manifest: {manifest_path}; pass stop_if_first_run_fails=False to continue anyway).")
            
            # Update counters and estimate remaining time
            n_done += 1
            t_done += time.time() - t0
            left = sum(1 for jb in jobs[j + 1:]
                       if (lambda i: i is None or manifest.loc[i, 'status'] != 'OK')(_find(jb['position_index'], jb['case'], jb['offset_mm'])))
            print(f"moving-sample study: {n_done} simulated this session, avg {_format_duration(t_done / n_done)}/run, {left} left - ETA {_format_duration(left * t_done / n_done)}.")

        # Check completeness: ensure every OK run has its corresponding data.pkl file
        planned = {(jb['position_index'], jb['case']) for jb in jobs}
        ok = manifest[(manifest['status'] == 'OK') & manifest.apply(
            lambda r: (r['position_index'], r['case']) in planned, axis=1)]
        
        # Identify OK runs that are missing their data.pkl file
        missing = [r for r in ok['results_folder']
                   if not (_resolve_cookie_run_folder(study_dir, r) / 'data.pkl').is_file()]
        n_failed = int((manifest['status'] == 'FAILED').sum())
        
        # Count the number of runs that are still open (not OK)
        n_open = len(jobs) - len(ok)

        # Report the current status of the simulation runs. For MODE 13 and 14, the baseline simulation and empty-belt runs are required before analyzing the moving sample line array.
        print(f"\nrun_moving_sample_line_array_simulation(): {len(ok) - len(missing)} of {len(jobs)} runs have a data.pkl; {n_failed} FAILED, {n_open} not OK yet{f', {len(missing)} OK rows lack data.pkl: {missing}' if missing else ''}.")
        if not n_open and not missing:
            nxt = ("MODE == 13 (analyze_moving_sample_line_array()) or MODE == 14" if simulate_baseline and simulate_empty
                   else "MODE == 14 (analyze_moving_sample_camera(); MODE == 13 needs the baseline and empty-belt runs)")
            print(f"run_moving_sample_line_array_simulation(): all data collected - run {nxt} next{' (this was only the pilot position: set MSLA_PILOT = False for the full scan)' if pilot else ''}.")
        
        return manifest

# endregion

def _build_parms_for_doe_row(base_parms: dict, 
                             row: pd.Series,
                             water_fraction_low: float,
                             water_fraction_high: float = WATER_FRACTION_HIGH,porous_air_fraction: float = POROUS_AIR_FRACTION
                              ) -> dict:
    """This function constructs a complete parameter dictionary for a single DOE row by deep-copying the base parameters and overriding them with the specific values from the row.

    Args:
        base_parms (dict): 
            The base parameter dictionary to copy.
        row (pd.Series): 
            A row from the DOE CSV containing the specific factor levels.
        water_fraction_low (float): 
            The low water fraction for the simulation.
        water_fraction_high (float, optional): 
            The high water fraction for the simulation. Defaults to WATER_FRACTION_HIGH.
        porous_air_fraction (float, optional): 
            The porous air fraction for the simulation. Defaults to POROUS_AIR_FRACTION.
    
    Returns:
        dict: A complete parameter dictionary for the given DOE row.

    Raises:
        ValueError: If the material or position in the row is not recognized, or if the depth exceeds the SUT thickness.
    """

    # Get run label
    parms = copy.deepcopy(base_parms)
    run_label = row['run'] if 'run' in row else '?'

    # Convert diameter from mm to meters and set the foreign object size accordingly.
    d_m = row['diameter_mm'] / 1000.0
    parms['FOREIGN_OBJECT']['SIZE'] = [d_m, d_m, d_m]

    # Check material against allowed list and set it in the parameters
    material = row['material']
    if material not in FO_MATERIALS_ALLOWED:
        raise ValueError(
            f"_build_parms_for_doe_row(): unknown material level {material!r} in CSV row (run={run_label}) - add it to FO_MATERIALS_ALLOWED (and to consts.json's MATERIAL_PROPERTIES if it isn't defined there yet).")
    parms['FOREIGN_OBJECT']['MATERIAL'] = material

    #  The FO position is specified relative to the SUT, with x and y fractions and a z fraction computed from depth. Check that the position is allowed and compute the z fraction accordingly.
    position = row['position']
    if position not in POSITION_X_FRACTION:
        raise ValueError(
            f"_build_parms_for_doe_row(): unknown position level {position!r} in CSV row (run={run_label}) - add it to POSITION_X_FRACTION.")
    thickness_m = _get_sut_thickness_m(base_parms['SUT'])
    depth_m = row['depth_mm'] / 1000.0
    if depth_m > thickness_m:
        raise ValueError(
            f"_build_parms_for_doe_row(): depth_mm={row['depth_mm']} in CSV row (run={run_label}) exceeds the base parms file's SUT thickness ({thickness_m * 1000:.2f} mm) - the object can't be embedded that deep.")
    z_fraction = 1.0 - depth_m / thickness_m
    parms['FOREIGN_OBJECT']['POSITION_RELATIVE'] = [
        POSITION_X_FRACTION[position], 0.5, z_fraction]

    # Map the moisture level from the CSV row to the target water fraction in the SUT recipe.
    moisture = row['moisture']
    if moisture == 'Low':
        target_water = water_fraction_low
    elif moisture == 'High':
        target_water = water_fraction_high
    else:
        raise ValueError(
            f"_build_parms_for_doe_row(): unknown moisture level {moisture!r} in CSV row (run={run_label}).")
    fractions = _rescale_fractions(
        base_parms['SUT']['INGREDIENTS_FRACTIONS'], 'WATER', target_water)

    # Map the heterogeneity level (air mixing porosity) from the CSV row to the target air fraction in the SUT recipe.
    heterogeneity = row['heterogeneity']
    if heterogeneity == 'Porous/Layered':
        fractions = _rescale_fractions(fractions, 'AIR', porous_air_fraction)
    elif heterogeneity != 'Homogeneous':
        raise ValueError(
            f"_build_parms_for_doe_row(): unknown heterogeneity level {heterogeneity!r} in CSV row (run={run_label}).")
    parms['SUT']['INGREDIENTS_FRACTIONS'] = fractions

    return parms


def _ingredients_to_volume_fractions(mass_fractions: dict,
                                     air_volume_fraction: float = 0.0) -> dict:
    """This function converts mass fractions of solid ingredients to volume fractions, taking into account the air volume fraction if specified.

    Args:
        mass_fractions (dict): 
            Ingredient -> mass fraction (~sums to 1).
        air_volume_fraction (float): 
            Air share of the total volume, 0 <= a < 1.

    Returns:
        dict: Ingredient -> volume fraction (+ 'AIR' if a > 0), sums to 1.

    Raises:
        ValueError: Unknown ingredient (add its density to _INGREDIENT_DENSITY_G_CM3) or air fraction outside [0, 1).
    """

    # Check for unknown ingredients and validate air volume fraction.
    unknown = [k for k in mass_fractions if k not in _INGREDIENT_DENSITY_G_CM3]
    if unknown:
        raise ValueError(
            f"_ingredients_to_volume_fractions(): no density for {unknown} - add it to _INGREDIENT_DENSITY_G_CM3.")
    if not 0.0 <= air_volume_fraction < 1.0:
        raise ValueError(
            f"_ingredients_to_volume_fractions(): air_volume_fraction must be in [0, 1), got {air_volume_fraction!r}.")

    # Convert mass fractions to volume fractions for the solid ingredients.
    specific_volume = {k: w / _INGREDIENT_DENSITY_G_CM3[k]
                       for k, w in mass_fractions.items() if w > 0}
    total = sum(specific_volume.values())
    out = {k: (1.0 - air_volume_fraction) * v / total for k, v in specific_volume.items()}

    # Include air volume fraction if specified.
    if air_volume_fraction > 0:
        out['AIR'] = float(air_volume_fraction)

    return out


def _resolve_fraction_basis(block: dict) -> None:
    """This function modifies the block in place, converting its ingredient fractions to volume fractions as required by EMSimulation.

    Args:
        block (dict): 
            A SUT or layer block containing ingredient fractions and optional overlay keys.

    Raises:
        ValueError: Bad basis, COMPONENTS combined with INGREDIENTS_FRACTIONS, or VOLUME_SHAREs not summing to 1.

    Returns:
        None
    """
    # Handle the case where the block has components.
    if 'COMPONENTS' in block:

        # If the block has COMPONENTS, it cannot have INGREDIENTS_FRACTIONS directly.
        if 'INGREDIENTS_FRACTIONS' in block:
            raise ValueError("_resolve_fraction_basis(): a block cannot have both 'COMPONENTS' and 'INGREDIENTS_FRACTIONS'.")
        components = block.pop('COMPONENTS')

        # Ensure that the volume shares of the components sum to 1.
        if abs(sum(c['VOLUME_SHARE'] for c in components) - 1.0) > 1e-6:
            raise ValueError("_resolve_fraction_basis(): COMPONENTS' VOLUME_SHAREs must sum to 1.")
        
        # Resolve the fractions for each component and accumulate them into the mixed dictionary.
        mixed = {}
        for component in components:
            sub = {k: component[k] for k in
                   ('INGREDIENTS_FRACTIONS', 
                    'FRACTIONS_BASIS', 
                    'AIR_VOLUME_FRACTION')
                   if k in component}
            _resolve_fraction_basis(sub)
            for ingredient, fraction in sub['INGREDIENTS_FRACTIONS'].items():
                mixed[ingredient] = mixed.get(ingredient, 0.0) + component['VOLUME_SHARE'] * fraction
        block['INGREDIENTS_FRACTIONS'] = mixed
        block.pop('FRACTIONS_BASIS', None)
        block.pop('AIR_VOLUME_FRACTION', None)
        return

    # Handle the case where the block has no components: resolve fractions based on the FRACTIONS_BASIS.
    basis = str(block.pop('FRACTIONS_BASIS', 'VOLUME')).upper()
    air = block.pop('AIR_VOLUME_FRACTION', None)
    if basis == 'MASS':
        block['INGREDIENTS_FRACTIONS'] = _ingredients_to_volume_fractions(
            block['INGREDIENTS_FRACTIONS'], 
            air or 0.0)
    elif basis == 'VOLUME':
        if air is not None:
            block['INGREDIENTS_FRACTIONS'] = _rescale_fractions(
                block['INGREDIENTS_FRACTIONS'], 
                'AIR', air)
    else:
        raise ValueError(f"_resolve_fraction_basis(): FRACTIONS_BASIS must be 'MASS' or 'VOLUME', got {basis!r}.")


def _homogenize_layers(sut: dict) -> dict:
    """This function collapses a layered SUT into a single homogeneous SUT by computing the thickness-weighted mean of the ingredient fractions.

    Args:
        sut (dict): 
            SUT dict containing 'LAYERS' and a 'SIZE' (CYLINDER dict or RECTANGULAR list).

    Returns:
        dict: New SUT dict without 'LAYERS', with 'INGREDIENTS_FRACTIONS' and an updated SIZE thickness.

    Raises:
        KeyError: If 'LAYERS' or 'SIZE' is missing in the input SUT dict.
    """
    # Get layers and thicknesses
    layers = sut['LAYERS']
    thickness = np.array([layer['THICKNESS'] for layer in layers], dtype=float)
    
    # Determine fractions for each ingredient based on thickness-weighted average
    keys = sorted({k for layer in layers for k in layer['INGREDIENTS_FRACTIONS']})
    fractions = {k: float(sum(layer['INGREDIENTS_FRACTIONS'].get(k, 0.0) * layer['THICKNESS'] for layer in layers) / thickness.sum()) for k in keys}
    total = sum(fractions.values())
    fractions = {k: v / total for k, v in fractions.items()}

    # Construct the output SUT dict (no layers) and with updated fractions and thickness
    out = {k: v for k, v in sut.items() if k != 'LAYERS'}
    out['INGREDIENTS_FRACTIONS'] = fractions
    if isinstance(out['SIZE'], dict):
        out['SIZE'] = {**out['SIZE'], 'THICKNESS': float(thickness.sum())}
    else:
        out['SIZE'] = [out['SIZE'][0], out['SIZE'][1], float(thickness.sum())]
    
    return out


def _cookie_total_thickness_mm(sut: dict) -> float:
    """This function returns the total thickness of the SUT in mm, either by summing the layer thicknesses for a layered SUT or by using the SIZE's thickness for a homogeneous SUT.

    Args:
        sut (dict): 
            SUT dict containing either 'LAYERS' or a 'SIZE' with 'THICKNESS'.

    Returns:
        float: Total thickness of the SUT in mm.
    
    Raises:
        KeyError: If neither 'LAYERS' nor 'SIZE' with 'THICKNESS' is present in the input SUT dict.
    """
    if 'LAYERS' in sut:
        return 1000.0 * float(sum(layer['THICKNESS'] for layer in sut['LAYERS']))
    
    return 1000.0 * _get_sut_thickness_m(sut)


def _pack_spheres(n: int, r_eff: float, half_x: float, half_y: float,
                  disc: bool, thickness: float, clear_half: float,
                  seed: int, max_iter: int = 3000):
    """This function attempts to pack 'n' equal spheres inside a slab while avoiding overlaps and a central keep-out box based on a Monte-Carlo growth algorithm.

    Args:
        n (int): 
            Number of spheres.
        r_eff (float): 
            Sphere radius incl. half the minimum gap [m]; walls, keep-out box and neighbors stay >= r_eff away.
        half_x, half_y (float): 
            Footprint half extents in m (disc: half_x is the radius).
        disc (bool): 
            True for a round footprint, False for a rectangle.
        thickness (float): 
            Slab thickness in m.
        clear_half (float): 
            Half edge of the keep-out box in m (0: none).
        seed (int): 
            RNG seed (the layout is deterministic).
        max_iter (int): 
            Sweep budget.

    Returns:
        np.ndarray or None: (n, 3) centers [x, y, z] relative to the footprint center / belt surface [m], or None if it did not pack.

    Raises:
        ValueError: If the spheres could not be packed within the given constraints.
    """
    # Initialize the random number generator and the mid-plane of the slab.
    rng = np.random.default_rng(seed)
    z_mid = thickness / 2

    # Define helper function to check if a sphere placement is valid.
    def ok(i, c, r, P):
        
        if disc:
            if np.hypot(c[0], c[1]) + r > half_x:
                return False
        elif abs(c[0]) + r > half_x or abs(c[1]) + r > half_y:
            return False
        
        if not (r <= c[2] <= thickness - r):
            return False
        
        if clear_half > 0:
            d = np.maximum(np.abs(c - 
                np.array([0.0, 0.0, z_mid])) - clear_half, 0.0)
            if np.linalg.norm(d) < r:
                return False
        
        dist = np.linalg.norm(P - c, axis=1)
        dist[i] = np.inf
        return bool(np.all(dist >= 2 * r))

    # Start with an initial guess for the sphere radius and an empty placement array.
    r = 0.4 * r_eff
    P = np.full((n, 3), 1e3)
    
    # Attempt to place all spheres randomly within the footprint and slab thickness (with maximum number of tries). If not all spheres could be placed, return None.
    placed, tries = 0, 0
    while placed < n and tries < 200000:
        tries += 1
        if disc:
            a, rr = 2 * np.pi * rng.random(), half_x * np.sqrt(rng.random())
            xy = [rr * np.cos(a), rr * np.sin(a)]
        else:
            xy = [rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)]
        c = np.array([xy[0], xy[1], rng.uniform(r, thickness - r)])
        if ok(placed, c, r, P):
            P[placed] = c
            placed += 1
    if placed < n:
        return None
    
    # Iteratively try to improve the sphere packing by small random displacements and radius increases.
    best_r, stall = r, 0
    for _ in range(max_iter):
        for i in rng.permutation(n):
            c = P[i] + rng.normal(0.0, 0.15 * r_eff, 3)
            if ok(i, c, r, P):
                P[i] = c
        r_new = min(r_eff, r * 1.01 + 1e-5)
        if all(ok(i, P[i], r_new, P) for i in range(n)):
            r = r_new
        if r >= r_eff - 1e-12:
            return P
        stall = 0 if r > best_r + 1e-6 else stall + 1
        best_r = max(best_r, r)
        if stall > 250:
            return None
        
    return None


def _resolve_component(component: dict) -> dict:
    """This function resolves the volume fractions for a single component, taking into account its specified fractions basis and air volume fraction.
    
    Args:
        component (dict): 
            A single component entry from COMPONENTS.

    Returns:
        dict: Resolved ingredient fractions for the component.

    Raises:
        ValueError: If the component's fractions cannot be resolved.
    """
    sub = {k: copy.deepcopy(component[k]) for k in
           ('INGREDIENTS_FRACTIONS', 
            'FRACTIONS_BASIS', 
            'AIR_VOLUME_FRACTION')
           if k in component}
    _resolve_fraction_basis(sub)

    return sub['INGREDIENTS_FRACTIONS']


def _resolve_discrete_component(sut: dict, fo_clear_mm: float) -> None:
    """This function attempts to place the discrete component as real spheres within the SUT, respecting the specified minimum gap and seed for reproducibility. If one entry of sut['COMPONENTS'] carries a 'DISCRETE' spec ({'SHAPE': 'SPHERE', 'DIAMETER' [m], 'MIN_GAP' [m], 'SEED'}),
    place that component as real spheres instead of smearing it into the effective medium.

    Args:
        sut (dict): 
            Overlay SUT (plain, with COMPONENTS).
        fo_clear_mm (float): 
            Edge of the keep-out cube at the SUT center [mm] (the largest foreign object of the study).

    Returns:
        None

    Raises:
        ValueError: More than one DISCRETE component, or no layout possible at all.
    """
    # Identify the discrete component within the SUT's components list and check for validity.
    components = sut.get('COMPONENTS') or []
    idx = [i for i, c in enumerate(components) if 'DISCRETE' in c]
    if not idx:
        return
    if len(idx) > 1:
        raise ValueError("_resolve_discrete_component(): only one DISCRETE component is supported.")
    i_disc = idx[0]
    spec = components[i_disc]['DISCRETE']
    if str(spec.get('SHAPE', 'SPHERE')).upper() != 'SPHERE':
        raise ValueError("_resolve_discrete_component(): DISCRETE SHAPE must be 'SPHERE'.")

    # Extract the discrete component's specifications
    diameter = float(spec['DIAMETER'])
    gap = float(spec.get('MIN_GAP', 0.0))
    seed = int(spec.get('SEED', 0))
    r_eff = diameter / 2 + gap / 2
    size = sut['SIZE']

    # Determine the SUT's half-dimensions and volume based on its shape.
    if sut['SHAPE'] == 'CYLINDER':
        half_x = half_y = size['DIAMETER'] / 2
        thickness = size['THICKNESS']
        volume = np.pi * half_x ** 2 * thickness
        disc = True
    else:
        half_x, half_y, thickness = size[0] / 2, size[1] / 2, size[2]
        volume = size[0] * size[1] * size[2]
        disc = False

    # Compute the clear half-dimension for the keep-out cube and the target number of spheres. Also calculate the volume of a single sphere.
    clear_half = min(fo_clear_mm / 1000.0, thickness) / 2
    v_sphere = np.pi / 6 * diameter ** 3
    share = components[i_disc]['VOLUME_SHARE']
    n_target = int(np.floor(share * volume / v_sphere + 1e-9))

    # Attempt to pack the spheres within the SUT, using a cache to avoid redundant computations.
    key = (n_target, r_eff, half_x, half_y, disc, thickness, clear_half, seed)
    if key not in _PACKING_CACHE:
        
        placed, centers = 0, None
        for n in range(n_target, 0, -1):
            centers = _pack_spheres(n, r_eff, half_x, half_y, disc, thickness, clear_half, seed)
            if centers is not None:
                placed = n
                break
        
        if centers is None:
            raise ValueError("_resolve_discrete_component(): no sphere layout found.")
        _PACKING_CACHE[key] = (placed, centers)

    placed, centers = _PACKING_CACHE[key]

    # Compute the volume fraction of the discrete component and update the matrix for the remaining components.
    f_disc = placed * v_sphere / volume
    if f_disc > share + 1e-9:
        raise ValueError("_resolve_discrete_component(): spheres exceed the component's share.")
    matrix = {}
    for i, comp in enumerate(components):
        w = (share - f_disc) if i == i_disc else comp['VOLUME_SHARE']
        for ingredient, fraction in _resolve_component(comp).items():
            matrix[ingredient] = matrix.get(ingredient, 0.0) + w / (1.0 - f_disc) * fraction

    # Remove the original components list and replace it with the updated ingredients fractions and inclusions information.
    sut.pop('COMPONENTS')
    sut['INGREDIENTS_FRACTIONS'] = matrix
    sut['INCLUSIONS'] = {
        'SHAPE': 'SPHERE',
        'DIAMETER': diameter,
        'CENTERS': [[float(x), float(y), float(z)] for x, y, z in centers],
        'INGREDIENTS_FRACTIONS': _resolve_component(components[i_disc]),
        'INFO': {'NAME': components[i_disc].get('NAME', ''),
                 'n_target': n_target, 'n_placed': placed,
                 'share_target': share, 'share_in_spheres': f_disc,
                 'share_in_matrix_as_crumbs': share - f_disc,
                 'min_gap_m': gap, 'seed': seed, 'fo_clear_mm': fo_clear_mm},}


def _build_parms_for_cookie_run(base_parms: dict, 
                                cookie_overlay: dict,
                                fo_size_mm: float, 
                                fo_material: str,
                                baseline: bool = False,
                                homogenize_layers: bool = False,
                                fo_clear_mm: float = None,
                                fo_position_fraction: list = None) -> dict:
    """This function prepares the full set of parameters for a single simulation run in the cookie study, including the SUT configuration, foreign object placement, and any necessary adjustments for baseline or homogenized layers.

    Args:
        base_parms (dict): 
            Parms of PARMS_FILE (already loaded).
        cookie_overlay (dict): 
            Loaded parameters/cookie_types/*.json.
        fo_size_mm (float): 
            Cube edge length in mm.
        fo_material (str): 
            Key of consts.json MATERIAL_PROPERTIES (e.g. 'PVC').
        baseline (bool): 
            Build the no-anomaly reference run.
        homogenize_layers (bool): 
            Flatten a 'LAYERS' SUT into one mixture via _homogenize_layers() (comparison only; default False).
        fo_clear_mm (float): 
            Edge in mm of the keep-out cube around the SUT center used when a DISCRETE component is placed (the largest foreign object of the study, so the chip layout is identical for baseline and every size). Default: fo_size_mm.
        fo_position_fraction (list): 
            [x, y, z] POSITION_RELATIVE fractions (0..1 within the SUT's bounding box); default None = [0.5, 0.5, 0.5] (center).

    Returns:
        dict: Parms ready for EMSimulation(parms=..., consts=...).

    Raises:
        ValueError: If required keys are missing or invalid in the input dictionaries.
    """
    # Get parameters and cookie overlay
    parms = copy.deepcopy(base_parms)
    sut = copy.deepcopy(cookie_overlay['SUT'])

    # Resolve the discrete component  within the SUT. Homogenize layers if requested.
    _resolve_discrete_component(sut, fo_clear_mm if fo_clear_mm else fo_size_mm)
    for block in ((sut['LAYERS'] if 'LAYERS' in sut else [sut]) + list(sut.get('FREEFORM', {}).get('REGIONS', [])) + [sut['FREEFORM'][k] for k in ('SKIN', 'RIM', 'FO_SHELL') if sut.get('FREEFORM', {}).get(k)]):
        _resolve_fraction_basis(block)
    if homogenize_layers and 'LAYERS' in sut:
        sut = _homogenize_layers(sut)
    parms['SUT'] = sut
    for key in ('L_X_GRID', 'L_Y_GRID', 'L_Z_GRID'):
        if key in cookie_overlay.get('GRID', {}):
            parms[key] = cookie_overlay['GRID'][key]

    # Set up the foreign object parameters in the simulation dictionary. 
    d_m = fo_size_mm / 1000.0
    fo = parms['FOREIGN_OBJECT']
    fo['SIZE'] = [d_m, d_m, d_m]
    fo['POSITION_RELATIVE'] = (list(fo_position_fraction) if fo_position_fraction is not None else [0.5, 0.5, 0.5])  # default: center
    fo['MATERIAL'] = fo_material
    fo['MODE'] = 'SUT' if baseline else 'MATERIAL'
    
    return parms


def _cookie_position_fractions(sut: dict, fo_size_mm: float, n_positions: int,
                               axis_idx: int, edge_clearance_mm: float) -> list:
    """This function calculates the relative position fractions for placing the foreign object along the specified axis, ensuring it stays within the SUT's footprint while maintaining the required edge clearance.

    Args:
        sut (dict): 
            Cookie overlay's SUT block (only 'SHAPE'/'SIZE' are used).
        fo_size_mm (float): 
            Cube edge length in mm.
        n_positions (int): 
            Number of fractions to return (>= 2).
        axis_idx (int): 
            0 (x) or 1 (y); ignored for a CYLINDER SUT.
        edge_clearance_mm (float): 
            Minimum clearance in mm at the farthest position.

    Returns:
        list[float]: n_positions evenly spaced fractions in [0.5, edge_fraction] (index 0 = center, index -1 = edge).

    Raises:
        ValueError: If n_positions < 2.
    """
    # Convert FO size and edge clearance from mm to meters. 
    s = fo_size_mm / 2000.0
    clear = edge_clearance_mm / 1000.0
    size = sut['SIZE']

    # Determine the maximum allowable offset for the foreign object along the specified axis if the SUT is either a CYLINDER.
    if sut['SHAPE'] == 'CYLINDER':
        radius = size['DIAMETER'] / 2
        extent = size['DIAMETER']
        budget = (radius - clear) ** 2 - s ** 2
        offset_max = max(np.sqrt(budget) - s, 0.0) if budget > 0 else 0.0
    else:
        extent = size[axis_idx]
        offset_max = max(extent / 2 - s - clear, 0.0)

    # Convert the maximum offset into a relative fraction of the SUT's extent.
    edge_fraction = 0.5 + offset_max / extent
    
    return [float(v) for v in np.linspace(0.5, edge_fraction, n_positions)]


def _cookie_footprint_extent_mm(sut: dict, axis_idx: int) -> float:
    """This function returns the footprint extent of the cookie along the specified axis (0 = x, 1 = y) in mm.

    Args:
        sut (dict): 
            Cookie overlay's SUT block (only 'SHAPE'/'SIZE' are used).
        axis_idx (int): 
            0 (x) or 1 (y); ignored for a CYLINDER SUT.
    
    Returns:
        float: Full footprint extent along the specified axis in mm.

    Raises:
        None
    """
    size = sut['SIZE']
    if sut['SHAPE'] == 'CYLINDER':
        return 1000.0 * size['DIAMETER']
    
    return 1000.0 * size[axis_idx]


def _smallest_detectable_size(sizes_mm, values, floor: float,
                              rule: str = 'conservative') -> tuple:
    """This function walks down from the largest tested size to find the smallest size whose response exceeds the specified floor. Isolated bumps below the first crossing are ignored. The crossing is interpolated linearly in log(size)-log(response).

    Args:
        sizes_mm (array-like): 
            Tested sizes in mm, any order.
        values (array-like): 
            Response (e.g. rms_dphase_max) per size.
        floor (float): 
            Detection threshold in the response's unit.
        rule (str): 
            'conservative' (default): the largest size tested must clear the floor. 'envelope': the walk starts at the largest size that clears the floor, wherever that is in the sweep.

    Returns:
        tuple: (mds_mm, status). status: 'interpolated' (crossing found), 'below_range' (still detectable at the smallest size; mds_mm is that size) or 'not_detected' (even the largest size is below the floor; mds_mm is NaN).

    Raises:
        None
    """
    # Sort sizes in descending order and align floor values accordingly.
    order = np.argsort(np.asarray(sizes_mm, dtype=float))[::-1]
    s = np.asarray(sizes_mm, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    
    # Identify which response values are above the detection floor.
    above = v > floor
    if rule not in ('conservative', 'envelope'):
        raise ValueError(f"_smallest_detectable_size(): unknown rule {rule!r} (use 'conservative' or 'envelope').")
    
    # Determine if any size is detectable and if the largest size meets the rule.
    if len(s) == 0 or not above.any() or (rule == 'conservative' and not above[0]):
        return float('nan'), 'not_detected'

    # Find the last size that is above the detection floor according to the rule (envelope).
    k = int(np.argmax(above)) if rule == 'envelope' else 0
    while k + 1 < len(s) and above[k + 1]:
        k += 1
    if k == len(s) - 1:
        return float(s[-1]), 'below_range'
    
    # Interpolate logarithmically between the last detectable size and the next one below the floor. Compute the interpolation factor between the last detectable size and the next one below the floor.
    x0, x1 = np.log(s[k]), np.log(s[k + 1])
    y0, y1 = np.log(v[k]), np.log(max(v[k + 1], 1e-12))
    t = (np.log(floor) - y0) / (y1 - y0)
    
    return float(np.exp(x0 + t * (x1 - x0))), 'interpolated'


def _max_detectable_offset_mm(offsets_mm, values, floor: float,
                              rule: str = 'conservative') -> tuple:
    """This function finds the largest offset from the SUT center in  mm whose response still exceeds the detection floor (crossing is linearly interpolated in offset and logarithmically).

    Args:
        offsets_mm (array-like): 
            Tested offsets from center in mm, any order, may include 0.
        values (array-like): 
            Response (e.g. rms_dphase_max) per offset.
        floor (float): 
            Detection threshold in the response's unit.
        rule (str): 
            'conservative' (default): the offset closest to center must clear the floor. 'envelope': the walk starts at the offset closest to center that clears the floor.

    Returns:
        tuple: (max_offset_mm, status). status: 'interpolated' (crossing found), 'beyond_range' (still detectable at the largest offset; true limit >= it) or 'not_detected' (the center, or every offset under 'envelope', is below the floor).

    Raises:
        ValueError: If an unknown rule is provided.
    """
    # Sort offsets and corresponding values in ascending order of offsets.
    order = np.argsort(np.asarray(offsets_mm, dtype=float))
    o = np.asarray(offsets_mm, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    
    # Identify which offsets have responses above the detection floor.
    above = v > floor
    if rule not in ('conservative', 'envelope'):
        raise ValueError(f"_max_detectable_offset_mm(): unknown rule {rule!r} "
                         f"(use 'conservative' or 'envelope').")
    
    # Determine if any offset meets the detection criterion based on the rule (conservative or envelope).
    if len(o) == 0 or not above.any() or (rule == 'conservative' and not above[0]):
        return float('nan'), 'not_detected'
    
    # Find the last offset that is above the detection floor, starting from the first above-floor offset according to the rule.
    k = int(np.argmax(above)) if rule == 'envelope' else 0
    while k + 1 < len(o) and above[k + 1]:
        k += 1
    if k == len(o) - 1:
        return float(o[-1]), 'beyond_range'

    # Interpolate to find the offset where the response crosses the detection floor.
    x0, x1 = o[k], o[k + 1]
    y0, y1 = np.log(max(v[k], 1e-12)), np.log(max(v[k + 1], 1e-12))
    t = (np.log(floor) - y0) / (y1 - y0)
    
    return float(x0 + t * (x1 - x0)), 'interpolated'


def _floor_for_metric(response_metric: str, base_parms: dict):
    """This function returns the detection floor for a given response metric based on the base parameters.

    Args:
        response_metric (str): 
            The response metric for which to determine the detection floor.
        base_parms (dict): 
            Dictionary containing base parameters, including 'AMP_NOISE_FLOOR_DB' and 'PHASE_NOISE_FLOOR_DEG'.

    Returns:
        float or None: The detection floor for the given response metric, or None if the metric has no defined floor.

    Raises:
        KeyError: If the required base parameter for the response metric is not found in base_parms.
    """
    if response_metric.startswith('rms_dphase') or response_metric.startswith('mean_dphase'):
        return base_parms.get('PHASE_NOISE_FLOOR_DEG')
    if response_metric.startswith('rms_dmag') or response_metric.startswith('mean_dmag'):
        return base_parms.get('AMP_NOISE_FLOOR_DB')
    
    return None


def _crb_floor_for_metric(response_metric: str, snr_db: float):
    """This function calculates the detection floor for a given response metric based on the assumed per-bin SNR from the Cramer-Rao bound (sigma = 1/sqrt(2*SNR)).

    Args:
        response_metric (str): 
            The response metric for which to determine the detection floor.
        snr_db (float): 
            The assumed per-bin SNR in decibels.

    Returns:
        float or None: The detection floor for the given response metric, or None if the metric has no defined floor.
    
    Raises:
        ValueError: If the response metric is not recognized.
    """
    sigma = 1.0 / np.sqrt(2.0 * 10.0 ** (snr_db / 10.0))
    if response_metric.startswith(('rms_dphase', 'mean_dphase')):
        return float(np.degrees(sigma))
    if response_metric.startswith(('rms_dmag', 'mean_dmag')):
        return float(20.0 / np.log(10.0) * sigma)
    
    return None


def _cookie_score_mds(results: 'pd.DataFrame', 
                      response_metrics: list, 
                      floors: dict,
                      rule: str = 'conservative') -> 'pd.DataFrame':
    """This function calculates the smallest detectable foreign-object size per product based on the response metrics and their detection floors.
    
    Args:
        results (pd.DataFrame): 
            The DataFrame containing the measurement results.
        response_metrics (list): 
            The list of response metrics to consider.
        floors (dict): 
            A dictionary mapping each response metric to its detection floor.
        rule (str, optional): 
            The rule to use for determining the smallest detectable size. Defaults to 'conservative'.

    Returns:
        pd.DataFrame: A DataFrame containing the smallest detectable size and status for each cookie and response metric, as well as the combined result.

    Raises:
        ValueError: If any of the response metrics are not recognized.
    """
    # Iterate over each cookie (product) in the results DataFrame.
    mds_rows = []
    for name, sub in results.groupby('cookie', sort=False):
        
        row = dict(cookie=name, thickness_mm=sub['thickness_mm'].iloc[0])
        ratio = np.zeros(len(sub))

        # Iterate over each response metric and calculate the smallest detectable size.
        for m in response_metrics:
            if floors[m] is None:
                row[f'mds_mm_{m}'], row[f'status_{m}'] = float('nan'), 'no_floor_for_metric'
                continue
            (row[f'mds_mm_{m}'], 
             row[f'status_{m}']) = _smallest_detectable_size(sub['fo_size_mm'], 
                                                             sub[m], 
                                                             floors[m], 
                                                             rule)
            ratio = np.maximum(ratio, sub[m].to_numpy() / floors[m])

        # Calculate the combined smallest detectable size across all response metrics.
        (row['mds_mm_combined'], 
         row['status_combined']) = _smallest_detectable_size(sub['fo_size_mm'], 
                                                             ratio, 1.0, 
                                                             rule)
        mds_rows.append(row)

    return pd.DataFrame(mds_rows)


def _cookie_score_position_range(results: 'pd.DataFrame', 
                                 response_metrics: list,
                                 floors: dict, 
                                 rule: str = 'conservative') -> 'pd.DataFrame':
    """This function calculates the largest detectable FO-position offset for each product and response metric, as well as a combined metric.

    Args:
        results (pd.DataFrame): 
            DataFrame containing the test results.
        response_metrics (list): 
            List of response metrics to consider.
        floors (dict): 
            Dictionary mapping each response metric to its threshold floor.
        rule (str, optional): 
            Rule to apply in the calculation. Defaults to 'conservative'.

    Returns:
        pd.DataFrame: DataFrame with the largest detectable FO-position offset for each product and response metric, as well as a combined metric.

    Raises:
        ValueError: If a requested response metric is not found in the results DataFrame.
    """
    rows = []
    # Iterate over each cookie (product) in the results DataFrame.
    for name, sub in results.groupby('cookie', sort=False):

        row = dict(cookie=name, 
                   thickness_mm=sub['thickness_mm'].iloc[0],
                   fo_size_mm=sub['fo_size_mm'].iloc[0],
                   max_offset_tested_mm=float(sub['offset_mm'].max()))
        ratio = np.zeros(len(sub))

        # Iterate over each response metric and calculate the maximum detectable offset for each.
        for m in response_metrics:
            if floors[m] is None:
                row[f'max_offset_mm_{m}'], row[f'status_{m}'] = float('nan'), 'no_floor_for_metric'
                continue
            row[f'max_offset_mm_{m}'], row[f'status_{m}'] = _max_detectable_offset_mm(
                sub['offset_mm'], sub[m], floors[m], rule)
            ratio = np.maximum(ratio, sub[m].to_numpy() / floors[m])

        # Calculate the combined maximum detectable offset using the ratio of each response metric to its floor.
        (row['max_offset_mm_combined'], 
         row['status_combined']) = _max_detectable_offset_mm(sub['offset_mm'], 
                                                             ratio, 
                                                             1.0, 
                                                             rule)
        rows.append(row)
    
    return pd.DataFrame(rows)


def _cookie_required_result_files(base_parms: dict) -> list:
    """This function determines the required result files for a given run folder based on the presence of the ALSO_MIRRORED_RX flag in the base parameters.

    Args:
        base_parms (dict):
            Base parameters of the run folder.
    
    Returns:
        list: Required result files for the run folder.

    Raises:
        None
    """
    
    return (['data_reflection.pkl', 'data_transmission.pkl'] if base_parms.get('ALSO_MIRRORED_RX', False) else ['data.pkl'])


def _cookie_sides_in_run_folder(folder) -> list:
    """This function checks the run folder for the presence of receiver side data files and returns the corresponding sides.
    
    Args:
        folder (str or Path): 
            Path to the run folder.

    Returns:
        list: Receiver sides stored in the run folder.

    Raises:
        None
    """
    folder = Path(folder)
    found = [sd for sd in ('reflection', 'transmission') if (folder / f'data_{sd}.pkl').is_file()]
    
    return found or (['legacy'] if (folder / 'data.pkl').is_file() else [])


def _cookie_select_rx_sides(study_dir, baseline_folder, rx_sides, caller: str) -> list:
    """This function determines which receiver sides should be scored based on the detected sides in the baseline run folder and the requested sides.
    
    Args:
        study_dir (str or Path): 
            Path to the study directory.
        baseline_folder (str or Path): 
            Baseline run folder within the study directory.
        rx_sides (list or None): 
            Requested receiver sides to score. If None, all detected sides are used.
        caller (str): 
            Name of the calling function for error messages.

    Returns:
        list: Receiver sides to score.

    Raises:
        FileNotFoundError: If a requested receiver side is not found in the baseline run folder.
    """
    # Determine the receiver sides present in the baseline run folder.
    detected = _cookie_sides_in_run_folder(_resolve_cookie_run_folder(study_dir, baseline_folder))

    # Check that all requested sides are present in the detected sides.
    sides = list(detected if rx_sides is None else rx_sides)
    for sd in sides:
        if sd not in detected:
            raise FileNotFoundError(f"{caller}: rx_side {sd!r} requested but the baseline run folder only holds {detected}.")
    
    return sides


def _cookie_describe_product(overlay: dict, chip_layout_csv, sizes_tested: list) -> dict:
    """This function describes the product geometry, composition, and tested cube sizes based on the provided overlay and chip layout.

    Args:
        overlay (dict): 
            Inclusions
        chip_layout_csv (str or Path): 
            CSV file of chip layout of a product with DISCRETE spheres (sphere count); missing = no count.
        sizes_tested (list): 
            Cube edge lengths in mm run for this product.

    Returns:
        dict: product, geometry, thickness_mm, composition, air_volume_fraction, water_mass_pct, n_cube_sizes, cube_sizes_mm.

    Raises:
        None
    """
    # If the product is a cylinder, describe its geometry accordingly. Otherwise, treat it as a cuboid.
    sut, size = overlay['SUT'], overlay['SUT']['SIZE']
    if sut['SHAPE'] == 'CYLINDER':

        geometry = f"cylinder, D {1e3 * size['DIAMETER']:g} x {1e3 * size['THICKNESS']:g} mm"
        thickness, volume = size['THICKNESS'], np.pi / 4 * size['DIAMETER'] ** 2 * size['THICKNESS']

    else:

        geometry = f"cuboid, {1e3 * size[0]:g} x {1e3 * size[1]:g} x {1e3 * size[2]:g} mm"
        thickness, volume = size[2], size[0] * size[1] * size[2]

    def _water(part):
        """This helper function returns the water mass percentage for a given part.
        Args:
            part (dict): 
                A dictionary representing a part of the product, containing ingredient fractions.
        
        Returns:
            str: Water mass percentage as a string.

        Raises:
            None
        """
        return f"{100 * part['INGREDIENTS_FRACTIONS']['WATER']:g}"

    # Determine the composition, air volume fraction, and water mass percentage based on the structure of the SUT.
    if 'LAYERS' in sut:

        layers = sut['LAYERS']
        composition = 'layers from TX side: ' + ', '.join(f"{1e3 * l['THICKNESS']:g} mm {l['NAME']}" for l in layers)
        air = ' / '.join(f"{l.get('AIR_VOLUME_FRACTION', 0):g}" for l in layers)
        water = ' / '.join(_water(l) for l in layers)

    elif 'COMPONENTS' in sut:
        
        parts, texts = sut['COMPONENTS'], []
        for c in parts:
            
            text = f"{c['NAME']} {100 * c['VOLUME_SHARE']:g} vol% (air {c.get('AIR_VOLUME_FRACTION', 0):g})"
            disc = c.get('DISCRETE')
            
            if disc is not None and Path(chip_layout_csv).is_file():
                n = len(pd.read_csv(chip_layout_csv))
                share = n * np.pi / 6 * disc['DIAMETER'] ** 3 / volume
                text += (f": {n} {disc['SHAPE'].lower()}s (D {1e3 * disc['DIAMETER']:g} mm, {100 * share:.1f} vol%) and crumbs ({100 * (c['VOLUME_SHARE'] - share):.1f} vol%)")
            texts.append(text)
        
        composition = '; '.join(texts)
        air = ' / '.join(f"{c.get('AIR_VOLUME_FRACTION', 0):g}" for c in parts)
        water = ' / '.join(_water(c) for c in parts)

    else:

        composition = 'homogeneous'
        air, water = f"{sut.get('AIR_VOLUME_FRACTION', 0):g}", _water(sut)
    sizes = sorted(sizes_tested, reverse=True)
    
    return dict(product=_cookie_short_name(overlay['COOKIE_NAME']), 
                geometry=geometry, 
                thickness_mm=1e3 * thickness,
                composition=composition, 
                air_volume_fraction=air, 
                water_mass_pct=water, 
                n_cube_sizes=len(sizes),
                cube_sizes_mm=' '.join(f'{s:g}' for s in sizes))


def _export_cookie_size_paper_tables(mds_all: 'pd.DataFrame', 
                                     ratio_all: 'pd.DataFrame', 
                                     sens_df: 'pd.DataFrame', 
                                     sides: list,
                                     manifest: 'pd.DataFrame', cookie_parms_files: list, 
                                     out_dir) -> None:
    """This function generates and exports tables related to cookie detectability as a function of cookie size.

    Args:
        mds_all (pd.DataFrame): 
            Per side and product: thickness_mm, mds_mm_<metric>, status_<metric> (incl. 'combined').
        ratio_all (pd.DataFrame): 
            Per side, product and size: combined_over_floor.
        sens_df (pd.DataFrame): 
            Per side, SNR and product: floor_<metric>, mds_mm_combined, status_combined (empty = no SNR tables).
        sides (list): 
            Receiver sides scored.
        manifest (pd.DataFrame): 
            Size-study manifest (cookie, fo_size_mm).
        cookie_parms_files (list): 
            Product overlay files for cookie_detectability_fo_size_products.csv; None skips that table.
        out_dir (str or Path): 
            Destination folder (results_analysis).

    Returns:
        None

    Raises:
        None
    """
    # 
    out_dir = Path(out_dir)

    # After ensuring that output directory exists, extract metric keys from mds_all columns, excluding 'combined' initially, then append 'combined' at the end.
    keys = [c[len('mds_mm_'):] for c in mds_all.columns if c.startswith('mds_mm_')]
    keys = [k for k in keys if k != 'combined'] + ['combined']
    label = {k: 'combined' if k == 'combined' else _cookie_metric_short_name(k) for k in keys}

    # Process each receiver side separately.
    for sd in sides:

        # Determine the file suffix based on the receiver side.
        suffix = '' if sd == 'legacy' else f'_{sd}'
        m = mds_all[mds_all['rx_side'] == sd].reset_index(drop=True)
        if m.empty:
            continue

        # Create a DataFrame with product names and thickness, then add MDS and status columns for each metric.
        t = pd.DataFrame({'product': m['cookie'].map(_cookie_short_name), 'thickness_mm': m['thickness_mm']})
        for k in keys:
            t[f'mds_{label[k]}_mm'] = m[f'mds_mm_{k}'].round(2)
        t['mds_over_thickness'] = (m['mds_mm_combined'] / m['thickness_mm']).round(3)

        # Add status columns for each metric and save the DataFrame to a CSV file.
        for k in keys:
            t[f'status_{label[k]}'] = m[f'status_{k}']
        t.sort_values('product').to_csv(out_dir / f'cookie_detectability_fo_size_mds{suffix}.csv', index=False)

        # Process the ratio data for the current receiver side and save it to a CSV file.
        r = ratio_all[ratio_all['rx_side'] == sd].assign(product=lambda d: d['cookie'].map(_cookie_short_name))
        
        # Pivot the ratio data to create a wide-format table with 'fo_size_mm' as the index and products as columns.
        wide = r.pivot(index='fo_size_mm', 
                       columns='product', 
                       values='combined_over_floor').sort_index(ascending=False).round(3)
        wide.index.name = 'edge_mm'
        wide.sort_index(axis=1).to_csv(out_dir / f'cookie_detectability_fo_size_ratio{suffix}.csv')

        # Process the sensitivity data for the current receiver side and save it to a CSV file.
        if len(sens_df):

            # Filter the sensitivity DataFrame for the current receiver side.
            s = sens_df[sens_df['rx_side'] == sd]
            snrs = sorted(s['snr_db'].unique())
            t = pd.DataFrame({'product': m['cookie'].map(_cookie_short_name)})

            # Add MDS and status columns for each SNR value.
            for snr in snrs:
                ss = s[s['snr_db'] == snr].set_index('cookie')
                t[f'mds_{snr:g}dB_mm'] = ss['mds_mm_combined'].reindex(m['cookie']).round(2).to_numpy()
                t[f'status_{snr:g}dB'] = ss['status_combined'].reindex(m['cookie']).to_numpy()
            
            t.sort_values('product').to_csv(out_dir / f'cookie_detectability_fo_size_snr{suffix}.csv', index=False)

    # Export SNR thresholds for cookie detectability based on FO size.
    if len(sens_df):
        fl = [c for c in sens_df.columns if c.startswith('floor_')]
        sens_df[['snr_db'] + fl].drop_duplicates('snr_db').sort_values('snr_db').rename(columns={c: c + ('_deg' if c.endswith('phase') else '_db') for c in fl}).to_csv(out_dir / 'cookie_detectability_fo_size_snr_thresholds.csv', index=False)

    # Iterate over each cookie parameter file and extract the product information.
    if cookie_parms_files:

        rows = []
        for f in cookie_parms_files:
            ov = ModelMixin._load_config(f)
            sizes = [float(x) for x in manifest.loc[(manifest['cookie'] == ov['COOKIE_NAME']) & (manifest['fo_size_mm'] != 'base'), 'fo_size_mm']]
            
            slug = re.sub(r'[^A-Za-z0-9]+', '_', ov['COOKIE_NAME']).strip('_').lower()
            rows.append(_cookie_describe_product(ov, out_dir / f'chip_layout_{slug}.csv', sizes))
        
        pd.DataFrame(rows).sort_values('product').to_csv(out_dir / 'cookie_detectability_fo_size_products.csv', index=False)

    # Prepare and export the MDS comparison table for cookie detectability based on FO size.
    if {'reflection', 'transmission'} <= set(sides):
        
        a, b = (mds_all[mds_all['rx_side'] == sd].set_index('cookie') for sd in ('reflection', 'transmission'))
        t = pd.DataFrame(
            {'product': [_cookie_short_name(c) for c in a.index], 
            'thickness_mm': a['thickness_mm'].to_numpy(),
            'mds_reflection_mm': a['mds_mm_combined'].round(2).to_numpy(),
            'mds_transmission_mm': b['mds_mm_combined'].reindex(a.index).round(2).to_numpy()})
        t['factor'] = (t['mds_transmission_mm'] / t['mds_reflection_mm']).round(2)
        t['status_reflection'] = a['status_combined'].to_numpy()
        t['status_transmission'] = b['status_combined'].reindex(a.index).to_numpy()
        
        # Export the MDS comparison table for cookie detectability based on FO size.
        t.sort_values('product').to_csv(out_dir / 'cookie_detectability_fo_size_mds_comparison.csv', index=False)
        _plot_cookie_detectability_comparison_journal(mds_all, sens_df, out_dir / f'cookie_detectability_fo_size_comparison.png')


def _cookie_offset_label(values) -> str:
    """
    This function returns a label (string) representing the nominal offset label for a given set of position step values.

    Args:
        values (iterable): 
            A collection of numeric position step values.

    Returns:
        str: A string representing the nominal offset label. If all values are the same, returns that value as a string. Otherwise, returns a range in the format 'min-max'.

    Raises:
        None
    """
    a, b = f'{min(values):.1f}', f'{max(values):.1f}'
    return a if a == b else f'{a}-{b}'


def _cookie_position_combined_ratio(df: 'pd.DataFrame', response_metrics: list, floors: dict) -> 'np.ndarray':
    """This function calculates the combined response-to-threshold ratio for each row in the DataFrame, considering only the metrics that have a defined floor value. It selects the largest ratio among the applicable metrics for each row.

    Args:
        df (pd.DataFrame): 
            The input DataFrame containing response metrics for each row.
        response_metrics (list): 
            A list of response metric column names to consider.
        floors (dict): 
            A dictionary mapping response metric names to their respective floor values.

    Returns:
        np.ndarray: An array containing the combined response-to-threshold ratio for each row.

    Raises:
        None
    """
    r = np.zeros(len(df))
    for m in response_metrics:
        if floors.get(m) is not None:
            r = np.maximum(r, df[m].to_numpy(dtype=float) / floors[m])
    
    return r


def _export_cookie_position_paper_tables(results_all: 'pd.DataFrame', 
                                         range_all: 'pd.DataFrame', 
                                         sens_df: 'pd.DataFrame', 
                                         sides: list,
                                         response_metrics: list, 
                                         floors: dict, 
                                         snr_db_sweep: tuple, out_dir) -> None:
    """This function exports various tables and a comparison figure related to cookie detectability at different FO positions into the specified output directory.

    Args:
        results_all (pd.DataFrame): 
            Per side, product and position: rx_side, cookie, position_index, offset_mm, one column per response metric.
        range_all (pd.DataFrame): 
            Per side and product: max_offset_tested_mm, max_offset_mm_<metric|combined>, status_<...>.
        sens_df (pd.DataFrame): 
            Per side, SNR and product: max_offset_mm_<...>, status_<...> (empty = no SNR tables).
        sides (list): 
            Receiver sides scored ('reflection', 'transmission' or 'legacy').
        response_metrics (list): 
            Metric columns (phase first, as in the paper).
        floors (dict): 
            Metric -> threshold of the nominal case (1.3 deg / 0.2 dB).
        snr_db_sweep (tuple): 
            Assumed per-bin SNRs in dB for the coverage tables.
        out_dir (str or Path): 
            Destination folder (results_analysis).
    """
    out_dir = Path(out_dir)

    # Prepare the results DataFrame and compute the combined response-to-threshold ratio.
    res = results_all.copy()
    res['position_index'] = pd.to_numeric(res['position_index'])
    res['ratio'] = _cookie_position_combined_ratio(res, 
                                                   response_metrics, 
                                                   floors)
    res['product'] = res['cookie'].map(_cookie_short_name)
    prods = sorted(dict.fromkeys(res['product']))
    keys = list(response_metrics) + ['combined']
    label = {k: 'combined' if k == 'combined' else _cookie_metric_short_name(k) for k in keys}

    # Prepare the position offset table and save it to a CSV file.
    off = res.pivot_table(index='position_index', columns='product', values='offset_mm', aggfunc='first')[prods].round(2)
    off.insert(0, 'offset_label', [_cookie_offset_label(off.loc[i].to_numpy(dtype=float)) for i in off.index])
    off.to_csv(out_dir / 'cookie_detectability_fo_position_offsets.csv')

    # Iterate over each receiver side and generate the corresponding range and ratio tables.
    for sd in sides:

        suffix = '' if sd == 'legacy' else f'_{sd}'
        r = range_all[range_all['rx_side'] == sd]
        if r.empty:
            continue

        # Create a data frame for the range table.
        t = pd.DataFrame(
        {'product': r['cookie'].map(_cookie_short_name).to_numpy(), 
        'max_offset_tested_mm': r['max_offset_tested_mm'].round(1).to_numpy()})

        # Add the maximum offset and status columns for each response metric.
        for k in keys:
            t[f'max_offset_{label[k]}_mm'] = r[f'max_offset_mm_{k}'].round(1).to_numpy()
        for k in keys:
            t[f'status_{label[k]}'] = r[f'status_{k}'].to_numpy()
        t.sort_values('product').to_csv(out_dir / f'cookie_detectability_fo_position_range{suffix}.csv', index=False)

        # Create the ratio table for the current receiver side.
        sub = res[res['rx_side'] == sd]
        wide = sub.pivot_table(index='position_index', 
                               columns='product', values='ratio', aggfunc='first')[[p for p in prods if p in set(sub['product'])]].round(3)
        
        # Prepare the output table by adding the offset label and summary rows. Then save it to a CSV file.
        out = wide.copy()
        out.insert(0, 'offset_label', off.loc[wide.index, 'offset_label'])
        out = out.astype(object)
        out.index = out.index.astype(str)
        out.loc['n_above_threshold'] = [''] + [int((wide[p] > 1).sum()) for p in wide.columns]
        out.loc['min_ratio'] = [''] + [float(wide[p].min()) for p in wide.columns]
        out.index.name = 'position_index'
        out.to_csv(out_dir / f'cookie_detectability_fo_position_ratio{suffix}.csv')

        # Create the SNR table for the current receiver side.
        if len(sens_df):

            # Filter the sensitivity DataFrame for the current receiver side.
            s_ = sens_df[sens_df['rx_side'] == sd]
            snrs = sorted(s_['snr_db'].unique())
            t = pd.DataFrame({'product': r['cookie'].map(_cookie_short_name).to_numpy()})

            # Add the maximum offset and status columns for each SNR value.
            for snr in snrs:
                ss = s_[s_['snr_db'] == snr].set_index('cookie')
                t[f'max_offset_{snr:g}dB_mm'] = ss['max_offset_mm_combined'].reindex(r['cookie']).round(1).to_numpy()
                t[f'status_{snr:g}dB'] = ss['status_combined'].reindex(r['cookie']).to_numpy()
            t.sort_values('product').to_csv(out_dir / f'cookie_detectability_fo_position_snr{suffix}.csv', index=False)

    # Iterate over each receiver side and SNR value to compute coverage statistics.
    snrs = list(snr_db_sweep)
    cov_rows, share_rows = {}, []
    for sd in sides:
        sub = res[res['rx_side'] == sd]
        for snr in snrs:

            # Compute the number of positions above the Cramer-Rao threshold for each product.
            fl = {m: _crb_floor_for_metric(m, snr) for m in response_metrics}
            above = _cookie_position_combined_ratio(sub, 
                                                    response_metrics, 
                                                    fl) > 1.0

            # Count the number of positions above the threshold and the total number of positions for each product.
            cnt = pd.Series(above, index=sub.index).groupby(sub['product']).sum()
            npos = pd.Series(1, index=sub.index).groupby(sub['product']).sum()

            # Store the coverage information for each product.
            for p in cnt.index:
                cov_rows.setdefault(p, {})[f'n_above_{sd}_{snr:g}dB'] = int(cnt[p])
                cov_rows[p]['n_positions'] = int(npos[p])
            
            share_rows.append(dict(rx_side=sd, snr_db=snr, share_percent=round(float((cnt / npos).mean() * 100), 1)))
    
    # Convert the coverage and share information and export as CSV files.
    cov = pd.DataFrame.from_dict(cov_rows, orient='index').reindex(prods)
    cov = cov[['n_positions'] + [c for c in cov.columns if c != 'n_positions']]
    cov.index.name = 'product'
    cov.to_csv(out_dir / 'cookie_detectability_fo_position_coverage.csv')
    share = pd.DataFrame(share_rows)
    share.to_csv(out_dir / 'cookie_detectability_fo_position_coverage_share.csv', index=False)

    # Compute the factor between reflection and transmission for the selected phase metric.
    if {'reflection', 'transmission'} <= set(sides):

        # Select the phase metric to use for the comparison.
        phase_m = next((m for m in response_metrics if 'dphase' in m), response_metrics[0])

        # Extract the phase metric values for reflection and transmission sides.
        a = res[res['rx_side'] == 'reflection'].set_index(['product', 'position_index'])[phase_m]
        b = res[res['rx_side'] == 'transmission'].set_index(['product', 'position_index'])[phase_m]
        q = (a / b.reindex(a.index)).rename('factor').reset_index()
        
        # Compute the factor for each product and store the center, edge, and median values.
        rows = []
        for p, g in q.groupby('product'):
            g = g.sort_values('position_index')
            rows.append(dict(product=p, 
                             center=round(float(g['factor'].iloc[0]), 2), edge=round(float(g['factor'].iloc[-1]), 2),
                             median=round(float(g['factor'].median()), 2)))

        # Sort and deduplicate the products before exporting the computed factors as a CSV file.
        pd.DataFrame(rows).sort_values('product').to_csv(out_dir / 'cookie_detectability_fo_position_factor.csv', index=False)
        thickness = res.drop_duplicates('product').set_index('product')['thickness_mm']
        _plot_cookie_position_comparison_journal(res, cov, share, thickness, snrs, out_dir / f'cookie_detectability_fo_position_comparison.png')


def _export_cookie_frequency_paper_tables(results_all: 'pd.DataFrame', 
                                          sides: list, 
                                          response_metrics: list, 
                                          floors: dict,
                                          snr_db_sweep: tuple, 
                                          out_dir) -> None:
    """This function exports the frequency subsection's tables and comparison figure for the given results, sides, response metrics, floors, and SNR sweep.

    Args:
        results_all (pd.DataFrame): 
            Per side, product and band: rx_side, cookie, band, f_center_ghz, fo_size_over_lambda0, thickness_mm,
            one column per response metric.
        sides (list): 
            Receiver sides scored.
        response_metrics (list): 
            Metric columns (phase first, as in the paper).
        floors (dict): 
            Metric -> nominal threshold (1.3 deg / 0.2 dB).
        snr_db_sweep (tuple): 
            Assumed per-bin SNRs in dB for the coverage tables.
        out_dir (str or Path): 
            Destination folder.
    """
    out_dir = Path(out_dir)
    
    # Prepare the results DataFrame with additional computed columns (ratio, product).
    res = results_all.copy()
    res['ratio'] = _cookie_position_combined_ratio(res, response_metrics, floors)
    res['product'] = res['cookie'].map(_cookie_short_name)
    
    # Determine the unique products and bands for later use.
    prods = sorted(dict.fromkeys(res['product']))
    bands = res.drop_duplicates('band').sort_values('f_center_ghz')[['band', 'f_center_ghz', 'fo_size_over_lambda0']]
    band_order = list(bands['band'])
    metric_label = {m: ('phase_deg' if 'dphase' in m else 'amplitude_db' if 'dmag' in m else m) for m in response_metrics}

    # Iterate over receiver sides and generate frequency response and ratio tables.
    for sd in sides:

        # Skip if there is no data for this receiver side.
        suffix = '' if sd == 'legacy' else f'_{sd}'
        sub = res[res['rx_side'] == sd]
        if sub.empty:
            continue

        # Filter the products that have data for this receiver side.
        have = [p for p in prods if p in set(sub['product'])]
        t = bands.set_index('band').copy()
        
        # Generate frequency response tables for each response metric.
        for m in response_metrics:
            w = sub.pivot_table(index='band', 
                                columns='product', 
                                values=m, 
                                aggfunc='first')
            for p in have:
                t[f'{metric_label[m]}_{p}'] = w[p].reindex(t.index).round(3)
        t.reindex(band_order).to_csv(out_dir / f'cookie_detectability_frequency_response{suffix}.csv')

        # Generate frequency ratio tables for each receiver side.
        wide = sub.pivot_table(index='band', 
                               columns='product', 
                               values='ratio', 
                               aggfunc='first')[have].reindex(band_order).round(3)
        
        # Add summary statistics for the frequency ratio table and export it to CSV.
        out = wide.copy()
        out.insert(0, 'f_center_ghz', 
                   bands.set_index('band').loc[wide.index, 'f_center_ghz'].to_numpy())
        out = out.astype(object)
        out.loc['n_above_threshold'] = [''] + [int((wide[p] > 1).sum()) for p in have]
        out.loc['min_ratio'] = [''] + [float(wide[p].min()) for p in have]
        out.index.name = 'band_ghz'
        out.to_csv(out_dir / f'cookie_detectability_frequency_ratio{suffix}.csv')

    # Compute coverage and share statistics for each receiver side and SNR.
    snrs = list(snr_db_sweep)
    cov_rows, share_rows = {}, []
    for sd in sides:
        
        # Filter the results for the current receiver side.
        sub = res[res['rx_side'] == sd]
        
        # Compute coverage statistics for each SNR.
        for snr in snrs:

            # Determine the CRB floor for each response metric at the current SNR.
            fl = {m: _crb_floor_for_metric(m, snr) for m in response_metrics}
            above = _cookie_position_combined_ratio(sub, response_metrics, fl) > 1.0
            cnt = pd.Series(above, index=sub.index).groupby(sub['product']).sum()
            nb = pd.Series(1, index=sub.index).groupby(sub['product']).sum()
            
            for p in cnt.index:
                cov_rows.setdefault(p, {})[f'n_above_{sd}_{snr:g}dB'] = int(cnt[p])
                cov_rows[p]['n_bands'] = int(nb[p])
            share_rows.append(dict(rx_side=sd, snr_db=snr, share_percent=round(float((cnt / nb).mean() * 100), 1)))

    # Convert the coverage and share statistics to DataFrames and export them to CSV.
    cov = pd.DataFrame.from_dict(cov_rows, orient='index').reindex(prods)
    cov = cov[['n_bands'] + [c for c in cov.columns if c != 'n_bands']]
    cov.index.name = 'product'
    cov.to_csv(out_dir / 'cookie_detectability_frequency_coverage.csv')
    
    # Export the share statistics to CSV.
    share = pd.DataFrame(share_rows)
    share.to_csv(out_dir / 'cookie_detectability_frequency_coverage_share.csv', index=False)

    # Compute and export the SNR thresholds for each response metric.
    if snrs:
        th = pd.DataFrame([{'snr_db': snr, **{f'floor_{_cookie_metric_short_name(m)}': round(_crb_floor_for_metric(m, snr), 4) for m in response_metrics if _crb_floor_for_metric(m, snr) is not None}} for snr in snrs])
        th.rename(columns={c: c + ('_deg' if c.endswith('phase') else '_db') for c in th.columns if c.startswith('floor_')}).to_csv(
            out_dir / 'cookie_detectability_frequency_snr_thresholds.csv', index=False)
    
    # Export per-side SNR tables: number of bands above the threshold per product.
    for sd in sides:
        cols = [c for c in cov.columns if c.startswith(f'n_above_{sd}_')]
        if cols:
            cov[['n_bands'] + cols].rename(columns=lambda c: c.replace(f'n_above_{sd}_', 'n_above_')).to_csv(out_dir / f"cookie_detectability_frequency_snr{'' if sd == 'legacy' else '_' + sd}.csv")

    # Compute and export the reflection/transmission phase factor table.
    if {'reflection', 'transmission'} <= set(sides):

        # Select the phase metric and get data to use for the factor calculation.
        phase_m = next((m for m in response_metrics if 'dphase' in m), response_metrics[0])
        a = res[res['rx_side'] == 'reflection'].set_index(['band', 'product'])[phase_m]
        b = res[res['rx_side'] == 'transmission'].set_index(['band', 'product'])[phase_m]

        # Compute the reflection/transmission phase factor for each band and product.
        q = (a / b.reindex(a.index)).rename('factor').reset_index()
        w = q.pivot_table(index='band', columns='product', values='factor', aggfunc='first').reindex(band_order).round(2)
        
        # Insert and export
        w.insert(0, 'f_center_ghz', bands.set_index('band').loc[w.index, 'f_center_ghz'].to_numpy())
        w.index.name = 'band_ghz'
        w.to_csv(out_dir / 'cookie_detectability_frequency_factor.csv')
        thickness = res.drop_duplicates('product').set_index('product')['thickness_mm']
        _plot_cookie_position_comparison_journal(
            res, 
            cov.rename(columns={'n_bands': 'n_positions'}), 
            share, 
            thickness, 
            snrs,
            out_dir / f'cookie_detectability_frequency_comparison.png', x_col='f_center_ghz',
            x_label='Band-center frequency / GHz', 
            title_a='(a) Response vs. frequency', 
            what='bands',
            x_ticks=list(bands['f_center_ghz']), 
            legend_loc='upper left', 
            y_headroom=8.0)


def _count_pending_doe_rows(rows_df: pd.DataFrame, manifest_df: pd.DataFrame,
                             retry_failed: bool) -> int:
    """This function iterates over the rows in 'rows_df' and counts how many would still be attempted by the DOE loop, considering the current status in 'manifest_df' and the `retry_failed` flag.
    
    Args:
        rows_df (pd.DataFrame): 
            Slice of the DOE table with 'run' column.
        manifest_df (pd.DataFrame): 
            Current manifest with 'status' column indexed by 'run'.
        retry_failed (bool): 
            Whether to count rows marked 'FAILED' for retry.

    Returns:
        int: Number of rows that would still be attempted by the DOE loop.

    Raises:
        None
    """
    count = 0
    for _, r in rows_df.iterrows():

        run_label = r['run']
        prior_status = (manifest_df.loc[run_label, 'status']
                         if run_label in manifest_df.index else None)
        
        if prior_status == 'OK':
            continue
        if prior_status == 'FAILED' and not retry_failed:
            continue
        count += 1

    return count


def run_doe_batch(doe_csv_file: str, 
                  parms_file: str, 
                  consts_file: str,
                  retry_failed: bool = True,
                  run_subset: list = None) -> pd.DataFrame:
    """This function runs a batch of DOE simulations as specified in the CSV file, handling retries and manifest updates automatically (MODE 2).

    Args:
        doe_csv_file (str): 
            DOE CSV path
        parms_file (str): 
            Base parms file
        consts_file (str): 
            Passed to each row's EMSimulation(consts=consts_file).
        retry_failed (bool): 
            Retry manifest rows marked 'FAILED' (default True) or leave them skipped; 'OK' rows are always skipped.
        run_subset (list): 
            Restrict this call to these DOE 'run' ids (default None: all rows); other manifest entries stay untouched.

    Returns:
        pandas.DataFrame: The manifest as of this call (also written to the manifest CSV after every row).

    Raises:
        ValueError: If doe_csv_file has no 'run' column, or run_subset contains an id not present in doe_csv_file.
    """
    # Resolve DOE CSV path and read it into a DataFrame.
    doe_path = Path(doe_csv_file)
    if not doe_path.is_absolute():
        doe_path = Path(__file__).resolve().parent / doe_path
    doe_df = pd.read_csv(doe_path)
    if 'run' not in doe_df.columns:
        raise ValueError(
            f"run_doe_batch(): '{doe_path}' has no 'run' column - a stable per-row identifier is required so the manifest can match CSV rows across separate calls (for resuming/skipping already-simulated rows).")

    # Restrict to the subset of runs if specified. Validate that all requested runs exist in the DOE CSV.
    if run_subset is not None:
        missing = sorted(set(run_subset) - set(doe_df['run']))
        if missing:
            raise ValueError(
                f"run_doe_batch(): run_subset contains 'run' id(s) not present in '{doe_path}': {missing}.")
        doe_df = doe_df[doe_df['run'].isin(run_subset)].reset_index(drop=True)
        print(f"run_doe_batch(): run_subset given - restricting this call to {len(doe_df)} of '{doe_path}''s rows (run id(s): {sorted(run_subset)}). Every other row's manifest entry (if any) is left untouched.")

    # Load the base parameters from the configuration file and check if simulation should be run before analysis (parameter) and output file deletion settings.
    base_parms = ModelMixin._load_config(parms_file)
    if isinstance(base_parms.get('FREQUENCY_RANGE'), list):
        base_parms['FREQUENCY_RANGE'] = tuple(base_parms['FREQUENCY_RANGE'])
    if not base_parms.get('SIMULATE_BEFORE_ANALYSIS', True):
        print(f"run_doe_batch(): WARNING - base parms file has SIMULATE_BEFORE_ANALYSIS = false. Each row would just analyze the SAME EXISTING_RESULTS_FOLDER again instead of running a fresh simulation. Set it true in '{parms_file}' before running the DOE batch.")
    if not base_parms.get('DELETE_OUTPUT_FILES_AFTER_SAVE', False):
        print(f"run_doe_batch(): NOTE - base parms file has DELETE_OUTPUT_FILES_AFTER_SAVE unset/false, so every run's raw gprMax .out file(s) are kept on disk, not just data.pkl. Set that key true in '{parms_file}' if you don't need them kept for every row (analyze()/analyze_doe_batch() only ever read data.pkl).")
    
    water_fraction_low = base_parms['SUT']['INGREDIENTS_FRACTIONS']['WATER']

    # Prepare the manifest file path and columns for tracking DOE run results.
    manifest_path = (Path(base_parms['PATH_INPUT_OUTPUT_FILES'])
                      / f"doe_manifest_{doe_path.stem}.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_columns = list(doe_df.columns) + ['results_folder', 'status', 'error']

    # Check if the manifest file exists and check for rows already marked OK.
    if manifest_path.is_file():
        manifest_df = pd.read_csv(manifest_path, 
                                  dtype={'results_folder': str, 'error': str})
        manifest_df = manifest_df.set_index(manifest_df['run'].astype(doe_df['run'].dtype), drop=False)
        n_ok = (manifest_df['status'] == 'OK').sum()
        
        print(f"run_doe_batch(): resuming from existing manifest '{manifest_path}' - {n_ok}/{len(manifest_df)} row(s) already marked OK there will be skipped.")
    else:
        manifest_df = pd.DataFrame(columns=manifest_columns)
        manifest_df = manifest_df.set_index(pd.Index([], name='run'), drop=False)

    print(f"run_doe_batch(): running {len(doe_df)} DOE parameter set(s) "
          f"from '{doe_path}'. Manifest: '{manifest_path}'.")

    # Compute ETA over rows actually simulated in this call
    batch_start_time = time.time()
    n_attempted = 0
    attempted_time_total = 0.0

    # Iterate over each DOE row and run the simulation if needed.
    for i, row in doe_df.iterrows():

         # Get row label and check prior status in the manifest.
        run_label = row['run']
        prior_status = (manifest_df.loc[run_label, 'status']
                         if run_label in manifest_df.index else None)
        if prior_status == 'OK':
            print(f"\n=== DOE run {run_label} ({i + 1}/{len(doe_df)}): already OK in manifest - skipping ===")
            continue
        if prior_status == 'FAILED' and not retry_failed:
            print(f"\n=== DOE run {run_label} ({i + 1}/{len(doe_df)}): previously FAILED and retry_failed = False - skipping ===")
            continue

        print(f"\n=== DOE run {run_label} ({i + 1}/{len(doe_df)}): diameter={row['diameter_mm']} mm, depth={row['depth_mm']} mm, position={row['position']}, material={row['material']}, heterogeneity={row['heterogeneity']}, moisture={row['moisture']} ===")

        row_start_time = time.time()
        manifest_row = row.to_dict()

        # Build the parameters for this DOE row and run the simulation
        try:
            # Parameters building...
            row_parms = _build_parms_for_doe_row(base_parms, 
                                                 row,water_fraction_low=water_fraction_low)
            em_simulation_row = __import__('EMSimulation').EMSimulation(parms=row_parms, consts=consts_file)

            # Run the simulation for this DOE row
            em_simulation_row.simulate()
            
            manifest_row['results_folder'] = str(em_simulation_row.path_folder)
            manifest_row['status'] = 'OK'
            manifest_row['error'] = ''
        
        except Exception as e:

            # Log and continue (with retrying)
            print(f"DOE run {run_label}: FAILED - {e!r}")
            manifest_row['results_folder'] = ''
            manifest_row['status'] = 'FAILED'
            manifest_row['error'] = repr(e)
        
        finally:

            plt.close('all')

        # Update the manifest and save it as CSV file.
        manifest_df.loc[run_label] = manifest_row
        manifest_df = manifest_df.reindex(columns=manifest_columns)
        manifest_df.to_csv(manifest_path, index=False)

        # Update progress and estimate time remaining for the batch
        n_attempted += 1
        attempted_time_total += time.time() - row_start_time
        avg_row_time = attempted_time_total / n_attempted
        rows_left = _count_pending_doe_rows(doe_df.iloc[i + 1:], 
                                            manifest_df, 
                                            retry_failed)
        eta_seconds = rows_left * avg_row_time
        eta_finish = datetime.fromtimestamp(time.time() + eta_seconds)
        
        print(f"run_doe_batch(): DOE Batch: {i + 1}/{len(doe_df)} processed - {n_attempted} simulated this session, avg {_format_duration(avg_row_time)}/row, elapsed {_format_duration(time.time() - batch_start_time)}. {rows_left} row(s) left to simulate - ETA {_format_duration(eta_seconds)} (~{eta_finish:%Y-%m-%d %H:%M}).")

    # Get number of successful and failed runs and show
    n_ok = (manifest_df['status'] == 'OK').sum()
    n_failed = (manifest_df['status'] == 'FAILED').sum()
    print(f"\nrun_doe_batch(): done. {n_ok} OK, {n_failed} FAILED, {len(doe_df) - len(manifest_df)} not yet attempted. Manifest: '{manifest_path}'.")
    
    return manifest_df
