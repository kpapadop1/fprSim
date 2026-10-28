import itertools
import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.fft
import scipy.stats
import statsmodels.api as sm
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from statsmodels.formula.api import ols

from EMSimulationHelpers import _resolve_cookie_run_folder, _savefig
from EMSimulationPlots import (_anova_plot_pvalue_heatmap,
                               _anova_plot_r_squared_by_response, _plot_mimo_channel_draw_surface,_plot_mimo_channel_sensitivity, plot_combined_contrast_vs_rmsdphase, 
                               PlotsMixin, _FF_FACTOR_ABBR, _FF_FACTOR_NAMES, _FF_FACTOR_SHORT, _FF_HALF_NORMAL_RESPONSES,
                               _FF_SHARE_LABELS, _FF_STUDIES, _FF_VARIANCE_RESPONSES, _plot_ff_half_normal, _plot_ff_interactions,
                               _plot_ff_main_effects, _plot_ff_variance_shares)
from EMSimulationParameters import (ANOVA_CODE_TO_FACTOR, 
                                    ANOVA_FACTOR_CODES, 
                                    MIMO_RESPONSE_CHOICES, _ANOVA_NON_RESPONSE_COLUMNS, _NOISE_FLOOR_UNSET,
                                    _RESPONSE_COLOR_PRIMARY, _RESPONSE_COLOR_SECONDARY, _SRC_STRIDE_CHECK_CONSTS, _SRC_STRIDE_CHECK_PARMS)
if TYPE_CHECKING:
    from EMSimulation import EMSimulation

class AnalysisMixin:
    """Methods of EMSimulation with regard to analysis, used as a class mixin (calculate differential metrics, analyze, scoring, apply ANOVA, and MIMO channel analysis, source-array check, compute moving-sample images)."""

    @staticmethod
    def _unwrap_2d(H: np.ndarray) -> np.ndarray:
        """Apply two-pass phase unwrap (radians) to remove 2*pi offsets, which has the same dimensions as H (n_tx_pos, n_freq): First by frequency axis, followed by TX-position."""
        phase = np.unwrap(np.angle(H), axis=1)
        if H.shape[0] > 1:
            phase = np.unwrap(phase, axis=0)
        return phase

    @staticmethod
    def _group_delay_ns_from_phase(phase_rad: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Determine group delay from an unwrapped phase array (tau_g(f) = -(1 / 2*pi) * d(phase)/df in ns)."""
        tau_g_s = -np.gradient(phase_rad, f, axis=-1) / (2 * np.pi)
        return tau_g_s * 1e9

    @staticmethod
    def _load_run_data(folder: str, verbose: bool = True, side: str = None) -> tuple:
        """Load MIMO channel matrix, frequency vector, and related data from a run folder's data.pkl.

        Args:
            folder (str): 
                Run folder containing data.pkl (or data_<side>.pkl).
            verbose (bool): 
                Print a "loading results" line (default True).
            side (str): 
                None (default) reads data.pkl; 'reflection' / 'transmission' reads data_<side>.pkl of a run made with ALSO_MIRRORED_RX.

        Returns:
            tuple: 
                MIMO channel matrix H_tf (n_channels, n_freq), frequency vector f_fft, TX position tx_pos or None, dielectric or None, channel_labels.

        Raises:
            FileNotFoundError: If "folder" doesn't exist.
        """

        # Apply checks and print status
        if not Path(folder).exists():
            raise FileNotFoundError(
                f"EMSimulation._load_run_data(): specified folder "
                f"'{folder}' does not exist.")
        
        if verbose:
            print(f"_load_run_data(): loading results from '{folder}'...")

        if side is not None and side not in ('reflection', 'transmission'):
            raise ValueError(f"EMSimulation._load_run_data(): side must be None, 'reflection' or 'transmission', got {side!r}.")

        # Load data if .pkl file exists
        fname = 'data.pkl' if side is None else f'data_{side}.pkl'
        if not (Path(folder) / fname).is_file():
            raise FileNotFoundError(f"EMSimulation._load_run_data(): {Path(folder) / fname} does not exist"
                                    + ("" if side is None else " (run made without ALSO_MIRRORED_RX, or not finished)") + ".")
        with open(Path(folder) / fname, 'rb') as f:
            data = pickle.load(f)
        if side is not None and data.get('rx_side') != side:
            raise RuntimeError(f"EMSimulation._load_run_data(): {Path(folder) / fname} holds rx_side={data.get('rx_side')!r}, expected {side!r}.")

        # Get data and pack tuple
        H_tf = data['H_tf']
        if H_tf.ndim == 3:
            n_tx, n_rx, n_freq = H_tf.shape
            H_tf = H_tf.reshape(n_tx * n_rx, n_freq)
            channel_labels = [f'TX{t} RX{r}' for t in range(n_tx) for r in range(n_rx)]
        else:
            channel_labels = [f'TX{i}' for i in range(H_tf.shape[0])]

        return (H_tf, data['f_fft'], data.get('tx_pos', None),
                data.get('dielectric', None), channel_labels)

    @staticmethod
    def _compute_differential_metrics(H_tf: np.ndarray, 
                                      f_fft: np.ndarray,
                                      H_tf_base: np.ndarray, 
                                      f_fft_base: np.ndarray,
                                      smooth_window: int = 0,
                                      delay_res_ns: float = None,
                                      amp_noise_floor_db: float = None,phase_noise_floor_deg: float = None) -> dict:
        """Compute differential max. and mean (along channels) metrics, that is one run vs. a baseline (H_ratio = H_tf / H_tf_base).

        Args:
            H_tf (np.ndarray): 
                This run's transfer function, shape (n_tx_pos, n_freq).
            f_fft (np.ndarray): 
                This run's frequency axis, shape (n_freq,).
            H_tf_base, f_fft_base (np.ndarray): 
                Baseline H_tf / frequency axis, same shapes.
            smooth_window (int): 
                Savitzky-Golay window (odd, frequency samples) applied to group delay (default 0: none).
            delay_res_ns (float): 
                detectability threshold for the RMS delta group delay panels; None disables that panel.
            amp_noise_floor_db (float): 
                detectability threshold for the RMS delta amplitude panels; None disables that panel.
            phase_noise_floor_deg (float): 
                detectability threshold for the RMS delta phase panels; None disables that panel.

        Returns:
            dict: 
                '<name>_max' and '<name>_mean' (float) for mean_dtau, rms_dtau, mean_dmag, rms_dmag, mean_dphase, rms_dphase, n_exceed, weighted_score,'frobenius_rel' (float), the RMS deviation of H_ratio across all channels and bins.

        Raises:
            ValueError: If H_tf's shape doesn't match H_tf_base's.
        """
        if H_tf.shape != H_tf_base.shape:
            raise ValueError(
                f"EMSimulation._compute_differential_metrics(): shape of MIMO channel matrix "
                f"{H_tf.shape} != baseline shape {H_tf_base.shape}. Runs "
                f"must share the same N_POS and frequency grid as the "
                f"baseline to compute a differential.")
        if not np.allclose(f_fft, f_fft_base):
            print("_compute_differential_metrics(): warning. Frequency "
                  "axis differs from baseline. Differential quantities "
                  "may be misleading.")

        n_tx_pos = H_tf.shape[0]
        H_ratio_all = H_tf / H_tf_base
        dmag_db_all = 20 * np.log10(np.abs(H_ratio_all))
        dphase_rad_all = AnalysisMixin._unwrap_2d(H_ratio_all)
        dphase_deg_all = np.degrees(dphase_rad_all)
        dtau_g_ns_all = AnalysisMixin._group_delay_ns_from_phase(dphase_rad_all, f_fft)

        # Comput relative frobenius norm
        frobenius_rel = float(np.linalg.norm(H_ratio_all - 1)
                               / np.sqrt(H_ratio_all.size))

        def _maybe_smooth(y):
            """Apply smoothing using a Savitzky-Golay filter if smooth_window is valid, else return y unchanged.
            
            Args:
                y (np.ndarray): Input array to smooth.

            Returns:
                np.ndarray: Smoothed array if smoothing is applied, else the original array.
            """
            if smooth_window <= 0:
                return y
            window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            if window <= 1 or window >= len(y):
                return y
            return savgol_filter(y, window_length=window, polyorder=min(3, window - 1))

        # Preallocate
        mean_dtau = np.zeros(n_tx_pos)
        rms_dtau = np.zeros(n_tx_pos)
        mean_dmag = np.zeros(n_tx_pos)
        rms_dmag = np.zeros(n_tx_pos)
        mean_dphase = np.zeros(n_tx_pos)
        rms_dphase = np.zeros(n_tx_pos)

        # Average or apply smoothing
        for tx_idx in range(n_tx_pos):
            dmag_db = dmag_db_all[tx_idx]
            dphase_deg = dphase_deg_all[tx_idx]
            dtau_g_ns = _maybe_smooth(dtau_g_ns_all[tx_idx])
            mean_dtau[tx_idx] = np.mean(dtau_g_ns)
            rms_dtau[tx_idx] = np.sqrt(np.mean(dtau_g_ns ** 2))
            mean_dmag[tx_idx] = np.mean(dmag_db)
            rms_dmag[tx_idx] = np.sqrt(np.mean(dmag_db ** 2))
            mean_dphase[tx_idx] = np.mean(dphase_deg)
            rms_dphase[tx_idx] = np.sqrt(np.mean(dphase_deg ** 2))

        # Assign detectability scores for every TX-RX channel
        n_exceed = np.zeros(n_tx_pos, dtype=int)
        weighted_score = np.zeros(n_tx_pos)
        if delay_res_ns is not None:
            n_exceed += (rms_dtau > delay_res_ns).astype(int)
            weighted_score += np.maximum(0.0, rms_dtau / delay_res_ns - 1.0)
        if amp_noise_floor_db is not None:
            n_exceed += (rms_dmag > amp_noise_floor_db).astype(int)
            weighted_score += np.maximum(0.0, rms_dmag / amp_noise_floor_db - 1.0)
        if phase_noise_floor_deg is not None:
            n_exceed += (rms_dphase > phase_noise_floor_deg).astype(int)
            weighted_score += np.maximum(0.0, rms_dphase / phase_noise_floor_deg - 1.0)

        # Set results dictionary with max and mean for each metric, plus frobenius_rel
        out = {}
        for name, arr in [('mean_dtau', mean_dtau), ('rms_dtau', rms_dtau),
                           ('mean_dmag', mean_dmag), ('rms_dmag', rms_dmag),
                           ('mean_dphase', mean_dphase), ('rms_dphase', rms_dphase),
                           ('n_exceed', n_exceed), ('weighted_score', weighted_score)]:
            out[f'{name}_max'] = float(np.max(arr))
            out[f'{name}_mean'] = float(np.mean(arr))
        out['frobenius_rel'] = frobenius_rel
        return out

    def analyze(self, 
                results_folder: list, 
                baseline_folder: str = None,
                smooth_window: int = 0, 
                amp_noise_floor_db=_NOISE_FLOOR_UNSET,
                phase_noise_floor_deg=_NOISE_FLOOR_UNSET) -> None:
        """Analyze by loading H_tf, f_fft and tx_pos from pickeled results file and then plot H_tf magnitude, phase and group delay.

        Args:
            results_folder (list): 
                Folders with the runs to post-process (each containing data.pkl).
            baseline_folder (str): 
                Optional reference run (same H_tf shape and frequency axis); None disables the differential figures.
            smooth_window (int): 
                Flag: Odd Savitzky-Golay window (frequency samples) for the group delay curves (default: 0 -> disabled).
            amp_noise_floor_db (float): 
                Detectability threshold for the RMS delta |H| panels. Omitted = instance value (AMP_NOISE_FLOOR_DB; None: disable for this call; float: override).
            phase_noise_floor_deg (float): 
                Detectability threshold for the RMS delta phase panels (PHASE_NOISE_FLOOR_DEG).

        Raises:
            ValueError: If a run's H_tf shape does not match the baseline's.
        """
        # Get noise-floor thresholds
        if amp_noise_floor_db is _NOISE_FLOOR_UNSET:
            amp_noise_floor_db = self.amp_noise_floor_db
        if phase_noise_floor_deg is _NOISE_FLOOR_UNSET:
            phase_noise_floor_deg = self.phase_noise_floor_deg

        # Load results from pickled files in lists
        H_tf_list = []
        f_fft_list = []
        tx_pos_list = []
        dielectric_list = []
        channel_labels_list = []
        for folder in results_folder:
            H_tf, f_fft, tx_pos, dielectric, channel_labels = self._load_run_data(folder)
            H_tf_list.append(H_tf)
            f_fft_list.append(f_fft)
            tx_pos_list.append(tx_pos)
            dielectric_list.append(dielectric)
            channel_labels_list.append(channel_labels)

        def _maybe_smooth(y: np.ndarray) -> np.ndarray:
            """Apply smooting using Savitzky-Golay filter if smooth_window allows it."""
            window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            if window <= 1 or window >= len(y):
                return y
            return savgol_filter(y, window_length=window, polyorder=min(3, window - 1))

        n_runs = len(H_tf_list)
        run_labels = [Path(folder).name for folder in results_folder]

        # Set up Figure 1 containing magnitude, phase, group delay vs. frequency
        fig1, axes = plt.subplots(nrows=3, 
                                  ncols=n_runs, 
                                  figsize=(6 * n_runs, 11),
                                  squeeze=False, 
                                  sharex=True)
        fig1.canvas.manager.set_window_title("postprocess(): Transfer Function H_tf")

        # Show per-channel group delay statistics for this run
        mean_tau_g_list = []
        std_tau_g_list = []
        for col, (H_tf, f_fft, label, channel_labels) in enumerate(
                zip(H_tf_list, f_fft_list, run_labels, channel_labels_list)):
            
            f_ghz = np.asarray(f_fft) / 1e9
            n_tx_pos = H_tf.shape[0]
            cmap = plt.get_cmap('viridis')
            colors = [cmap(ch_idx / max(n_tx_pos - 1, 1)) for ch_idx in range(n_tx_pos)]

            ax_mag, ax_phase, ax_delay = axes[0, col], axes[1, col], axes[2, col]
            mean_tau_g = np.zeros(n_tx_pos)
            std_tau_g = np.zeros(n_tx_pos)

            # Unwrap and differentiate once over the whole (n_channels, n_freq) array
            phase_rad_all = self._unwrap_2d(H_tf)
            tau_g_ns_all = self._group_delay_ns_from_phase(phase_rad_all, f_fft)

            # Determine relevant values for each channel
            for ch_idx in range(n_tx_pos):
                mag_db = 20 * np.log10(np.abs(H_tf[ch_idx]))
                phase_deg = np.degrees(phase_rad_all[ch_idx])
                tau_g_ns = _maybe_smooth(tau_g_ns_all[ch_idx])
                mean_tau_g[ch_idx] = np.mean(tau_g_ns)
                std_tau_g[ch_idx] = np.std(tau_g_ns)

                # Get labels and plot
                ch_label = channel_labels[ch_idx]
                ax_mag.plot(f_ghz, mag_db, color=colors[ch_idx], label=ch_label)
                ax_phase.plot(f_ghz, phase_deg, color=colors[ch_idx], label=ch_label)
                ax_delay.plot(f_ghz, tau_g_ns, color=colors[ch_idx], label=ch_label)

            mean_tau_g_list.append(mean_tau_g)
            std_tau_g_list.append(std_tau_g)

            # Set figure labels and grid
            ax_mag.set_title(label)
            ax_mag.set_ylabel("|H_tf| / dB")
            ax_mag.grid(True, alpha=0.3)
            ax_phase.set_ylabel("phase(H_tf) / deg")
            ax_phase.grid(True, alpha=0.3)
            ax_delay.set_xlabel("Frequency / GHz")
            ax_delay.set_ylabel("group delay / ns")
            ax_delay.grid(True, alpha=0.3)
            if n_tx_pos <= 12:
                ax_mag.legend(fontsize=7, ncol=2 if n_tx_pos > 6 else 1)

        plt.tight_layout()

        # Determine group delay mean/std per TX-RX channel for every run. Start by setting up Figure 2.
        longest_labels = max(channel_labels_list, key=len)
        fig2, (ax_mean, ax_std) = plt.subplots(nrows=1, 
                                               ncols=2, 
                                               figsize=(13, 5))
        fig2.canvas.manager.set_window_title("postprocess(): group delay vs. TX-RX channel")

        # Plot mean and std of group delay for each run
        for H_tf, mean_tau_g, std_tau_g, label in zip(
                H_tf_list, mean_tau_g_list, std_tau_g_list, run_labels):
            n_tx_pos = H_tf.shape[0]
            x_vals = np.arange(n_tx_pos)
            ax_mean.plot(x_vals, mean_tau_g, marker='o', label=label)
            ax_std.plot(x_vals, std_tau_g, marker='o', label=label)

        # Set ticks and labels for the group delay panels
        x_label = "TX-RX channel"
        x_ticks = np.arange(len(longest_labels))
        for ax in (ax_mean, ax_std):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(longest_labels, rotation=60, ha='right', fontsize=7)
        ax_mean.set_xlabel(x_label)
        ax_mean.set_ylabel("Mean group delay (over band) / ns")
        ax_mean.set_title("Bulk delay per TX-RX channel")
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(fontsize=8)

        # Set ticks and labels for the standard deviation panel
        ax_std.set_xlabel(x_label)
        ax_std.set_ylabel("Std of group delay (over band) / ns")
        ax_std.set_title("Dispersion per TX-RX channel (anomaly signature)")
        ax_std.grid(True, alpha=0.3)
        ax_std.legend(fontsize=8)
        plt.tight_layout()

        # Apply differential comparison using H_ratio = H_tf_run / H_tf_baseline (Figures 3, 4)
        if baseline_folder is not None:

            H_tf_base, f_fft_base, tx_pos_base, _, _ = self._load_run_data(baseline_folder)
            baseline_name = Path(baseline_folder).name

            # Get delay resolution
            bandwidth_hz = float(np.asarray(f_fft_base).max() - np.asarray(f_fft_base).min())
            delay_res_ns = (1e9 / bandwidth_hz) if bandwidth_hz > 0 else None

            # Determine reference for the dB/phase panels
            noise_floor_mag_db = None
            noise_floor_phase_deg = None

            # Set up figure 3
            fig3, axes3 = plt.subplots(nrows=3, 
                                       ncols=n_runs, 
                                       figsize=(6 * n_runs, 11),
                                       squeeze=False, 
                                       sharex=True)
            fig3.canvas.manager.set_window_title(
                f"postprocess(): Differential vs. baseline ({baseline_name})")

            # Preallocate lists for mean and RMS values across runs
            mean_dtau_list = []
            rms_dtau_list = []
            mean_dmag_list = []
            rms_dmag_list = []
            mean_dphase_list = []
            rms_dphase_list = []

            # Plot differential metrics for each run vs. the baseline (magnitude, phase and group delay over frequency)
            for col, (H_tf, f_fft, label, channel_labels) in enumerate(
                    zip(H_tf_list, f_fft_list, run_labels, channel_labels_list)):
                if H_tf.shape != H_tf_base.shape:
                    raise ValueError(
                        f"postprocess(): H_tf shape mismatch between run "
                        f"'{label}' {H_tf.shape} and baseline "
                        f"'{baseline_name}' {H_tf_base.shape} - "
                        f"cannot compute a differential (runs must share "
                        f"the same N_TX/N_RX and frequency grid as the "
                        f"baseline).")
                if not np.allclose(f_fft, f_fft_base):
                    print(f"postprocess(): warning. Frequency axis of "
                          f"'{label}' differs from baseline's; differential "
                          f"quantities may be misleading.")

                f_ghz = np.asarray(f_fft) / 1e9
                n_tx_pos = H_tf.shape[0]
                cmap = plt.get_cmap('viridis')
                colors = [cmap(tx_idx / max(n_tx_pos - 1, 1)) for tx_idx in range(n_tx_pos)]

                ax_dmag, ax_dphase, ax_ddelay = axes3[0, col], axes3[1, col], axes3[2, col]
                mean_dtau = np.zeros(n_tx_pos)
                rms_dtau = np.zeros(n_tx_pos)
                mean_dmag = np.zeros(n_tx_pos)
                rms_dmag = np.zeros(n_tx_pos)
                mean_dphase = np.zeros(n_tx_pos)
                rms_dphase = np.zeros(n_tx_pos)
                is_self_diff = (label == baseline_name)
                self_max_mag_db = 0.0
                self_max_phase_deg = 0.0

                # Unwrap
                H_ratio_all = H_tf / H_tf_base
                dmag_db_all = 20 * np.log10(np.abs(H_ratio_all))
                dphase_rad_all = self._unwrap_2d(H_ratio_all)
                dphase_deg_all = np.degrees(dphase_rad_all)
                dtau_g_ns_all = self._group_delay_ns_from_phase(dphase_rad_all, f_fft)

                # Apply averaging and smooting for each TX-RX channel
                for tx_idx in range(n_tx_pos):

                    dmag_db = dmag_db_all[tx_idx]
                    dphase_deg = dphase_deg_all[tx_idx]
                    dtau_g_ns = _maybe_smooth(dtau_g_ns_all[tx_idx])
                    mean_dtau[tx_idx] = np.mean(dtau_g_ns)
                    rms_dtau[tx_idx] = np.sqrt(np.mean(dtau_g_ns ** 2))
                    mean_dmag[tx_idx] = np.mean(dmag_db)
                    rms_dmag[tx_idx] = np.sqrt(np.mean(dmag_db ** 2))
                    mean_dphase[tx_idx] = np.mean(dphase_deg)
                    rms_dphase[tx_idx] = np.sqrt(np.mean(dphase_deg ** 2))

                    if is_self_diff:
                        self_max_mag_db = max(self_max_mag_db, np.max(np.abs(dmag_db)))
                        self_max_phase_deg = max(self_max_phase_deg, np.max(np.abs(dphase_deg)))

                    ch_label = channel_labels[tx_idx]
                    ax_dmag.plot(f_ghz, dmag_db, color=colors[tx_idx], label=ch_label)
                    ax_dphase.plot(f_ghz, dphase_deg, color=colors[tx_idx], label=ch_label)
                    ax_ddelay.plot(f_ghz, dtau_g_ns, color=colors[tx_idx], label=ch_label)

                mean_dtau_list.append(mean_dtau)
                rms_dtau_list.append(rms_dtau)
                mean_dmag_list.append(mean_dmag)
                rms_dmag_list.append(rms_dmag)
                mean_dphase_list.append(mean_dphase)
                rms_dphase_list.append(rms_dphase)
                if is_self_diff:
                    noise_floor_mag_db = self_max_mag_db
                    noise_floor_phase_deg = self_max_phase_deg

                ax_dmag.set_title(label)
                ax_dmag.axhline(0, color='gray', linewidth=0.8, linestyle='--')
                ax_dmag.set_ylabel("Delta|H_tf| / dB (vs. baseline)")
                ax_dmag.grid(True, alpha=0.3)
                ax_dphase.axhline(0, color='gray', linewidth=0.8, linestyle='--')
                ax_dphase.set_ylabel("Delta phase / deg (vs. baseline)")
                ax_dphase.grid(True, alpha=0.3)
                ax_ddelay.axhline(0, color='gray', linewidth=0.8, linestyle='--')
                ax_ddelay.set_xlabel("Frequency / GHz")
                ax_ddelay.set_ylabel("Delta group delay / ns (vs. baseline)")
                ax_ddelay.grid(True, alpha=0.3)
                if n_tx_pos <= 12:
                    ax_dmag.legend(fontsize=7, ncol=2 if n_tx_pos > 6 else 1)

            # Check if baseline folder exists
            if noise_floor_mag_db is None:
                print(f"postprocess(): baseline_folder ('{baseline_name}') is "
                      f"not included in results_folder. Skipping the "
                      f"numerical-floor reference band on the dB/phase "
                      f"panels. Include it in results_folder too (its "
                      f"diff is an exact zero-check) to get that "
                      f"reference automatically.")

            # Draw the reference bands on every column
            for col in range(n_runs):
                ax_dmag, ax_dphase, ax_ddelay = axes3[0, col], axes3[1, col], axes3[2, col]
                if noise_floor_mag_db is not None:
                    ax_dmag.axhspan(-noise_floor_mag_db, noise_floor_mag_db,
                                     color='gray', alpha=0.15, zorder=0)
                    ax_dmag.text(0.02, 0.95, f"numerical floor: ±{noise_floor_mag_db:.3g} dB",
                                 transform=ax_dmag.transAxes, fontsize=7, color='gray', va='top')
                if amp_noise_floor_db is not None:
                    ax_dmag.axhline(amp_noise_floor_db, color='steelblue', linewidth=1.0, linestyle=':')
                    ax_dmag.axhline(-amp_noise_floor_db, color='steelblue', linewidth=1.0, linestyle=':')
                    ax_dmag.text(0.02, 0.88, f"system floor: ±{amp_noise_floor_db:.3g} dB",
                                 transform=ax_dmag.transAxes, fontsize=7, color='steelblue', va='top')
                if noise_floor_phase_deg is not None:
                    ax_dphase.axhspan(-noise_floor_phase_deg, noise_floor_phase_deg,
                                       color='gray', alpha=0.15, zorder=0)
                    ax_dphase.text(0.02, 0.95, f"numerical floor: ±{noise_floor_phase_deg:.3g} deg",
                                   transform=ax_dphase.transAxes, fontsize=7, color='gray', va='top')
                if phase_noise_floor_deg is not None:
                    ax_dphase.axhline(phase_noise_floor_deg, color='steelblue', linewidth=1.0, linestyle=':')
                    ax_dphase.axhline(-phase_noise_floor_deg, color='steelblue', linewidth=1.0, linestyle=':')
                    ax_dphase.text(0.02, 0.88, f"system floor: ±{phase_noise_floor_deg:.3g} deg",
                                   transform=ax_dphase.transAxes, fontsize=7, color='steelblue', va='top')
                if delay_res_ns is not None:
                    ax_ddelay.axhspan(-delay_res_ns, delay_res_ns,
                                       color='steelblue', alpha=0.12, zorder=0)
                    ax_ddelay.text(0.02, 0.95, f"±1/bandwidth: ±{delay_res_ns:.3g} ns",
                                   transform=ax_ddelay.transAxes, fontsize=7, color='steelblue', va='top')

            plt.tight_layout()

            # Determine and plot mean/RMS vs. TX-RX channel for group delay, amplitude and phase
            n_available_channels = (int(delay_res_ns is not None)
                                     + int(amp_noise_floor_db is not None)
                                     + int(phase_noise_floor_deg is not None))

            # Set up Figure 4 and subplots
            fig4 = plt.figure(figsize=(13, 19))
            gs4 = fig4.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 0.8], hspace=0.35)
            fig4.canvas.manager.set_window_title(
                f"postprocess(): differential summary vs. TX position (baseline: {baseline_name})")
            ax_dmean = fig4.add_subplot(gs4[0, 0])
            ax_drms = fig4.add_subplot(gs4[0, 1])
            ax_dmagmean = fig4.add_subplot(gs4[1, 0])
            ax_dmagrms = fig4.add_subplot(gs4[1, 1])
            ax_dphasemean = fig4.add_subplot(gs4[2, 0])
            ax_dphaserms = fig4.add_subplot(gs4[2, 1])
            ax_confidence = fig4.add_subplot(gs4[3, :])
            ax_confidence_weighted = ax_confidence.twinx()

            # Plot per-channel statistics for each run for each channel
            for (H_tf, tx_pos, mean_dtau, rms_dtau, mean_dmag, rms_dmag,
                 mean_dphase, rms_dphase, label) in zip(
                    H_tf_list, tx_pos_list, mean_dtau_list, rms_dtau_list,
                    mean_dmag_list, rms_dmag_list, mean_dphase_list,
                    rms_dphase_list, run_labels):
                
                n_tx_pos = H_tf.shape[0]

                # Draw delta metrics (group delay, amplitude, phase)
                x_vals = np.arange(n_tx_pos)
                ax_dmean.plot(x_vals, mean_dtau, marker='o', label=label)
                ax_drms.plot(x_vals, rms_dtau, marker='o', label=label)
                ax_dmagmean.plot(x_vals, mean_dmag, marker='o', label=label)
                ax_dmagrms.plot(x_vals, rms_dmag, marker='o', label=label)
                ax_dphasemean.plot(x_vals, mean_dphase, marker='o', label=label)
                ax_dphaserms.plot(x_vals, rms_dphase, marker='o', label=label)

                # Determine combined detectability by number of channels whose RMS deviation exceeds its own threshold
                n_exceed = np.zeros(n_tx_pos, dtype=int)
                weighted_score = np.zeros(n_tx_pos)
                if delay_res_ns is not None:
                    n_exceed += (rms_dtau > delay_res_ns).astype(int)
                    weighted_score += np.maximum(0.0, rms_dtau / delay_res_ns - 1.0)
                if amp_noise_floor_db is not None:
                    n_exceed += (rms_dmag > amp_noise_floor_db).astype(int)
                    weighted_score += np.maximum(0.0, rms_dmag / amp_noise_floor_db - 1.0)
                if phase_noise_floor_deg is not None:
                    n_exceed += (rms_dphase > phase_noise_floor_deg).astype(int)
                    weighted_score += np.maximum(0.0, rms_dphase / phase_noise_floor_deg - 1.0)
                count_line, = ax_confidence.plot(x_vals, n_exceed, marker='o', label=label, drawstyle='steps-mid')
                ax_confidence_weighted.plot(x_vals, weighted_score, marker='s', markersize=4,
                                             linestyle='--', linewidth=1.0, alpha=0.7,
                                             color=count_line.get_color())

            # Set ticks and labels for the differential summary panels
            x_ticks = np.arange(len(longest_labels))
            for ax in (ax_dmean, ax_drms, ax_dmagmean, ax_dmagrms,
                       ax_dphasemean, ax_dphaserms, ax_confidence):
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(longest_labels, rotation=60, ha='right', fontsize=6)

            # Draw delay resolution 1/bandwidth (blue) in delta group delay panels 
            if delay_res_ns is not None:
                ax_dmean.axhspan(-delay_res_ns, delay_res_ns, color='steelblue', alpha=0.12, zorder=0)
                ax_dmean.text(0.02, 0.95, f"±1/bandwidth: ±{delay_res_ns:.3g} ns",
                              transform=ax_dmean.transAxes, fontsize=7, color='steelblue', va='top')
                ax_drms.axhspan(0, delay_res_ns, color='steelblue', alpha=0.12, zorder=0)
                ax_drms.axhline(delay_res_ns, color='steelblue', linewidth=1.0, linestyle=':')
                ax_drms.text(0.02, 0.95, f"1/bandwidth: {delay_res_ns:.3g} ns",
                             transform=ax_drms.transAxes, fontsize=7, color='steelblue', va='top')

            # Draw numerical floor in delta magnitude panels
            if noise_floor_mag_db is not None:
                ax_dmagmean.axhspan(-noise_floor_mag_db, noise_floor_mag_db, color='gray', alpha=0.15, zorder=0)
                ax_dmagmean.text(0.02, 0.95, f"numerical floor: ±{noise_floor_mag_db:.3g} dB",
                                  transform=ax_dmagmean.transAxes, fontsize=7, color='gray', va='top')
                ax_dmagrms.axhspan(0, noise_floor_mag_db, color='gray', alpha=0.15, zorder=0)
                ax_dmagrms.axhline(noise_floor_mag_db, color='gray', linewidth=1.0, linestyle=':')
                ax_dmagrms.text(0.02, 0.95, f"numerical floor: {noise_floor_mag_db:.3g} dB",
                                 transform=ax_dmagrms.transAxes, fontsize=7, color='gray', va='top')

            # Draw noise floor in delta magnitude panels
            if amp_noise_floor_db is not None:
                ax_dmagmean.axhline(amp_noise_floor_db, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dmagmean.axhline(-amp_noise_floor_db, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dmagmean.text(0.02, 0.85, f"system floor: ±{amp_noise_floor_db:.3g} dB",
                                  transform=ax_dmagmean.transAxes, fontsize=7, color='steelblue', va='top')
                ax_dmagrms.axhspan(0, amp_noise_floor_db, color='steelblue', alpha=0.12, zorder=0)
                ax_dmagrms.axhline(amp_noise_floor_db, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dmagrms.text(0.02, 0.85, f"system floor: {amp_noise_floor_db:.3g} dB",
                                 transform=ax_dmagrms.transAxes, fontsize=7, color='steelblue', va='top')

            # Draw numerical floor in delta phase panels
            if noise_floor_phase_deg is not None:
                ax_dphasemean.axhspan(-noise_floor_phase_deg, noise_floor_phase_deg, color='gray', alpha=0.15, zorder=0)
                ax_dphasemean.text(0.02, 0.95, f"numerical floor: ±{noise_floor_phase_deg:.3g} deg",
                                    transform=ax_dphasemean.transAxes, fontsize=7, color='gray', va='top')
                ax_dphaserms.axhspan(0, noise_floor_phase_deg, color='gray', alpha=0.15, zorder=0)
                ax_dphaserms.axhline(noise_floor_phase_deg, color='gray', linewidth=1.0, linestyle=':')
                ax_dphaserms.text(0.02, 0.95, f"numerical floor: {noise_floor_phase_deg:.3g} deg",
                                   transform=ax_dphaserms.transAxes, fontsize=7, color='gray', va='top')

            # Draw system floor in delta phase panels
            if phase_noise_floor_deg is not None:
                ax_dphasemean.axhline(phase_noise_floor_deg, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dphasemean.axhline(-phase_noise_floor_deg, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dphasemean.text(0.02, 0.85, f"system floor: ±{phase_noise_floor_deg:.3g} deg",
                                    transform=ax_dphasemean.transAxes, fontsize=7, color='steelblue', va='top')
                ax_dphaserms.axhspan(0, phase_noise_floor_deg, color='steelblue', alpha=0.12, zorder=0)
                ax_dphaserms.axhline(phase_noise_floor_deg, color='steelblue', linewidth=1.0, linestyle=':')
                ax_dphaserms.text(0.02, 0.85, f"system floor: {phase_noise_floor_deg:.3g} deg",
                                   transform=ax_dphaserms.transAxes, fontsize=7, color='steelblue', va='top')

            # Set ticks and labels for the differential summary panels
            ax_dmean.set_xlabel(x_label)
            ax_dmean.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_dmean.set_ylabel("Mean delta group delay / ns")
            ax_dmean.set_title("Net delay shift vs. baseline")
            ax_dmean.grid(True, alpha=0.3)
            ax_dmean.legend(fontsize=8)

            ax_drms.set_xlabel(x_label)
            ax_drms.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_drms.set_ylabel("RMS delta group delay / ns")
            ax_drms.set_title("Differential dispersion vs. baseline (anomaly signature)")
            ax_drms.grid(True, alpha=0.3)
            ax_drms.legend(fontsize=8)

            ax_dmagmean.set_xlabel(x_label)
            ax_dmagmean.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_dmagmean.set_ylabel("Mean delta |H_tf| / dB")
            ax_dmagmean.set_title("Net amplitude shift vs. baseline")
            ax_dmagmean.grid(True, alpha=0.3)
            ax_dmagmean.legend(fontsize=8)

            ax_dmagrms.set_xlabel(x_label)
            ax_dmagrms.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_dmagrms.set_ylabel("RMS delta |H_tf| / dB")
            ax_dmagrms.set_title("Differential amplitude dispersion vs. baseline (anomaly signature)")
            ax_dmagrms.grid(True, alpha=0.3)
            ax_dmagrms.legend(fontsize=8)

            ax_dphasemean.set_xlabel(x_label)
            ax_dphasemean.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_dphasemean.set_ylabel("Mean delta phase / deg")
            ax_dphasemean.set_title("Net phase shift vs. baseline")
            ax_dphasemean.grid(True, alpha=0.3)
            ax_dphasemean.legend(fontsize=8)

            ax_dphaserms.set_xlabel(x_label)
            ax_dphaserms.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_dphaserms.set_ylabel("RMS delta phase / deg")
            ax_dphaserms.set_title("Differential phase dispersion vs. baseline (anomaly signature)")
            ax_dphaserms.grid(True, alpha=0.3)
            ax_dphaserms.legend(fontsize=8)

            channel_names = []
            if delay_res_ns is not None:
                channel_names.append("delay")
            if amp_noise_floor_db is not None:
                channel_names.append("amplitude")
            if phase_noise_floor_deg is not None:
                channel_names.append("phase")
            ax_confidence.set_xlabel(x_label)
            ax_confidence.set_ylabel("channels exceeding threshold")
            ax_confidence.set_yticks(range(n_available_channels + 1))
            ax_confidence.set_ylim(-0.3, n_available_channels + 0.3)
            ax_confidence_weighted.set_ylabel("weighted exceedance score")
            ax_confidence_weighted.set_ylim(bottom=0)
            ax_confidence.set_title(
                f"Combined detectability: channels exceeding their physical "
                f"threshold (of {n_available_channels}: "
                f"{', '.join(channel_names) if channel_names else 'none available'})")
            ax_confidence.grid(True, alpha=0.3)
            ax_confidence.legend(fontsize=8, loc='upper right')
            ax_confidence.text(
                0.02, 0.95,
                "solid o = channel count (left axis)\ndashed □ = weighted "
                "exceedance score, sum of (value/threshold - 1) (right axis)",
                transform=ax_confidence.transAxes, fontsize=7, color='gray', va='top')
            if n_available_channels < 3:
                missing = [n for n in ("delay", "amplitude", "phase") if n not in channel_names]
                ax_confidence.text(
                    0.02, 0.80,
                    f"note: {', '.join(missing)} threshold{'s' if len(missing) > 1 else ''} not "
                    f"supplied - pass amp_noise_floor_db/phase_noise_floor_deg to "
                    f"analyze() to include {'them' if len(missing) > 1 else 'it'}.",
                    transform=ax_confidence.transAxes, fontsize=7, color='gray', va='top')
            fig4.tight_layout()

        # Determine dielectric contrast between foreign object and SUT per run
        contrast_list = []
        for f_fft, dielectric, label in zip(f_fft_list, dielectric_list, run_labels):

            f_center = float((np.asarray(f_fft).min() + 
                              np.asarray(f_fft).max()) / 2)
            contrast = self._dielectric_contrast(f_center, dielectric)
            contrast_list.append(contrast)

            if contrast is None:
                reason = ("data.pkl predates the 'dielectric' snapshot (run "
                          "the simulation again to get one)" if dielectric is None
                          else "SUT mode == 'SURROUNDING' - no SUT material to "
                          "contrast against")
                print(f"analyze(): dielectric contrast for '{label}' - "
                      f"not available ({reason}).")
            else:
                print(f"analyze(): dielectric contrast for '{label}' at "
                      f"{f_center / 1e9:.2f} GHz - eps_fo={contrast['eps_fo']:.3g}, "
                      f"eps_sut={contrast['eps_sut']:.3g}, "
                      f"contrast_real={contrast['contrast_real']:+.3g}, "
                      f"contrast_complex_mag={contrast['contrast_complex_mag']:.3g}")

        if any(c is not None for c in contrast_list):

            # Set up Figure 5: dielectric contrast (materials-only)
            fig5, ax_contrast = plt.subplots(figsize=(max(6, 1.2 * n_runs), 5))
            fig5.canvas.manager.set_window_title("postprocess(): dielectric contrast (materials-only)")
            
            x_pos = np.arange(n_runs)
            contrast_real_vals = [c['contrast_real'] 
                                  if c is not None else np.nan for c in contrast_list]
            contrast_mag_vals = [c['contrast_complex_mag'] 
                                 if c is not None else np.nan for c in contrast_list]
            width = 0.35

            # Set figure denomination (ticks, labels, titles)
            ax_contrast.bar(x_pos - width / 2, contrast_real_vals, width,
                             label=PlotsMixin._response_label('contrast_real'),
                             color=_RESPONSE_COLOR_PRIMARY)
            ax_contrast.bar(x_pos + width / 2, contrast_mag_vals, width,
                             label=PlotsMixin._response_label('contrast_complex_mag'),
                             color=_RESPONSE_COLOR_SECONDARY)
            self._style_response_axis(
                ax_contrast, 'contrast_real', contrast_real_vals + contrast_mag_vals)
            ax_contrast.set_xticks(x_pos)
            ax_contrast.set_xticklabels(run_labels, 
                                        rotation=30, ha='right', fontsize=8)
            ax_contrast.set_ylabel("dielectric contrast (dimensionless)")
            ax_contrast.set_title("Foreign object vs. SUT dielectric contrast (materials-only, not derived from H_tf)")
            ax_contrast.grid(True, alpha=0.3, axis='y')
            ax_contrast.legend(fontsize=8)
            fig5.tight_layout()
       
        else:

            print("analyze(): no run has a usable dielectric contrast, so "
                  "the dielectric contrast figure is skipped.")

        plt.show()

    def analyze_doe_batch(self, doe_csv: str, baseline_folder: str,
                           manifest: str = None,
                           amp_noise_floor_db=_NOISE_FLOOR_UNSET,
                           phase_noise_floor_deg=_NOISE_FLOOR_UNSET,
                           smooth_window: int = 0,
                           response_metrics=('rms_dphase_max', 'contrast_real'),
                           factor_pairs: list = None,
                           output_csv: str = None,
                           run_anova: bool = True) -> pd.DataFrame:
        """Score and plot results of full-factorial DOE () by computing differential metrics against a baseline and optionally fitting ANOVA.

        Args:
            doe_csv (str): 
                DOE CSV file with the 'run' column (matched against the manifest) and the 6 factor columns. Relative paths resolve against this script's directory.
            baseline_folder (str): 
                Results folder used as the differential reference for every row.
            manifest (str): 
                Mode 2 manifest CSV
            amp_noise_floor_db, phase_noise_floor_deg (float): 
                See 'analyze()'
            smooth_window (int): 
                See 'analyze()'
            response_metrics (tuple of str): 
                Columns with their own main-effect/interaction figures (any '<name>_max'/'<name>_mean' of _compute_differential_metrics(), 'frobenius_rel', 'contrast_real' or 'contrast_complex_mag'; default: ('rms_dphase_max', 'contrast_real').
            factor_pairs (list of (str, str)): 
                Pairs for the 2-way interaction subplots (None: a fixed set -> diameter_mm x depth_mm,diameter_mm x material, depth_mm x moisture, position x material, material x heterogeneity,
                moisture x heterogeneity), restricted to present columns.
            output_csv (str): 
                Path of the augmented per-row results table
            run_anova (bool): 
                Also fit OLS + Type-II ANOVA (all factors + 2-way interactions) for every numeric response via
                fit_anova_all_responses(). Writes 'anova_<response>.csv', 'anova_summary_all_responses.csv', a p-value heatmap etc.(default: True).

        Returns:
            pandas.DataFrame: 
                One row per row in doe_csv: DOE columns, all computed metrics and 'response_status' ('OK' or the reason).

        Raises:
            FileNotFoundError: 
                If doe_csv, the manifest or the baseline's data.pkl don't exist.
            ValueError: 
                If doe_csv has no 'run' column.
        """
        # Set noise floors
        if amp_noise_floor_db is _NOISE_FLOOR_UNSET:
            amp_noise_floor_db = self.amp_noise_floor_db
        if phase_noise_floor_deg is _NOISE_FLOOR_UNSET:
            phase_noise_floor_deg = self.phase_noise_floor_deg

        # Read DOE CSV file
        doe_path = Path(doe_csv)
        if not doe_path.is_absolute():
            doe_path = Path(__file__).resolve().parent / doe_path
        if not doe_path.is_file():
            raise FileNotFoundError(
                f"analyze_doe_batch(): DOE CSV '{doe_path}' does not exist.")
        doe_df = pd.read_csv(doe_path)
        if 'run' not in doe_df.columns:
            raise ValueError(
                f"analyze_doe_batch(): '{doe_path}' has no 'run' column - "
                f"needed to match rows against the manifest.")

        # Determine manifest CSV file
        io_dir = Path(self.path_input_output_files)
        if manifest is not None:
            manifest_path = Path(manifest)
        else:
            preferred = io_dir / f"doe_manifest_{doe_path.stem}.csv"
            if preferred.is_file():
                manifest_path = preferred
            else:
                candidates = sorted(io_dir.glob('doe_manifest_*.csv'),
                                     key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    raise FileNotFoundError(
                        f"analyze_doe_batch(): no doe_manifest_*.csv found "
                        f"under '{io_dir}' - run MODE == 2 first, or pass "
                        f"manifest= explicitly.")
                manifest_path = candidates[0]
        print(f"analyze_doe_batch(): manifest '{manifest_path}'.")
        manifest_df = pd.read_csv(manifest_path)
        manifest_by_run = manifest_df.set_index(
            manifest_df['run'].astype(doe_df['run'].dtype))

        # Resolve manifest_path's grandparent to get the base directory for relative results_folder paths in the manifest
        manifest_base_dir = manifest_path.resolve().parent.parent

        # Load baseline data and compute bandwidth/delay resolution
        print(f"analyze_doe_batch(): loading baseline '{baseline_folder}'...")
        H_tf_base, f_fft_base, _, _, _ = self._load_run_data(baseline_folder)
        bandwidth_hz = float(np.asarray(f_fft_base).max() - 
                             np.asarray(f_fft_base).min())
        delay_res_ns = (1e9 / bandwidth_hz) if bandwidth_hz > 0 else None

        # Compute differential metrics for each DOE row, and record the reason if it cannot be scored
        rows_out = []
        for _, doe_row in doe_df.iterrows():

            run_label = doe_row['run']
            out_row = doe_row.to_dict()

            if run_label not in manifest_by_run.index:
                out_row['response_status'] = 'not in manifest (never attempted)'
            else:
                m = manifest_by_run.loc[run_label]
                if m['status'] != 'OK':
                    out_row['response_status'] = f"manifest status = {m['status']!r}"
                else:
                    try:
                        results_folder = Path(m['results_folder'])
                        if not results_folder.is_absolute():
                            results_folder = manifest_base_dir / results_folder
                        H_tf, f_fft, _, dielectric, _ = self._load_run_data(
                            results_folder, verbose=False)
                        metrics = self._compute_differential_metrics(
                            H_tf, f_fft, H_tf_base, f_fft_base,
                            smooth_window=smooth_window,
                            delay_res_ns=delay_res_ns,
                            amp_noise_floor_db=amp_noise_floor_db,
                            phase_noise_floor_deg=phase_noise_floor_deg)
                        out_row.update(metrics)

                        f_center = float((np.asarray(f_fft).min()
                                           + np.asarray(f_fft).max()) / 2)
                        contrast = self._dielectric_contrast(f_center, 
                                                             dielectric)
                        out_row['contrast_real'] = (contrast['contrast_real']
                                                     if contrast is not None else np.nan)
                        out_row['contrast_complex_mag'] = (
                            contrast['contrast_complex_mag']
                            if contrast is not None else np.nan)

                        out_row['response_status'] = 'OK'
                    except Exception as e:
                        print(f"analyze_doe_batch(): DOE run {run_label}."
                              f"Could not score: {e!r}")
                        out_row['response_status'] = f'ERROR: {e!r}'
            rows_out.append(out_row)

        # Export results to CSV
        results_df = pd.DataFrame(rows_out)
        n_ok = (results_df['response_status'] == 'OK').sum()
        print(f"analyze_doe_batch(): {n_ok}/{len(results_df)} row(s) scored.")
        out_csv_path = (Path(output_csv) if output_csv is not None
                         else doe_path.with_name(f"doe_results_{doe_path.stem}.csv"))
        results_df.to_csv(out_csv_path, index=False)
        print(f"analyze_doe_batch(): wrote '{out_csv_path}'.")

        scored_df = results_df[results_df['response_status'] == 'OK'].copy()
        if scored_df.empty:
            print("analyze_doe_batch(): no scored rows. Skipping factorial "
                  "plots and ANOVA.")
            return results_df
        
        # Fit optionally ANOVA. Plot main effects and 2-way interactions.
        if run_anova:
            try:
                print("analyze_doe_batch(): fitting ANOVA across every "
                      "response metric...")
                anova_summary_df = fit_anova_all_responses(
                    scored_df, output_dir=out_csv_path.parent)
                plot_combined_contrast_vs_rmsdphase(
                    scored_df, anova_summary_df, output_dir=out_csv_path.parent)
            except Exception as e:
                print(f"analyze_doe_batch(): ANOVA step failed ({e!r}) - "
                      f"skipping it; the scored results above ('{out_csv_path}') "
                      f"are unaffected. Re-run fit_anova_all_responses.py "
                      f"directly on that CSV to retry.")

        # Define factor columns
        factor_columns = [c for c in
                           ('diameter_mm', 'depth_mm', 
                            'position', 'material',
                            'heterogeneity', 'moisture')
                           if c in scored_df.columns]
        if factor_pairs is None:
            default_pairs = [('diameter_mm', 'depth_mm'), 
                             ('diameter_mm', 'material'),
                             ('depth_mm', 'moisture'), 
                             ('position', 'material'),
                             ('material', 'heterogeneity'), 
                             ('moisture', 'heterogeneity'),]
            factor_pairs = [p for p in default_pairs
                             if p[0] in factor_columns and p[1] in factor_columns]

        def _sort_key(v):
            """Sort key so by value."""
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v))

        # Draw for each two figures, response metric, main effects and 2-way interactions
        for response in response_metrics:

            if response not in scored_df.columns:
                print(f"analyze_doe_batch(): response metric {response!r} "
                      f"is not one of the computed columns. Skipping its "
                      f"figures.")
                continue

            # Plot main effects: one subplot per factor in factor_columns, mean at each level (with SEM error bars). Start wit setting up the Figure.
            ncols = 3
            nrows = int(np.ceil(len(factor_columns) / ncols))
            fig_main, axes_main = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            fig_main.canvas.manager.set_window_title(
                f"analyze_doe_batch(): Main effects - {response}")
            
            # Plot and set denomination for each factor subplot.
            axes_main_flat = axes_main.ravel()
            all_main_values = []
            for ax, factor in zip(axes_main_flat, factor_columns):
                grouped = scored_df.groupby(factor)[response].agg(['mean', 'std', 'count'])
                grouped = grouped.reindex(sorted(grouped.index, key=_sort_key))
                x = np.arange(len(grouped))
                sem_filled = (grouped['std'] / np.sqrt(grouped['count'])).fillna(0.0)
                ax.errorbar(x, grouped['mean'], yerr=sem_filled,
                             marker='o', capsize=4, color=_RESPONSE_COLOR_PRIMARY)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in grouped.index], rotation=20, ha='right')
                ax.set_xlabel(factor)
                ax.set_ylabel(PlotsMixin._response_label(response))
                ax.set_title(f"{factor} (n={int(grouped['count'].sum())})")
                ax.grid(True, alpha=0.3)
                all_main_values.extend((grouped['mean'] - sem_filled).tolist())
                all_main_values.extend((grouped['mean'] + sem_filled).tolist())
            for ax in axes_main_flat[:len(factor_columns)]:
                self._style_response_axis(ax, response, all_main_values)
            for ax in axes_main_flat[len(factor_columns):]:
                ax.axis('off')
            fig_main.suptitle(f"Single-factorial (main effects): {response}")
            fig_main.tight_layout()
            main_png_path = out_csv_path.parent / f"main_effects_{response}.png"
            _savefig(fig_main, main_png_path, dpi=300, bbox_inches='tight')
            print(f"analyze_doe_batch(): wrote '{main_png_path}'.")

            # Plot 2-way interactions (one subplot per pair in factor_pairs)
            pairs_here = [p for p in factor_pairs
                           if p[0] in factor_columns and p[1] in factor_columns]
            if pairs_here:
                ncols2 = min(3, len(pairs_here))
                nrows2 = int(np.ceil(len(pairs_here) / ncols2))
                fig_int, axes_int = plt.subplots(
                    nrows2, ncols2, figsize=(5.5 * ncols2, 4.5 * nrows2), squeeze=False)
                fig_int.canvas.manager.set_window_title(
                    f"analyze_doe_batch(): Interactions - {response}")

                trace_color_cycle = [_RESPONSE_COLOR_PRIMARY, 
                                     _RESPONSE_COLOR_SECONDARY]
                axes_int_flat = axes_int.ravel()
                all_int_values = []
                for ax, (fx, ftrace) in zip(axes_int_flat, pairs_here):
                    trace_vals = sorted(scored_df[ftrace].unique(), key=_sort_key)
                    for i, trace_val in enumerate(trace_vals):
                        sub = scored_df[scored_df[ftrace] == trace_val]
                        grouped = sub.groupby(fx)[response].mean()
                        grouped = grouped.reindex(sorted(grouped.index, key=_sort_key))
                        x = np.arange(len(grouped))
                        color = trace_color_cycle[i] if i < len(trace_color_cycle) else None
                        ax.plot(x, grouped.values, marker='o',
                                 label=f"{ftrace}={trace_val}", color=color)
                        ax.set_xticks(x)
                        ax.set_xticklabels([str(v) for v in grouped.index], rotation=20, ha='right')
                        all_int_values.extend(grouped.values.tolist())
                    ax.set_xlabel(fx)
                    ax.set_ylabel(PlotsMixin._response_label(response))
                    ax.set_title(f"{fx} x {ftrace}")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                for ax in axes_int_flat[:len(pairs_here)]:
                    self._style_response_axis(ax, response, all_int_values)
                for ax in axes_int_flat[len(pairs_here):]:
                    ax.axis('off')
                fig_int.suptitle(f"Multi-factorial (2-way interactions): {response}")
                fig_int.tight_layout()
                int_png_path = out_csv_path.parent / f"interactions_{response}.png"
                _savefig(fig_int, int_png_path, dpi=300, bbox_inches='tight')
                print(f"analyze_doe_batch(): wrote '{int_png_path}'.")
            else:
                print(f"analyze_doe_batch(): no usable factor pair for "
                      f"{response!r}. Skipping its interaction figure.")

        plt.show()
        return results_df


def _anova_rename_terms(terms):
    """Rewrite term labels related to ANOVA factor codes to their full factor names."""
    def _rename_one(term):
        return ':'.join(ANOVA_CODE_TO_FACTOR.get(part, part) for part in term.split(':'))
    return [_rename_one(t) for t in terms]


def anova_detect_response_metrics(results_df: pd.DataFrame) -> list:
    """Find every numeric column in DOE results table."""
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in _ANOVA_NON_RESPONSE_COLUMNS]


def fit_anova_all_responses(results_df: pd.DataFrame,
                             output_dir=None,
                             response_metrics: list = None,
                             factor_codes: list = ANOVA_FACTOR_CODES,
                             make_plots: bool = True) -> pd.DataFrame:
    """Fit an ANOVA model for each response metric in the DOE results table.

    Args:
        results_df (pandas.DataFrame): 
            DOE results table
        output_dir (str or Path): 
            Directory for CSV /PNG export (None: current directory).
        response_metrics (list of str): 
            Columns to fit.
        factor_codes (list of str): 
            The ±1 coded factor columns to fit ANOVA on (default: ANOVA_FACTOR_CODES).
        make_plots (bool): 
            Flag to export PNGs (default True).

    Returns:
        pandas.DataFrame: 
            Write summary. Constant responses are skipped.

    Raises:
        ValueError: 
            If results dataframe has no usable rows, one factor_codes is missing, no response columns are found, or all responses are constant.
    """
    # Get output directory and ensure it exists
    out_dir = Path(output_dir) if output_dir is not None else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raise error if any factor_codes are missing from results_df
    missing = [c for c in factor_codes if c not in results_df.columns]
    if missing:
        raise ValueError(
            f"fit_anova_all_responses(): results_df is missing factor "
            f"column(s) {missing}. Expected the DOE's own +-1 code "
            f"columns.")

    # Filter to only scored rows (response_status == 'OK') if that column exists, otherwise use all rows. Raise error if no usable rows remain.
    if 'response_status' in results_df.columns:
        fit_df = results_df[results_df['response_status'] == 'OK'].copy()
    else:
        fit_df = results_df.copy()
    if fit_df.empty:
        raise ValueError(
            "fit_anova_all_responses(): no usable ('OK') rows in "
            "results_df - nothing to fit.")

    # Determine response metrics to fit. Raise error if no response metrics remain.
    if response_metrics is None:
        response_metrics = anova_detect_response_metrics(fit_df)
    if not response_metrics:
        raise ValueError(
            "fit_anova_all_responses(): no response metric columns found "
            "(after excluding factor/identifier columns) - pass "
            "response_metrics= explicitly.")

    interaction_terms = [
        ':'.join(pair) for pair in itertools.combinations(factor_codes, 2)]
    formula_rhs = ' + '.join(factor_codes + interaction_terms)

    # Fit ANOVA for each response metric (skipping any that are constant or all-NaN)
    summary_rows = []
    n_fitted = 0
    for response in response_metrics:
        sub = fit_df[factor_codes + [response]].dropna(subset=[response])
        n_dropped = len(fit_df) - len(sub)
        if n_dropped:
            print(f"fit_anova_all_responses(): {response!r} has "
                  f"{n_dropped} NaN row(s) - dropped for this model only.")
        if sub.empty or sub[response].nunique() < 2:
            print(f"fit_anova_all_responses(): {response!r} has no usable "
                  f"variance across scored rows - skipping.")
            continue

        # Fit on a column renamed to 'response'
        model_df = sub.rename(columns={response: 'response'})
        model = ols(f"response ~ {formula_rhs}", data=model_df).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        anova_tbl = anova_tbl.reset_index().rename(columns={'index': 'term'})
        anova_tbl['term'] = _anova_rename_terms(anova_tbl['term'])
        anova_tbl.to_csv(out_dir / f"anova_{response}.csv", index=False)

        # Append summary rows for this response
        for _, row in anova_tbl.iterrows():
            summary_rows.append({
                'response': response,
                'term': row['term'],
                'sum_sq': row.get('sum_sq'),
                'df': row.get('df'),
                'F': row.get('F'),
                'p_value': row.get('PR(>F)'),
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'n_obs': int(model.nobs),
            })
        n_fitted += 1

    # Raise error if no response metrics were fitted
    if n_fitted == 0:
        raise ValueError(
            "fit_anova_all_responses(): every response metric was "
            "constant (or all-NaN) across scored rows and nothing was "
            "fitted.")

    # Write summary CSV and optionally make plots
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "anova_summary_all_responses.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"fit_anova_all_responses(): fit {n_fitted}/{len(response_metrics)} "
          f"response metric(s) ({len(fit_df)} scored row(s) each, minus "
          f"any per-response NaNs). Wrote '{summary_csv_path}' and "
          f"{n_fitted} anova_<response>.csv file(s) under '{out_dir}'.")

    # Optionally make plots
    if make_plots:
        _anova_plot_pvalue_heatmap(summary_df, out_dir)
        _anova_plot_r_squared_by_response(summary_df, out_dir)

    return summary_df


def _ff_tex_sci(x: float, nd: int = 2) -> str:
    """LaTeX scientific notation, e.g. 1.23e-05 -> '$1.23\\times10^{-5}$'."""
    if x == 0:
        return '$<10^{-300}$'
    e = int(np.floor(np.log10(abs(x))))
    return f'${x / 10**e:.{nd}f}\\times10^{{{e}}}$'


def _ff_fmt_p(p: float) -> str:
    """p-value for the LaTeX tables (scientific below 1e-3)."""
    return _ff_tex_sci(p) if p < 1e-3 else f'{min(p, 1.0):.3f}'


def _ff_fmt_F(F: float) -> str:
    """F statistic for the LaTeX tables."""
    if F < 1e-3:
        return '$<10^{-3}$'
    if F >= 1e4:
        return _ff_tex_sci(F)
    return f'{F:.3g}' if F < 100 else f'{F:.0f}'


def _ff_share_cell(share: float, p: float) -> str:
    """Table cell 'share (%)' with significance stars (*, **, *** for p < 0.05, 0.01, 0.001)."""
    stars = '' if (p is None or np.isnan(p)) else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''
    return (f'{share:.1f}' if share >= 1 else f'{share:.2f}') + (f'$^{{{stars}}}$' if stars else '')


def _ff_variance_shares(df: pd.DataFrame, response: str) -> tuple:
    """Type-II ANOVA of `response` with all main effects, 2- and 3-way interactions of the coded DOE factors.

    Args:
        df (pandas.DataFrame): Scored DOE rows with the columns of ANOVA_FACTOR_CODES.
        response (str): Response column.

    Returns:
        tuple: ({factor name / 'INT2' / 'INT3' / 'RES': (share of the total sum of squares in %, p or NaN), 'R2_2' / 'R2_3' /
            'adjR2_3': R^2 of the 2-way and 3-way model}, residual degrees of freedom of the 3-way model).
    """
    sub = df[ANOVA_FACTOR_CODES + [response]].dropna().rename(columns={response: 'y'})
    rhs = '(' + '+'.join(ANOVA_FACTOR_CODES) + ')'
    m3 = ols(f'y ~ {rhs}**3', data=sub).fit()
    a3 = sm.stats.anova_lm(m3, typ=2)
    m2 = ols(f'y ~ {rhs}**2', data=sub).fit()
    total = a3['sum_sq'].sum()
    shares = {ANOVA_CODE_TO_FACTOR[c]: (100 * a3.loc[c, 'sum_sq'] / total, a3.loc[c, 'PR(>F)']) for c in ANOVA_FACTOR_CODES}
    terms = [t for t in a3.index if t != 'Residual']
    for key, n_colons in (('INT2', 1), ('INT3', 2)):
        shares[key] = (100 * a3.loc[[t for t in terms if t.count(':') == n_colons], 'sum_sq'].sum() / total, np.nan)
    shares['RES'] = (100 * a3.loc['Residual', 'sum_sq'] / total, np.nan)
    shares.update(R2_2=m2.rsquared, R2_3=m3.rsquared, adjR2_3=m3.rsquared_adj)
    return shares, int(round(m3.df_resid))


def _ff_fo_only_shares(cell: pd.DataFrame, response: str) -> tuple:
    """Type-II ANOVA of `response` with the four foreign-object factors and their 2-way interactions, on the 16 runs of one
    sample recipe.

    Args:
        cell (pandas.DataFrame): DOE rows of one heterogeneity/moisture combination.
        response (str): Response column.

    Returns:
        tuple: ({factor / 'INT2' / 'RES': (share in %, p or NaN), 'R2', 'mean', 'median', 'lvl': {factor: {level: mean}}},
            residual degrees of freedom).
    """
    fo_factors = list(ANOVA_CODE_TO_FACTOR.values())[:4]
    m = ols(f"{response} ~ ({'+'.join(f'C({f})' for f in fo_factors)})**2", data=cell).fit()
    a = sm.stats.anova_lm(m, typ=2)
    total = a['sum_sq'].sum()
    shares = {f: (100 * a.loc[f'C({f})', 'sum_sq'] / total, a.loc[f'C({f})', 'PR(>F)']) for f in fo_factors}
    shares['INT2'] = (100 * a.loc[[i for i in a.index if ':' in i], 'sum_sq'].sum() / total, np.nan)
    shares['RES'] = (100 * a.loc['Residual', 'sum_sq'] / total, np.nan)
    shares.update(R2=m.rsquared, mean=cell[response].mean(), median=cell[response].median(),
                  lvl={f: cell.groupby(f)[response].mean().to_dict() for f in fo_factors})
    return shares, int(round(m.df_resid))


def _ff_half_normal_effects(df: pd.DataFrame, response: str) -> tuple:
    """All 63 factorial effects of the 2^6 design (mean at +1 minus mean at -1 of each contrast), their half-normal quantiles
    and Lenth's 95 % margin of error.

    Args:
        df (pandas.DataFrame): Scored DOE rows with the columns of ANOVA_FACTOR_CODES.
        response (str): Response column.

    Returns:
        tuple: (DataFrame 'term', 'order', 'effect', 'abs', 'q', 'sig' sorted by |effect|, margin of error).
    """
    y = df[response].values.astype(float)
    rows = []
    for order in range(1, len(ANOVA_FACTOR_CODES) + 1):
        for combo in itertools.combinations(ANOVA_FACTOR_CODES, order):
            contrast = np.prod([df[c].values for c in combo], axis=0)
            term = ':'.join(_FF_FACTOR_SHORT[ANOVA_CODE_TO_FACTOR[c]] for c in combo)
            rows.append((term, order, y[contrast > 0].mean() - y[contrast < 0].mean()))
    E = pd.DataFrame(rows, columns=['term', 'order', 'effect'])
    E['abs'] = E.effect.abs()
    E = E.sort_values('abs').reset_index(drop=True)
    n = len(E)
    E['q'] = sp.stats.norm.ppf(0.5 + 0.5 * (np.arange(1, n + 1) - 0.5) / n)
    # Lenth's pseudo standard error and margin of error (t quantile with n/3 degrees of freedom)
    s0 = 1.5 * np.median(E['abs'].values)
    pse = 1.5 * np.median(E['abs'].values[E['abs'].values < 2.5 * s0])
    me = sp.stats.t.ppf(0.975, n / 3) * pse
    E['sig'] = E['abs'] > me
    return E, me


def _ff_significant_pairs(summary: dict, responses=('contrast_real', 'rms_dphase_max')) -> list:
    """2-way interactions with p < 0.05 for any of `responses` in any study, as (x factor, trace factor, study initials);
    moisture is always the trace factor."""
    hits = {}
    for study, s in summary.items():
        sig = s[s.response.isin(list(responses)) & s.term.str.contains(':') & (s.p_value < 0.05)]
        for term in sig.term:
            hits.setdefault(frozenset(term.split(':')), set()).add(study[0])
    pairs = []
    for fx, ft in itertools.combinations(_FF_FACTOR_NAMES, 2):
        if frozenset((fx, ft)) in hits:
            who = ''.join(sorted(hits[frozenset((fx, ft))]))
            pairs.append((ft, fx, who) if fx == 'moisture' else (fx, ft, who))
    return pairs


def _ff_write_tex(lines: list, path: Path) -> None:
    """Write LaTeX table lines to `path`."""
    path.write_text('\n'.join(lines) + '\n')
    print(f"export_full_factorial_paper_assets(): wrote '{path}'.")


def export_full_factorial_paper_assets(study_dirs: dict, output_dir, doe_name: str = 'food_sim_doe_64run') -> dict:
    """Write the paper figures and LaTeX tables that compare the reflection- and transmission-based full-factorial DOE studies:
    fig_anova_variance_shares, fig_main_effects_shared, fig_interactions_shared, fig_half_normal (.png), table_headline.tex,
    table_summary_all_responses.tex, table_variance_shares.tex, table_fo_only.tex and paper_numbers.json (numbers quoted in the text).

    Args:
        study_dirs (dict): {'Reflection': folder, 'Transmission': folder} of both studies; each needs results_analysis/ with
            doe_results_<doe_name>.csv and anova_summary_all_responses.csv (MODE 3). Relative paths: to this file's folder.
        output_dir (str or Path): Output folder (created if missing).
        doe_name (str): Stem of the DOE CSV (default: 'food_sim_doe_64run').

    Returns:
        dict: The numbers written to paper_numbers.json, or None if a study has not been scored yet.
    """
    base = Path(__file__).resolve().parent
    data, summary = {}, {}
    for study in _FF_STUDIES:
        res_dir = Path(study_dirs[study])
        res_dir = (res_dir if res_dir.is_absolute() else base / res_dir) / 'results_analysis'
        doe_csv, summary_csv = res_dir / f'doe_results_{doe_name}.csv', res_dir / 'anova_summary_all_responses.csv'
        if not (doe_csv.is_file() and summary_csv.is_file()):
            print(f"export_full_factorial_paper_assets(): '{doe_csv.name}' or '{summary_csv.name}' missing in '{res_dir}' - "
                  f"score the {study.lower()} study with MODE 3 first. Nothing written.")
            return None
        df = pd.read_csv(doe_csv)
        data[study] = df[df['response_status'] == 'OK']
        summary[study] = pd.read_csv(summary_csv)
    out_dir = Path(output_dir)
    out_dir = out_dir if out_dir.is_absolute() else base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    factors = list(_FF_FACTOR_NAMES)
    level_order = {f: list(data[_FF_STUDIES[0]].sort_values(c).drop_duplicates(f)[f]) for c, f in ANOVA_CODE_TO_FACTOR.items()}
    numbers = {}

    # ---- Variance shares of the 3-way model: figure + table ----
    shares, df_res = {}, {}
    for study in _FF_STUDIES:
        for response, _ in _FF_VARIANCE_RESPONSES:
            shares[(study, response)], df_res[(study, response)] = _ff_variance_shares(data[study], response)
    _plot_ff_variance_shares(shares, out_dir)
    n_res = sorted(set(df_res.values()))
    L = [r'\begin{table*}[t]', r'\centering',
         r'\caption{Share of the total sum of squares (\%) per source in the $2^{6}$ full-factorial DOE for reflection (R) and '
         r'transmission (T), from the Type~II ANOVA with all main effects, two-way and three-way interactions '
         f'({"/".join(map(str, n_res))} residual degrees of freedom = four-way and higher interactions). Stars mark the significance '
         r'of the $F$-test of individual main effects: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. $R^2_{(2)}$: model with main '
         r'effects and two-way interactions only (Eq.~\ref{eq:doe_model}); $R^2_{(3)}$: including three-way interactions.}',
         r'\label{tab:anova_variance_shares}', r'\begin{tabular}{l' + 'cc' * len(_FF_VARIANCE_RESPONSES) + '}', r'\hline',
         ' & ' + ' & '.join(r'\multicolumn{2}{c}{\texttt{%s}}' % r.replace('_', r'\_') for r, _ in _FF_VARIANCE_RESPONSES) + r' \\',
         ' '.join(r'\cline{%d-%d}' % (2 + 2 * i, 3 + 2 * i) for i in range(len(_FF_VARIANCE_RESPONSES))),
         'Source & ' + ' & '.join('R & T' for _ in _FF_VARIANCE_RESPONSES) + r' \\', r'\hline']
    for source in factors + ['INT2', 'INT3', 'RES']:
        cells = [_ff_share_cell(*shares[(s, r)][source]) for r, _ in _FF_VARIANCE_RESPONSES for s in _FF_STUDIES]
        label = 'Residual (4-way+)' if source == 'RES' else _FF_SHARE_LABELS[source]
        L.append(label + ' & ' + ' & '.join(cells) + r' \\')
    L.append(r'\hline')
    for key, label in (('R2_2', '$R^2_{(2)}$'), ('R2_3', '$R^2_{(3)}$')):
        L.append(label + ' & ' + ' & '.join(f'{shares[(s, r)][key]:.3f}' for r, _ in _FF_VARIANCE_RESPONSES for s in _FF_STUDIES)
                 + r' \\')
    L += [r'\hline', r'\end{tabular}', r'\end{table*}']
    _ff_write_tex(L, out_dir / 'table_variance_shares.tex')
    numbers['shares'] = {f'{s}|{r}': {k: (v[0] if isinstance(v, tuple) else v) for k, v in d.items()}
                         for (s, r), d in shares.items()}

    # ---- Headline table: contrast_real and rms_dphase_max (2-way model of fit_anova_all_responses()) ----
    def _row(study, response, term):
        s = summary[study]
        return s[(s.response == response) & (s.term == term)].iloc[0]
    columns = (('Reflection', 'contrast_real'), ('Reflection', 'rms_dphase_max'), ('Transmission', 'rms_dphase_max'))
    terms = set()
    for study, response in columns:
        s = summary[study]
        terms |= set(s[(s.response == response) & (s.term != 'Residual') & (s.p_value < 0.05)].term)
    terms = sorted(terms, key=lambda t: (t.count(':'), [factors.index(a) for a in t.split(':')]))
    r2 = [f'{_row(s, r, "Residual").r_squared:.3f}' for s, r in columns]
    L = [r'\begin{table*}[t]', r'\centering',
         r'\caption{ANOVA results (Type~II, Eq.~\ref{eq:doe_model}) for the terms significant ($p<0.05$) in at least one of the '
         r'three columns: the materials-only dielectric contrast (identical for both studies) and \texttt{rms\_dphase\_max} for the '
         f'reflection and the transmission study. Residual: {int(_row(*columns[1], "Residual").df)} degrees of freedom (pooled '
         r'three-way-and-higher interactions, Section~\ref{subsec:methods_of_analysis}). '
         f'$R^2={r2[0]}$, ${r2[1]}$ and ${r2[2]}$, respectively.' + '}',
         r'\label{tab:anova_headline}', r'\begin{tabular}{lcccccc}', r'\hline',
         r' & \multicolumn{2}{c}{\texttt{contrast\_real}} & \multicolumn{2}{c}{\texttt{rms\_dphase\_max}, reflection} & '
         r'\multicolumn{2}{c}{\texttt{rms\_dphase\_max}, transmission} \\',
         r'\cline{2-3}\cline{4-5}\cline{6-7}', r'Term & $F$ & $p$ & $F$ & $p$ & $F$ & $p$ \\', r'\hline']
    for term in terms:
        cells = []
        for study, response in columns:
            row = _row(study, response, term)
            cells += [_ff_fmt_F(row.F), _ff_fmt_p(row.p_value)]
        L.append(' $\\times$ '.join(_FF_FACTOR_NAMES[x] for x in term.split(':')) + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\hline', r'\end{tabular}', r'\end{table*}']
    _ff_write_tex(L, out_dir / 'table_headline.tex')

    # ---- Summary over all response metrics ----
    def _top(study, response):
        s = summary[study]
        s = s[(s.response == response) & (s.term != 'Residual')].sort_values('p_value')
        t, p_material = s.iloc[0], s[s.term == 'material'].p_value.iloc[0]
        if t.p_value >= 0.05:
            return '--- (none)', t.r_squared, p_material
        name = '$\\times$'.join(_FF_FACTOR_ABBR[a] for a in t.term.split(':')) if ':' in t.term else _FF_FACTOR_NAMES[t.term]
        return name + ' (' + _ff_tex_sci(t.p_value, 1) + ')', t.r_squared, p_material
    responses = list(summary[_FF_STUDIES[0]].response.unique())
    L = [r'\begin{table*}[t]', r'\centering',
         f'\\caption{{ANOVA summary across all {len(responses)} response metrics for the reflection (R) and the transmission (T) '
         r'study: coefficient of determination of the model in Eq.~\ref{eq:doe_model}, most significant term ($p$-value), and '
         r'whether the foreign-object material reached significance ($p<0.05$). Het.: heterogeneity, Moist.: moisture, Mat.: '
         r'material, Diam.: diameter.}',
         r'\label{tab:anova_summary_all}', r'\begin{tabular}{lcl cl cc}', r'\hline',
         r'Response & $R^2$ (R) & Top term (R) & $R^2$ (T) & Top term (T) & Mat.\ sig.\ (R) & Mat.\ sig.\ (T)\\', r'\hline']
    n_material = {s: 0 for s in _FF_STUDIES}
    n_het_moist = {s: 0 for s in _FF_STUDIES}
    numbers['top_terms'] = {}
    for response in responses:
        top = {s: _top(s, response) for s in _FF_STUDIES}
        (tr, r2r, pr), (tt, r2t, pt) = top['Reflection'], top['Transmission']
        L.append(f'\\texttt{{{response.replace("_", chr(92) + "_")}}} & {r2r:.3f} & {tr} & {r2t:.3f} & {tt} & '
                 f'{"Yes" if pr < 0.05 else "No"} & {"Yes" if pt < 0.05 else "No"}\\\\')
        if not response.startswith('contrast'):
            for s in _FF_STUDIES:
                n_material[s] += int(top[s][2] < 0.05)
                ss = summary[s]
                ss = ss[(ss.response == response) & (ss.term != 'Residual')].sort_values('p_value')
                n_het_moist[s] += int('heterogeneity:moisture' in list(ss.term.iloc[:2]))
        numbers['top_terms'][response] = {'R': (tr, r2r), 'T': (tt, r2t)}
    L += [r'\hline', r'\end{tabular}', r'\end{table*}']
    _ff_write_tex(L, out_dir / 'table_summary_all_responses.tex')
    numbers.update(nmat=n_material, nhetmoist_top2=n_het_moist, n_responses=len(responses))

    # ---- Foreign-object-only ANOVA in the baseline sample recipe of each study ----
    base_cell = {'Reflection': ('Homogeneous', 'Low'), 'Transmission': ('Homogeneous', 'High')}
    fo_responses = (('rms_dphase_max', r'\texttt{rms\_dphase\_max}'), ('frobenius_rel', r'\texttt{frobenius\_rel}'))
    fo, fo_df_res = {}, set()
    for study, (het, moist) in base_cell.items():
        d = data[study]
        cell = d[(d.heterogeneity == het) & (d.moisture == moist)]
        for response, _ in fo_responses:
            fo[(study, response)], n = _ff_fo_only_shares(cell, response)
            fo_df_res.add(n)
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Foreign-object-only sensitivity in the baseline-matched sample recipe (homogeneous sample, low moisture for '
         r'reflection, high moisture for transmission; $2^{4}=16$ runs): share of the total sum of squares (\%) of the Type~II '
         r'ANOVA with the four foreign-object factors and their two-way interactions '
         f'({"/".join(map(str, sorted(fo_df_res)))} residual degrees of freedom). ' + r'Stars: $^{*}p<0.05$, $^{**}p<0.01$.}',
         r'\label{tab:fo_only}', r'\begin{tabular}{lcccc}', r'\hline',
         r' & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % label for _, label in fo_responses) + r'\\',
         r'\cline{2-3}\cline{4-5}', r'Source & R & T & R & T\\', r'\hline']
    for key, label in [(f, _FF_FACTOR_NAMES[f]) for f in factors[:4]] + [('INT2', '2-way interactions'),
                                                                         ('RES', 'Residual (3-/4-way)')]:
        L.append(label + ' & ' + ' & '.join(_ff_share_cell(*fo[(s, r)][key]) for r, _ in fo_responses for s in _FF_STUDIES) + r'\\')
    L.append(r'\hline')
    L.append(r'$R^2$ & ' + ' & '.join(f'{fo[(s, r)]["R2"]:.3f}' for r, _ in fo_responses for s in _FF_STUDIES) + r'\\')
    L.append(r'Mean response & ' + ' & '.join((f'{fo[(s, r)]["mean"]:.1f}$^\\circ$' if r == 'rms_dphase_max'
                                                else f'{fo[(s, r)]["mean"]:.3f}') for r, _ in fo_responses for s in _FF_STUDIES)
             + r'\\')
    L += [r'\hline', r'\end{tabular}', r'\end{table}']
    _ff_write_tex(L, out_dir / 'table_fo_only.tex')
    numbers['fo_only'] = {f'{s}|{r}': {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
                          for (s, r), d in fo.items()}

    # ---- Main effects, significant 2-way interactions and half-normal plots ----
    _plot_ff_main_effects(data, level_order, out_dir)
    pairs = _ff_significant_pairs(summary)
    numbers['sig_pairs'] = pairs
    _plot_ff_interactions(data, pairs, level_order, out_dir)
    effects, numbers['halfnormal'] = {}, {}
    for response, _ in _FF_HALF_NORMAL_RESPONSES:
        for study in _FF_STUDIES:
            E, me = _ff_half_normal_effects(data[study], response)
            effects[(response, study)] = (E, me)
            numbers['halfnormal'][f'{study}|{response}'] = {
                'ME': me, 'n_beyond': int(E.sig.sum()),
                'top': [(t.term, float(t.effect)) for _, t in E.sort_values('abs', ascending=False).head(8).iterrows()],
                'max_order_beyond': int(E[E.sig].order.max()) if E.sig.any() else 0}
    _plot_ff_half_normal(effects, out_dir)

    json_path = out_dir / 'paper_numbers.json'
    with open(json_path, 'w') as fh:
        json.dump(numbers, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"export_full_factorial_paper_assets(): wrote '{json_path}'.")
    return numbers


def _mimo_compute_combined_channel_metrics(H_ratio_combined: np.ndarray, 
                                           f_fft: np.ndarray,
                                           delay_res_ns: float,
                                           amp_noise_floor_db: float,phase_noise_floor_deg: float) -> dict:
    """Compute differential detectability metrics of one MIMO channel ratio spectrum that has already been averaged across channels.
    
    Args:
        H_ratio_combined (np.ndarray):
            Complex array of the combined H_ratio across N channels, shape (n_freq,).
        f_fft (np.ndarray):
            Frequency array corresponding to H_ratio_combined, shape (n_freq,).
        delay_res_ns (float):
            Delay resolution threshold in nanoseconds. If None, the delay metric is not scored.
        amp_noise_floor_db (float):
            Amplitude noise floor threshold in dB. If None, the amplitude metric is not scored.
        phase_noise_floor_deg (float):
            Phase noise floor threshold in degrees. If None, the phase metric is not scored.

    Returns:
        dict:
            Dictionary containing the computed metrics:
            - 'mean_dtau': Mean group delay in nanoseconds.
            - 'rms_dtau': RMS group delay in nanoseconds.
            - 'mean_dmag': Mean delta magnitude in dB.
            - 'rms_dmag': RMS delta magnitude in dB.
            - 'mean_dphase': Mean delta phase in degrees.
            - 'rms_dphase': RMS delta phase in degrees. 
    """

    # Compute differential metrics
    dmag_db = 20 * np.log10(np.abs(H_ratio_combined))
    dphase_rad = np.unwrap(np.angle(H_ratio_combined))
    dphase_deg = np.degrees(dphase_rad)
    dtau_ns = AnalysisMixin._group_delay_ns_from_phase(dphase_rad, f_fft)

    out = {'mean_dtau': float(np.mean(dtau_ns)),
           'rms_dtau': float(np.sqrt(np.mean(dtau_ns ** 2))),
           'mean_dmag': float(np.mean(dmag_db)),
           'rms_dmag': float(np.sqrt(np.mean(dmag_db ** 2))),
           'mean_dphase': float(np.mean(dphase_deg)),
           'rms_dphase': float(np.sqrt(np.mean(dphase_deg ** 2))),}

    # Score against thresholds using weights if provided for one MIMO channel
    n_exceed = 0
    weighted_score = 0.0
    if delay_res_ns is not None:
        n_exceed += int(out['rms_dtau'] > delay_res_ns)
        weighted_score += max(0.0, out['rms_dtau'] / delay_res_ns - 1.0)
    if amp_noise_floor_db is not None:
        n_exceed += int(out['rms_dmag'] > amp_noise_floor_db)
        weighted_score += max(0.0, out['rms_dmag'] / amp_noise_floor_db - 1.0)
    if phase_noise_floor_deg is not None:
        n_exceed += int(out['rms_dphase'] > phase_noise_floor_deg)
        weighted_score += max(0.0, out['rms_dphase'] / phase_noise_floor_deg - 1.0)
    out['n_exceed'] = n_exceed
    out['weighted_score'] = weighted_score

    return out


def _mimo_channel_sensitivity_for_run(H_ratio_flat: np.ndarray, 
                                      f_fft: np.ndarray,
                                      channel_counts, 
                                      n_draws: int,
                                      delay_res_ns: float, 
                                      amp_noise_floor_db: float,
                                      phase_noise_floor_deg: float,
                                      rng: np.random.Generator) -> list:
    """Score random channel subsets of one run's H_ratio.

    Args:
        H_ratio_flat (np.ndarray):
            Complex array of shape (n_channels_total, n_freq) representing one run's H_ratio
        f_fft (np.ndarray):
            Frequency array
        channel_counts (iterable of int):
            Iterable of channel counts to evaluate
        n_draws (int):
            Number of random channel subsets to draw for each N < n_total. For N == n_total, only the single full combination is used.
        delay_res_ns (float):
            Delay resolution threshold in nanoseconds. If None, the delay metric is not scored.
        amp_noise_floor_db (float):
            Amplitude noise floor threshold in dB. If None, the amplitude metric is not scored.
        phase_noise_floor_deg (float):
            Phase noise floor threshold in degrees. If None, the phase metric is not scored.
        rng (np.random.Generator):
            Random number generator for reproducibility.

    Returns:
        list of dict:
            Each dict contains the following keys:
            - 'n_channels': Number of channels combined
            - 'n_draws_used': Number of random draws used for this channel count
            - '<metric>_mean': Mean of the metric across draws
            - '<metric>_std': Standard deviation of the metric across draws
    """
    n_total = H_ratio_flat.shape[0]
    rows = []
    for n_ch in channel_counts:
        
        if n_ch > n_total:
            continue
        draws_used = 1 if n_ch == n_total else n_draws
        per_draw_metrics = {name: [] for name in MIMO_RESPONSE_CHOICES}

        # Pick random channel subsets and compute metrics for each draw
        for _ in range(draws_used):
            if n_ch == n_total:
                idx = np.arange(n_total)
            else:
                idx = rng.choice(n_total, size=n_ch, replace=False)
            combined = H_ratio_flat[idx, :].mean(axis=0)
            metrics = _mimo_compute_combined_channel_metrics(combined, 
                                                        f_fft, 
                                                        delay_res_ns, amp_noise_floor_db,phase_noise_floor_deg)
            for name in MIMO_RESPONSE_CHOICES:
                per_draw_metrics[name].append(metrics[name])
        row = {'n_channels': n_ch, 'n_draws_used': draws_used}

        # Compute mean and std for each metric across draws
        for name in MIMO_RESPONSE_CHOICES:
            vals = np.asarray(per_draw_metrics[name], dtype=float)
            row[f'{name}_mean'] = float(np.mean(vals))
            row[f'{name}_std'] = float(np.std(vals))
        rows.append(row)

    return rows


def analyze_mimo_channel_sensitivity(
        study_dir: str = 'studies/reflection-based full-factorial sensitivity study',
        output_dir: str = 'studies/mimo sensitivity study',
        doe_results_csv: str = None,
        manifest: str = None,
        baseline_folder: str = None,
        channel_counts=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25),
        n_draws: int = 200,
        response_metric: str = 'rms_dphase',
        amp_noise_floor_db=_NOISE_FLOOR_UNSET,
        phase_noise_floor_deg=_NOISE_FLOOR_UNSET,
        seed: int = 0) -> pd.DataFrame:
    """Determines whether the foreign-object detectability can be improved as more MIMO TX-RX channels are coherently combined, based on the DOE results. For each N in 'channel_counts', random channel subsets are drawn and coherently averaged, and the detectability metrics are computed. The mean and standard deviation of each metric across draws are reported.

    Args:
        study_dir (str): 
            Folder holding input_output_files/ (runs + manifest) and results_analysis/doe_results_*.csv.
        output_dir (str): 
            Output folder for the CSV and PNG files (default 'studies/mimo sensitivity study').
        doe_results_csv (str): 
            Scored DOE table (None: newest doe_results_*.csv under `study_dir`/results_analysis).
        manifest (str): 
            Mode 2 manifest CSV file (None: newest doe_manifest_*.csv under `study_dir`/input_output_files).
        baseline_folder (str): 
            'No anomaly' reference results folder (None = newest folder under `study_dir`/input_output_files ending in '_base').
        channel_counts (list of int): 
            Channel counts to evaluate (default: (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25)).
        n_draws (int): 
            Random channel subsets per (run, N) for N < 25 (default 200).
        response_metric (str): 
            MIMO_RESPONSE_CHOICES entry plotted against N in the summary PNG (default 'rms_dphase').
        amp_noise_floor_db, phase_noise_floor_deg (float): 
            Detectability thresholds for n_exceed/weighted_score (None: drops that channel, other values set the threshold).
        seed (int): 
            Random seed for channel-subset sampling (default 0).

    Returns:
        pandas.DataFrame: 
            One row per (DOE run, N): run, the DOE label columns present (material, moisture, heterogeneity, diameter_mm,
            depth_mm, position, contrast_real, contrast_complex_mag), n_channels, n_draws_used and every MIMO_RESPONSE_CHOICES metric's
            _mean/_std across draws.

    Raises:
        FileNotFoundError: 
            If 'doe_results_csv' can't be resolved.
    """
    if amp_noise_floor_db is _NOISE_FLOOR_UNSET:
        amp_noise_floor_db = 0.2
    if phase_noise_floor_deg is _NOISE_FLOOR_UNSET:
        phase_noise_floor_deg = 1.3

    # Get path or create output directory
    study_path = Path(study_dir)
    io_dir = study_path / 'input_output_files'
    results_dir = study_path / 'results_analysis'
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve DOE results CSV file
    if doe_results_csv is not None:
        doe_results_path = Path(doe_results_csv)
    else:
        candidates = sorted(results_dir.glob('doe_results_*.csv'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_sensitivity(): no doe_results_*.csv "
                f"found under '{results_dir}' - run MODE == 3 first, or "
                f"pass doe_results_csv= explicitly.")
        doe_results_path = candidates[0]
    print(f"analyze_mimo_channel_sensitivity(): DOE results table "
          f"'{doe_results_path}'.")

    # Read DOE results and filter to 'OK' rows
    doe_results = pd.read_csv(doe_results_path)
    doe_results = doe_results[doe_results['response_status'] == 'OK'].copy()

    # Resolve manifest CSV file and read it, indexed by 'run'
    if manifest is not None:
        manifest_path = Path(manifest)
    else:
        candidates = sorted(io_dir.glob('doe_manifest_*.csv'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_sensitivity(): no doe_manifest_*.csv "
                f"found under '{io_dir}' - run MODE == 2 first, or pass "
                f"manifest= explicitly.")
        manifest_path = candidates[0]
    manifest_df = pd.read_csv(manifest_path)
    manifest_by_run = manifest_df.set_index(
        manifest_df['run'].astype(doe_results['run'].dtype))
    
    # Get the base directory (grandparent of manifest_path).
    manifest_base_dir = manifest_path.resolve().parent.parent

    # Load baseline data and compute bandwidth/delay resolution
    if baseline_folder is not None:
        baseline_path = Path(baseline_folder)
    else:
        base_candidates = sorted(
            [p for p in io_dir.iterdir() if p.is_dir() and p.name.endswith('_base')])
        if not base_candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_sensitivity(): no '*_base' folder "
                f"found under '{io_dir}' - pass baseline_folder= "
                f"explicitly.")
        baseline_path = base_candidates[-1]
    print(f"analyze_mimo_channel_sensitivity(): baseline '{baseline_path}'.")
    
    # Load baseline H_tf and f_fft, compute bandwidth and delay resolution
    H_tf_base, f_fft, _, _, _ = AnalysisMixin._load_run_data(baseline_path, verbose=False)
    bandwidth_hz = float(np.asarray(f_fft).max() - np.asarray(f_fft).min())
    delay_res_ns = (1e9 / bandwidth_hz) if bandwidth_hz > 0 else None

    # Determine which label columns are present in the DOE results table
    label_cols = [c for c in
                  ('material', 'moisture', 'heterogeneity', 'diameter_mm',
                   'depth_mm', 'position', 'contrast_real', 'contrast_complex_mag')
                  if c in doe_results.columns]

    # Score each DOE run's H_tf against the baseline, for each channel count and random draw, and record the mean/std of each metric across draws.
    rng = np.random.default_rng(seed)
    all_rows = []
    n_scored = 0
    for _, doe_row in doe_results.iterrows():
        run_label = doe_row['run']
        if run_label not in manifest_by_run.index:
            continue
        m = manifest_by_run.loc[run_label]
        try:
            results_folder = Path(m['results_folder'])
            if not results_folder.is_absolute():
                results_folder = manifest_base_dir / results_folder
            H_tf, run_f_fft, _, _, _ = AnalysisMixin._load_run_data(
                results_folder, verbose=False)
            if H_tf.shape != H_tf_base.shape:
                print(f"analyze_mimo_channel_sensitivity(): run {run_label} "
                      f"- shape {H_tf.shape} != baseline {H_tf_base.shape}, "
                      f"skipping.")
                continue
            H_ratio_flat = H_tf / H_tf_base

            per_n_rows = _mimo_channel_sensitivity_for_run(
                H_ratio_flat, run_f_fft, channel_counts, n_draws,
                delay_res_ns, amp_noise_floor_db, phase_noise_floor_deg, rng)
            for row in per_n_rows:
                row['run'] = run_label
                for c in label_cols:
                    row[c] = doe_row[c]
                all_rows.append(row)
            n_scored += 1

        except Exception as e:
            print(f"analyze_mimo_channel_sensitivity(): run {run_label}. "
                  f"Could not score: {e!r}")

    print(f"analyze_mimo_channel_sensitivity(): {n_scored}/{len(doe_results)} "
          f"run(s) scored across {len(channel_counts)} channel count(s).")

    # Export results to CSV
    results_df = pd.DataFrame(all_rows)
    ordered_cols = (['run'] + label_cols + ['n_channels', 'n_draws_used'] +
                     [f'{name}_{stat}' for name in MIMO_RESPONSE_CHOICES
                      for stat in ('mean', 'std')])
    results_df = results_df[[c for c in ordered_cols if c in results_df.columns]]
    out_csv_path = out_dir / 'mimo_channel_sensitivity.csv'
    results_df.to_csv(out_csv_path, index=False)
    print(f"analyze_mimo_channel_sensitivity(): wrote '{out_csv_path}' "
          f"({len(results_df)} rows).")

    # Generate summary figure for the response metric
    if results_df.empty:
        print("analyze_mimo_channel_sensitivity(): no run scored - "
              "skipping the summary figure.")
        return results_df
    _plot_mimo_channel_sensitivity(results_df, 
                                   response_metric, 
                                   n_scored, 
                                   out_dir)

    return results_df


def analyze_mimo_channel_similarity(
        study_dir: str = 'studies/reflection-based full-factorial sensitivity study',
        output_dir: str = 'studies/mimo sensitivity study',
        manifest: str = None,
        baseline_folder: str = None) -> pd.DataFrame:
    """Explains the dependence of the coherently averaged MIMO response towards N - how similar are the per-channel phase deviations of each DOE run are.
    
    Args:
        study_dir (str): 
            Folder holding input_output_files/ (runs + manifest). 
        output_dir (str): 
            Output folder of the CSV (default 'studies/mimo sensitivity study').
        manifest (str): 
            Mode 2 manifest CSV (None: newest doe_manifest_*.csv under `study_dir`/input_output_files).
        baseline_folder (str): 
            'No anomaly' reference folder (None: newest folder under `study_dir`/input_output_files ending in '_base').

    Returns:
        pandas.DataFrame: 
            One row per scored run with the manifest's factor columns and the quantities above.

    Raises:
        FileNotFoundError: If no manifest or baseline folder is found.
    """

    # Get path and create directory
    io_dir = Path(study_dir) / 'input_output_files'
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine manifest and baseline paths
    if manifest is not None:
        manifest_path = Path(manifest)
    else:
        candidates = sorted(io_dir.glob('doe_manifest_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"analyze_mimo_channel_similarity(): no doe_manifest_*.csv found under '{io_dir}'.")
        manifest_path = candidates[0]

    # Determine baseline path
    if baseline_folder is not None:
        baseline_path = Path(baseline_folder)
    else:
        base_candidates = sorted(p for p in io_dir.iterdir() if p.is_dir() and p.name.endswith('_base'))
        if not base_candidates:
            raise FileNotFoundError(f"analyze_mimo_channel_similarity(): no '*_base' folder found under '{io_dir}'.")
        baseline_path = base_candidates[-1]
    print(f"analyze_mimo_channel_similarity(): manifest '{manifest_path}', baseline '{baseline_path}'.")

    # Read manifest file and get baseline data
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df['status'] == 'OK']
    manifest_base_dir = manifest_path.resolve().parent.parent
    H_tf_base, _, _, _, _ = AnalysisMixin._load_run_data(baseline_path, 
                                                         verbose=False)

    # Prepare label columns and initialize rows list for results
    label_cols = [c for c in ('material', 
                              'moisture', 
                              'heterogeneity', 
                              'diameter_mm', 
                              'depth_mm', 
                              'position')
                  if c in manifest_df.columns]
    rows = []

    # Iterate over each row in the manifest and analyze the corresponding run
    for _, m in manifest_df.iterrows():
        try:
            # Get the results folder for the current run and ensure it is an absolute path
            results_folder = Path(str(m['results_folder']).replace('\\', '/'))
            if not results_folder.is_absolute():
                results_folder = manifest_base_dir / results_folder
            H_tf, _, _, _, _ = AnalysisMixin._load_run_data(results_folder, 
                                                            verbose=False)

            # Check if the shape of the current run's channel matrix matches the baseline's shape
            if H_tf.shape != H_tf_base.shape:
                print(f"analyze_mimo_channel_similarity(): run {m['run']} - shape {H_tf.shape} != baseline "
                      f"{H_tf_base.shape}, skipping.")
                continue

            # Compute the ratio of the current run's channel matrix to the baseline's channel matrix
            H_ratio = H_tf / H_tf_base

            # Compute the phase difference and related metrics for the current run
            dphi = np.degrees(np.unwrap(np.angle(H_ratio), axis=1))
            rms_c = np.sqrt(np.mean(dphi ** 2, axis=1))
            dphi_coherent = np.degrees(np.unwrap(np.angle(H_ratio.mean(axis=0))))

            # Compute the cosine similarity between the normalized phase difference vectors
            unit = dphi / np.linalg.norm(dphi, axis=1, keepdims=True)
            cos = (unit @ unit.T)[np.triu_indices(dphi.shape[0], 1)]
            row = {'run': m['run'], **{c: m[c] for c in label_cols},
                   'rms_dphase_single': float(rms_c.mean()),
                   'rms_dphase_coherent': float(np.sqrt(np.mean(dphi_coherent ** 2))),
                   'cos_mean': float(cos.mean()),
                   'cos_neg_share': float((cos < 0).mean()),
                   'same_sign_share': float((np.sign(dphi.mean(axis=1)) == np.sign(dphi.mean())).mean()),
                   'cv_rms': float(rms_c.std() / rms_c.mean())}
            row['retained'] = (row['rms_dphase_coherent'] / 
                               row['rms_dphase_single'])
            rows.append(row)

        except Exception as e:

            print(f"analyze_mimo_channel_similarity(): run {m['run']}. Could not score: {e!r}")

    # Convert the list of result rows into a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(rows)
    out_csv_path = out_dir / 'mimo_channel_similarity.csv'
    results_df.to_csv(out_csv_path, index=False)
    print(f"analyze_mimo_channel_similarity(): {len(results_df)}/{len(manifest_df)} run(s) scored, wrote '{out_csv_path}'.")

    # If the results DataFrame is empty, return it immediately
    if results_df.empty:
        return results_df

    # Print summary statistics for the relevant columns across all runs and grouped by metrics
    cols = ['rms_dphase_single', 
            'rms_dphase_coherent', 
            'retained', 
            'cos_mean', 
            'cos_neg_share', 
            'same_sign_share', 
            'cv_rms']
    print("analyze_mimo_channel_similarity(): mean over all runs:\n" + results_df[cols].mean().round(3).to_string())
    for group in (['position'], ['heterogeneity', 'moisture']):
        if all(g in results_df.columns for g in group):
            print(f"analyze_mimo_channel_similarity(): mean per {' / '.join(group)}:\n"
                  + results_df.groupby(group)[['rms_dphase_single', 'retained', 'cos_mean', 'cos_neg_share']].mean()
                  .round(3).to_string())
    
    return results_df


def _mimo_precompute_channel_subsets(n_total: int, 
                                     channel_counts, 
                                     n_draws: int,
                                     rng: np.random.Generator) -> dict:
    """Precompute the same n_draws random N-of-n_total channel-index subsets (without replacement) for every DOE run.

    Args:
        n_total (int): 
            Total channels available
        channel_counts (iterable of int): 
            Channel counts to generate subsets for
        n_draws (int): 
            Random subsets per N < n_total.
        rng (np.random.Generator): Source of randomness.

    Returns:
        dict: {n_channels: [index_array, ...]}, one list of 1D int index arrays per valid N.
    
    Raises:
        None
    """

    # Precompute the random channel subsets for each requested channel count.
    subsets_by_n = {}
    for n_ch in channel_counts:
        if n_ch > n_total:
            continue
        if n_ch == n_total:
            subsets_by_n[n_ch] = [np.arange(n_total)]
        else:
            subsets_by_n[n_ch] = [rng.choice(n_total, 
                                             size=n_ch, 
                                             replace=False)
                                   for _ in range(n_draws)]
            
    return subsets_by_n


def _mimo_channel_draws_for_run(H_ratio_flat: np.ndarray, f_fft: np.ndarray,
                                 subsets_by_n: dict, delay_res_ns: float,
                                 amp_noise_floor_db: float,
                                 phase_noise_floor_deg: float) -> list:
    """Like _mimo_channel_sensitivity_for_run(), but scores every draw of the shared subsets (by n) against this run's H_ratio. Returns one row per (n_channels, draw_index) instead of per-run mean/std.

    Args:
        H_ratio_flat (np.ndarray): 
            One run's H_tf / H_tf_base (TX-major flattened).
        f_fft (np.ndarray): 
            Frequency bins
        subsets_by_n (dict): 
            {n_channels: [index_array, ...]} from _mimo_precompute_channel_subsets().
        delay_res_ns, amp_noise_floor_db, phase_noise_floor_deg (float):
            Thresholds as in _mimo_compute_combined_channel_metrics().

    Returns:
        list of dict: 
            One per (n_channels, draw_index) with keys 'n_channels', 'draw_index' and every MIMO_RESPONSE_CHOICES metric's single-draw value.
    Raises:
        None
    """
    # Iterate over each channel count and its corresponding list of random subsets and compute the combined channel metrics for each draw.
    rows = []
    for n_ch, idx_list in subsets_by_n.items():
        for draw_idx, idx in enumerate(idx_list):
            combined = H_ratio_flat[idx, :].mean(axis=0)
            metrics = _mimo_compute_combined_channel_metrics(combined, 
                                                        f_fft, 
                                                        delay_res_ns, amp_noise_floor_db,phase_noise_floor_deg)
            row = {'n_channels': n_ch, 'draw_index': draw_idx}
            row.update(metrics)
            rows.append(row)

    return rows


def analyze_mimo_channel_draw_surface(
        study_dir: str = 'studies/reflection-based full-factorial sensitivity study',
        output_dir: str = 'studies/mimo sensitivity study',
        doe_results_csv: str = None,
        manifest: str = None,
        baseline_folder: str = None,
        channel_counts=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25),
        n_draws: int = 200,
        response_metric: str = 'rms_dphase',
        amp_noise_floor_db=_NOISE_FLOOR_UNSET,
        phase_noise_floor_deg=_NOISE_FLOOR_UNSET,
        seed: int = 0) -> pd.DataFrame:
    """Draw MIMO channel draw surface to answer, if an N-of-25 channel subset matters or only the count N (flat surface along the draw axis means only N matters).

    Args:
        study_dir, output_dir, doe_results_csv, manifest, baseline_folder: 
            See analyze_mimo_channel_sensitivity().
        channel_counts (list of int): 
            Channel counts to evaluate (default (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25)).
        n_draws (int): 
            Shared random subsets per N < N_max (default 200; N == N_max uses the single full combination).
        response_metric (str): 
            MIMO_RESPONSE_CHOICES entry plotted as surface height/color (default 'rms_dphase').
        amp_noise_floor_db, phase_noise_floor_deg (float): 
            As in analyze_mimo_channel_sensitivity() (None drops the channel, otherwise default values).
        seed (int): 
            Seed of the shared subset draws (default 0; independent of analyze_mimo_channel_sensitivity()).

    Returns:
        pandas.DataFrame: 
            One row per (n_channels, draw_index) with every MIMO_RESPONSE_CHOICES metric's mean/std across DOE runs and
            n_runs_scored.

    Raises:
        FileNotFoundError: If `doe_results_csv`/`manifest`/`baseline_folder` can't be resolved or don't exist.
    """
    if amp_noise_floor_db is _NOISE_FLOOR_UNSET:
        amp_noise_floor_db = 0.2
    if phase_noise_floor_deg is _NOISE_FLOOR_UNSET:
        phase_noise_floor_deg = 1.3

    # Get path or create output directory
    study_path = Path(study_dir)
    io_dir = study_path / 'input_output_files'
    results_dir = study_path / 'results_analysis'
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve DOE results CSV file
    if doe_results_csv is not None:
        doe_results_path = Path(doe_results_csv)
    else:
        candidates = sorted(results_dir.glob('doe_results_*.csv'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_draw_surface(): no doe_results_*.csv "
                f"found under '{results_dir}' - run MODE == 3 first, or "
                f"pass doe_results_csv= explicitly.")
        doe_results_path = candidates[0]
    print(f"analyze_mimo_channel_draw_surface(): DOE results table "
          f"'{doe_results_path}'.")

    # Read DOE results and filter to 'OK' rows
    doe_results = pd.read_csv(doe_results_path)
    doe_results = doe_results[doe_results['response_status'] == 'OK'].copy()

    # Resolve manifest CSV file and read it, indexed by 'run'
    if manifest is not None:
        manifest_path = Path(manifest)
    else:
        candidates = sorted(io_dir.glob('doe_manifest_*.csv'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_draw_surface(): no doe_manifest_*.csv "
                f"found under '{io_dir}' - run MODE == 2 first, or pass "
                f"manifest= explicitly.")
        manifest_path = candidates[0]

    # Resolve the manifest path to the most recent candidate if not explicitly provided.
    manifest_df = pd.read_csv(manifest_path)
    manifest_by_run = manifest_df.set_index(
        manifest_df['run'].astype(doe_results['run'].dtype))
    
    # Same base-dir derivation as analyze_mimo_channel_sensitivity().
    manifest_base_dir = manifest_path.resolve().parent.parent

    # Load baseline data and compute bandwidth/delay resolution
    if baseline_folder is not None:
        baseline_path = Path(baseline_folder)
    else:
        base_candidates = sorted(
            [p for p in io_dir.iterdir() if p.is_dir() and p.name.endswith('_base')])
        if not base_candidates:
            raise FileNotFoundError(
                f"analyze_mimo_channel_draw_surface(): no '*_base' folder "
                f"found under '{io_dir}' - pass baseline_folder= "
                f"explicitly.")
        baseline_path = base_candidates[-1]
    print(f"analyze_mimo_channel_draw_surface(): baseline '{baseline_path}'.")
    
    # Load baseline H_tf and f_fft, compute bandwidth and delay resolution
    H_tf_base, f_fft, _, _, _ = AnalysisMixin._load_run_data(baseline_path, 
                                                             verbose=False)
    bandwidth_hz = float(np.asarray(f_fft).max() - 
                         np.asarray(f_fft).min())
    delay_res_ns = (1e9 / bandwidth_hz) if bandwidth_hz > 0 else None
    n_total = H_tf_base.shape[0]

    # Precompute the same n_draws random N-of-n_total channel-index subsets (without replacement) for every DOE run (draw index k: same channel in every run).
    rng = np.random.default_rng(seed)
    subsets_by_n = _mimo_precompute_channel_subsets(n_total, channel_counts, n_draws, rng)

    # Score each DOE run's H_tf against the baseline, for each channel count and random draw, and record the mean/std of each metric across draws.
    all_rows = []
    n_scored = 0
    for _, doe_row in doe_results.iterrows():

        # Extract the run label and corresponding manifest entry.
        run_label = doe_row['run']
        if run_label not in manifest_by_run.index:
            continue
        m = manifest_by_run.loc[run_label]

        # Load the DOE run's results and compare against the baseline.
        try:
            results_folder = Path(m['results_folder'])
            if not results_folder.is_absolute():
                results_folder = manifest_base_dir / results_folder
            H_tf, run_f_fft, _, _, _ = AnalysisMixin._load_run_data(
                results_folder, 
                verbose=False)
            if H_tf.shape != H_tf_base.shape:
                print(f"analyze_mimo_channel_draw_surface(): run {run_label} "
                      f"- shape {H_tf.shape} != baseline {H_tf_base.shape}, "
                      f"skipping.")
                continue

            # Compute the ratio of the DOE run's channel response to the baseline.
            H_ratio_flat = H_tf / H_tf_base

            for row in _mimo_channel_draws_for_run(
                    H_ratio_flat, run_f_fft, subsets_by_n, delay_res_ns,
                    amp_noise_floor_db, phase_noise_floor_deg):
                row['run'] = run_label
                all_rows.append(row)
            n_scored += 1

        except Exception as e:

            print(f"analyze_mimo_channel_draw_surface(): run {run_label} - "
                  f"could not score: {e!r}")

    print(f"analyze_mimo_channel_draw_surface(): {n_scored}/{len(doe_results)} "
          f"run(s) scored across {len(subsets_by_n)} channel count(s), "
          f"{n_draws} shared draw(s) each.")

    # Export results to CSV
    draws_df = pd.DataFrame(all_rows)
    if draws_df.empty:
        print("analyze_mimo_channel_draw_surface(): no run scored - "
              "skipping the summary figure.")
        return draws_df

    # Average across DOE runs per cell, which is the opposite reduction of analyze_mimo_channel_sensitivity().
    agg = {name: ['mean', 'std'] for name in MIMO_RESPONSE_CHOICES}
    grouped = draws_df.groupby(['n_channels', 'draw_index']).agg(agg)
    grouped.columns = [f'{name}_{stat}' for name, stat in grouped.columns]
    grouped['n_runs_scored'] = draws_df.groupby(['n_channels', 'draw_index']).size()
    results_df = grouped.reset_index()
    ordered_cols = (['n_channels', 'draw_index', 'n_runs_scored'] +
                     [f'{name}_{stat}' for name in MIMO_RESPONSE_CHOICES
                      for stat in ('mean', 'std')])
    results_df = results_df[[c for c in ordered_cols if c in results_df.columns]]

    # Save results to CSV file
    out_csv_path = out_dir / 'mimo_channel_draw_surface.csv'
    results_df.to_csv(out_csv_path, index=False)
    print(f"analyze_mimo_channel_draw_surface(): wrote '{out_csv_path}' "
          f"({len(results_df)} rows).")

    # Plot MIMO channel draw surface
    _plot_mimo_channel_draw_surface(results_df, response_metric, n_scored, out_dir)

    return results_df


def _src_stride_build_geometry(parms_file=None, 
                               consts_file=None) -> "EMSimulation":
    """This function is intended for checking the source-array stride and related geometric parameters without performing a full simulation.

    Args:
        parms_file (str or None): 
            Parameter file of the study to check (grid, frequency band, TX positions, PML; None: default values).
        consts_file (str or None): 
            Constants file (required with parms_file; None: default values).

    Returns:
        em (EMSimulation):
            Partially initialised instance

    Raises:
        ValueError: If parms_file is given without consts_file.
    """
    # Check that if a parameter file is provided, a constants file must also be provided.
    if parms_file is not None and consts_file is None:
        raise ValueError("consts_file is required when parms_file is given.")

    # Load the parameter and constants files if provided, otherwise use default values.
    if parms_file is not None:
        with open(parms_file, encoding='utf-8') as fh:
            parms = json.load(fh)
        with open(consts_file, encoding='utf-8') as fh:
            consts = json.load(fh)
        parms['SIMULATE_BEFORE_ANALYSIS'] = False
        parms['USE_GPU'] = False
        parms['PATH_INPUT_OUTPUT_FILES'] = tempfile.mkdtemp(prefix='src_stride_check_')
        em = __import__('EMSimulation').EMSimulation(parms=parms, consts=consts)
    else:
        em = __import__('EMSimulation').EMSimulation(parms=_SRC_STRIDE_CHECK_PARMS, consts=_SRC_STRIDE_CHECK_CONSTS)

    # Determine the number of elements and the antenna location to compute dx and incidence angles.
    em._determine_number_of_elements()        # -> em.dx
    em._determine_antenna_location_huygens()  # -> em.incident_wave_alpha/beta

    # Mirror the domain_inner / n_x_domain_inner block from
    # _generate_simulation_input_file().
    em.pml_thickness = [em.n_pml * em.dx, em.n_pml * em.dy, em.n_pml * em.dz]
    em.domain_inner = [
        0.0 + em.pml_thickness[0] + em.n_pml_margin * em.dx,
        em.l_x_grid - em.pml_thickness[0] - em.n_pml_margin * em.dx,
        0.0 + em.pml_thickness[1] + em.n_pml_margin * em.dy,
        em.l_y_grid - em.pml_thickness[1] - em.n_pml_margin * em.dy,
        0.0 + em.pml_thickness[2] + em.n_pml_margin * em.dz,
        em.l_z_grid - em.pml_thickness[2] - em.n_pml_margin * em.dz,
    ]
    em.n_x_domain_inner = (em.domain_inner[1] - em.domain_inner[0]) / em.dx
    em.n_y_domain_inner = (em.domain_inner[3] - em.domain_inner[2]) / em.dy

    # Compute array stride for the source grid.
    em.src_array_stride = em._compute_source_array_stride()

    return em


def _src_stride_source_grid(em: "EMSimulation", i_tx: int):
    """This function generates the full-resolution source grid (x, y) and computes the per-point steering phase for a given transmit antenna index "i_tx".

    Args:
        em (EMSimulation class): 
            An instance of the EMSimulation class.
        i_tx (int): 
            Index of the transmit antenna.

    Returns:
        x, y (1D numpy arrays): 
            1D coordinate arrays for the source grid.
        X, Y (2D numpy arrays): 
            2D coordinate arrays for the source grid.
        phase (2D numpy array): 
            Per-point steering phase for the source grid.
        kx, ky (float): 
            Direction cosines of the incident wave.
        k0 (float): 
            Wavenumber corresponding to the shortest-wavelength driven frequency.
        lambda_min (float): 
            Shortest wavelength in the simulation.

    Raises:
        IndexError: If "i_tx" is out of bounds for the incident wave angles arrays.
    """
    # Extract the number of grid points in the inner domain and create the corresponding coordinate arrays.
    nx = int(em.n_x_domain_inner)
    ny = int(em.n_y_domain_inner)
    x = em.domain_inner[0] + np.arange(nx) * em.dx
    y = em.domain_inner[2] + np.arange(ny) * em.dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Extract the incident wave angles for the given transmit antenna index.
    alpha = em.incident_wave_alpha[i_tx]
    beta = em.incident_wave_beta[i_tx]
    kx = np.sin(alpha) * np.cos(beta)
    ky = np.sin(alpha) * np.sin(beta)

    # Compute the wavenumber and the shortest wavelength based on the maximum frequency.
    f_max = em.frequency_range[-1]
    k0 = 2 * np.pi * f_max / sp.constants.c
    lambda_min = sp.constants.c / f_max

    # Compute the per-point steering phase for the source grid.
    phase = np.exp(-1j * k0 * (kx * X + ky * Y))

    return x, y, X, Y, phase, kx, ky, k0, lambda_min


def _src_stride_masked_source(phase, stride):
    """This function creates a masked version of the source grid where only the points on the strided lattice retain their original phase values.
    
    Args:
        phase (2D numpy array): 
            Per-point steering phase for the source grid.
        stride (int): 
            Stride of the source lattice.

    Returns:
        U0 (2D numpy array): 
            Masked source grid with only the strided lattice points retaining their original phase values.
    
    Raises:
        None
    """
    U0 = np.zeros_like(phase)
    U0[::stride, ::stride] = phase[::stride, ::stride]
    return U0


def _src_stride_angular_spectrum_check(U0, dx, dy, k0, lambda_min, label):
    """This function analyzes the angular spectrum of a masked source grid to locate the main lobe and the strongest secondary (grating) peak inside the visible unit disk (in dB).
    
    Args:
        U0 (2D numpy array): 
            Masked source grid with only the strided lattice points retaining their original phase values.
        dx (float): 
            Grid spacing in the x-direction.
        dy (float): 
            Grid spacing in the y-direction.
        k0 (float): 
            Wavenumber of the incident wave.
        lambda_min (float): 
            Shortest wavelength in the simulation.
        label (str): 
            Label for the source grid (used in print statements).
    Returns:
        rel_db (float):
            Level of the strongest secondary peak relative to the main lobe in dB. 

    Raises:
        None
    """
    # Extract the shape of the input masked source grid and determine the FFT sizes.
    nx, ny = U0.shape
    Nfft_x = scipy.fft.next_fast_len(4 * nx)
    Nfft_y = scipy.fft.next_fast_len(4 * ny)

    # Compute the 2D FFT of the masked source grid and shift the zero-frequency component to the center.
    A = scipy.fft.fftshift(scipy.fft.fft2(U0, s=(Nfft_x, Nfft_y)))
    fx = scipy.fft.fftshift(scipy.fft.fftfreq(Nfft_x, d=dx))
    fy = scipy.fft.fftshift(scipy.fft.fftfreq(Nfft_y, d=dy))

    # Compute the direction cosines for the visible unit disk.
    lam = 2 * np.pi / k0
    U = fx[:, None] * lam * np.ones((1, len(fy)))
    V = np.ones((len(fx), 1)) * (fy[None, :] * lam)

    # Determine which points fall inside the visible unit disk.
    visible = (U**2 + V**2) <= 1.0
    mag = np.abs(A)
    mag_visible = np.where(visible, mag, 0.0)

    # Locate the main lobe within the visible unit disk.
    main_idx = np.unravel_index(np.argmax(mag_visible), mag.shape)
    main_level = mag[main_idx]
    u_main, v_main = U[main_idx], V[main_idx]

    # Compute the exclusion region around the main lobe to avoid detecting it as a secondary peak (~3x the main-lobe's Rayleigh width).
    aperture_u = nx * dx
    aperture_v = ny * dy
    du_excl = 3 * lam / aperture_u
    dv_excl = 3 * lam / aperture_v
    excl = ((U - u_main)**2 / du_excl**2 + (V - v_main)**2 / dv_excl**2) <= 1.0
    secondary = np.where(visible & ~excl, mag, 0.0)
    sec_idx = np.unravel_index(np.argmax(secondary), mag.shape)
    sec_level = mag[sec_idx]
    u_sec, v_sec = U[sec_idx], V[sec_idx]

    # Compute the relative level of the strongest secondary peak with respect to the main lobe.
    rel_db = 20 * np.log10(sec_level / main_level) if main_level > 0 else float("nan")

    print(f"[{label}] main lobe at (u,v)=({u_main:+.4f},{v_main:+.4f}); "
          f"strongest secondary peak at (u,v)=({u_sec:+.4f},{v_sec:+.4f}), "
          f"{rel_db:+.1f} dB relative to main lobe")
    
    return rel_db


def _src_stride_propagate_asm(U0, dx, dy, k0, z0, pad):
    """This function propagates the input field U0 over a distance z0 using the angular spectrum method (ASM).
    
    Args:
        U0 (np.ndarray): 
            Input field at the source plane.
        dx (float): 
            Grid spacing in the x-direction.
        dy (float): 
            Grid spacing in the y-direction.
        k0 (float): 
            Wavenumber.
        z0 (float): 
            Propagation distance.
        pad (int): 
            Padding size for FFT.

    Returns:
        np.ndarray: Propagated field at distance z0.

    Raises:
        None
    """
    # Pad the input field to avoid circular convolution effects in the FFT.
    nx, ny = U0.shape
    Nx, Ny = nx + 2 * pad, ny + 2 * pad
    U0p = np.zeros((Nx, Ny), dtype=complex)
    U0p[pad:pad + nx, pad:pad + ny] = U0

    # Compute the spatial frequency grids for the padded field.
    fx = scipy.fft.fftfreq(Nx, d=dx)
    fy = scipy.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    kz2 = k0**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2
    kz = np.sqrt(kz2.astype(complex))  # imaginary part => evanescent decay
    H = np.exp(1j * kz * z0)

    # Apply the angular spectrum transfer function to propagate the field.
    Uz = scipy.fft.ifft2(scipy.fft.fft2(U0p) * H)

    return Uz[pad:pad + nx, pad:pad + ny]


def verify_source_array_subsampling(output_dir=None, 
                                    parms_file=None, 
                                    consts_file=None) -> None:
    """This function checks whether the source array stride used in the EM simulation is appropriate for avoiding grating lobes (Huygens dipole source array) by comparing the stride against a worst-case scenario based on the wavelength and the grid spacing. A summary of the check is printed and saved to the specified output directory.

    Args:
        output_dir (str or Path, optional): 
            Directory to save the output files. Defaults to this file's directory.
        parms_file (str or Path, optional): 
            Path to the parameters file for the EM simulation. Defaults to None.
        consts_file (str or Path, optional): 
            Path to the constants file for the EM simulation. Defaults to None.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified parms_file or consts_file does not exist.
    """
    # Ensure the output directory exists.
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the EM simulation geometry using the provided parameter and constants files.
    em = _src_stride_build_geometry(parms_file, consts_file)
    print(f"Setup: {'parameter file ' + str(parms_file) if parms_file else 'built-in mini setup (NOT the production geometry)'}")
    print(f"dx = {em.dx * 1e3:.4f} mm, frequency band {em.frequency_range[0] / 1e9:.1f}-{em.frequency_range[-1] / 1e9:.1f} GHz, "
          f"{em.n_pos} TX position(s), max |sin(alpha)| = {np.max(np.abs(np.sin(em.incident_wave_alpha))):.4f}")
    rows = []
    stride_ok = em.src_array_stride
    lam_check = sp.constants.c / em.frequency_range[-1]

    # Determine the stride that would result in approximately one source per wavelength.
    stride_bad = max(stride_ok + 1, int(round(lam_check / em.dx)))

    # Select the worst-case TX position based on the maximum incidence angle.
    i_tx = int(np.argmax(np.abs(np.sin(em.incident_wave_alpha))))
    print(f"Using TX position {i_tx} (worst-case incidence angle) of {em.n_pos}")
    print(f"Array size: {int(em.n_x_domain_inner)} x {int(em.n_y_domain_inner)} "
          f"({int(em.n_x_domain_inner) * int(em.n_y_domain_inner)} pts at stride=1)")

    # Generate the source grid and associated parameters for the selected TX position.
    x, y, X, Y, phase, kx, ky, k0, lambda_min = _src_stride_source_grid(em, i_tx)

    # Conduct angular-spectrum grating-lobe check (Part A))
    print("\n--- Part A: angular-spectrum (Fourier) grating-lobe check ---")
    U_full = _src_stride_masked_source(phase, 1)
    U_ok = _src_stride_masked_source(phase, stride_ok)
    U_bad = _src_stride_masked_source(phase, stride_bad)
    db_full = _src_stride_angular_spectrum_check(U_full, em.dx, em.dy, 
                                                 k0, lambda_min, "stride=1 (full density)")
    db_ok = _src_stride_angular_spectrum_check(U_ok, em.dx, em.dy, 
                                               k0, lambda_min, f"stride={stride_ok} (chosen)")
    db_bad = _src_stride_angular_spectrum_check(U_bad, em.dx, em.dy, 
                                                k0, lambda_min, f"stride={stride_bad} (negative control, ~1 src/wavelength)")

    # Apply near-field propagation - full vs. chosen stride (Part B))
    print("\n--- Part B: near-field propagation (angular-spectrum method) ---")

    # Define the set of z-values (propagation distances) to evaluate, in millimeters.
    z_values_mm = [1.0, 5.0, 15.0]

    # If a parameters file is provided, include the top Huygens cap offset in the z-values.
    if parms_file is not None:
        z_cap_mm = getattr(em, 'huygens_top_cap_offset_cells', 15) * em.dx * 1e3
        z_values_mm = sorted(set([1.0, round(z_cap_mm, 3), 5.0, 15.0]))
    pad = 400  # guard cells against FFT periodic wraparound
    n_ok_expected = U_full.astype(bool).sum() / U_ok.astype(bool).sum()

    # Set up the figure and axes for plotting the results. Each row corresponds to a different z-value, and each column corresponds to full density, chosen stride, and their difference.
    fig, axes = plt.subplots(len(z_values_mm), 3, figsize=(12, 3.6 * len(z_values_mm)))
    for row, z_mm in enumerate(z_values_mm):

        # Convert the current z-value from millimeters to meters for propagation calculations.
        z0 = z_mm * 1e-3

        # Perform near-field propagation for the full density and chosen stride source distributions.
        U_full_z = _src_stride_propagate_asm(U_full, em.dx, em.dy, k0, z0, pad)
        U_ok_z = _src_stride_propagate_asm(U_ok, em.dx, em.dy, k0, z0, pad)

        # Compute the amplitude (magnitude) of the propagated fields.
        amp_full = np.abs(U_full_z)
        amp_ok = np.abs(U_ok_z)

        # Normalize each field by its own peak so the diff map shows pattern shape differences, not the overall amplitude scale.
        amplitude_ratio = amp_full.max() / amp_ok.max()
        amp_full_n = amp_full / amp_full.max()
        amp_ok_n = amp_ok / amp_ok.max()
        diff = np.abs(amp_ok_n - amp_full_n)
        rel_rms = np.sqrt(np.mean(diff**2))
        rel_max = diff.max()
        print(f"z={z_mm:.1f} mm: amplitude_ratio(full/stride)={amplitude_ratio:.1f}x (expected ~{n_ok_expected}x from dipole count - compensate by scaling TX_AMPLITUDE if absolute field level matters), peak-normalized pattern RMS diff={rel_rms:.4f}, max diff={rel_max:.4f}")

        rows.append({'part': 'B', 
                     'z_mm': z_mm, 
                     'pattern_rms_diff': rel_rms, 
                     'pattern_max_diff': rel_max,
                     'amplitude_ratio_full_over_stride': amplitude_ratio})

        # Plot the normalized full density, chosen stride, and their difference for the current z-value.
        for col, (data, title) in enumerate([
            (amp_full_n, f"|U| full density (norm.), z={z_mm:.1f} mm"),
            (amp_ok_n, f"|U| stride={stride_ok} (norm.), z={z_mm:.1f} mm"),
            (diff, f"|normalized pattern diff|, z={z_mm:.1f} mm"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin='lower', aspect='auto',
                            extent=[x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3])
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Adjust the layout of the figure to prevent overlapping elements.
    fig.tight_layout()
    out_path = output_dir / 'source_array_subsampling_check.png'
    _savefig(fig, out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved field comparison plot to {out_path}")

    # Print a summary of the source array subsampling analysis.
    print("\n--- Summary ---")
    print(f"lambda_min = {lambda_min * 1e3:.3f} mm, dx = {em.dx * 1e3:.4f} mm")
    print(f"stride=1 (full):        {int(em.n_x_domain_inner) * int(em.n_y_domain_inner)} sources, worst secondary lobe {db_full:+.1f} dB")
    n_ok = len(range(0, int(em.n_x_domain_inner), stride_ok)) * len(range(0, int(em.n_y_domain_inner), stride_ok))
    print(f"stride={stride_ok} (chosen):     {n_ok} sources, worst secondary lobe {db_ok:+.1f} dB")
    n_bad = len(range(0, int(em.n_x_domain_inner), stride_bad)) * len(range(0, int(em.n_y_domain_inner), stride_bad))
    print(f"stride={stride_bad} (negative control): {n_bad} sources, worst secondary lobe {db_bad:+.1f} dB")
    n_full = int(em.n_x_domain_inner) * int(em.n_y_domain_inner)
    
    # Append the summary information for each case to the rows list.
    for lbl, st, n_src, db in (('full', 1, n_full, db_full), ('chosen', stride_ok, n_ok, db_ok), ('negative_control', stride_bad, n_bad, db_bad)):
        rows.append({'part': 'A', 'case': lbl, 'stride': st, 'n_sources': n_src, 'worst_secondary_lobe_db': db,
                     'dx_mm': em.dx * 1e3, 'lambda_min_mm': lambda_min * 1e3,
                     'worst_incidence_sin': float(np.max(np.abs(np.sin(em.incident_wave_alpha)))),
                     'setup': str(parms_file) if parms_file else 'built-in mini setup'})
    
    # Save the summary of the source array subsampling analysis to a CSV file.
    csv_path = output_dir / 'source_array_subsampling_check.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved numbers to {csv_path}")


def _msla_load_scan(study_dir) -> dict:
    """ This function loads the scan configuration, manifest, and relevant data for a MODE 12 study (moving sample line array).

    Args:
        study_dir (str or Path): 
            Study folder of MODE 12.

    Returns:
        dict: config, offsets_mm (M,), f_hz (F,), H_base/H_fo/H_empty (M, K, F) complex (H_base/H_empty None if not simulated), k_tx (M, 3),
            k_rx (M, K, 3), c_m (M, 3) sample center, tau_off (M,), delta_m, rx_huygens (M, K, 3), nf2ff_mode,
            rx_y_mm (K,) RX y positions relative to the array center.

    Raises:
        FileNotFoundError: Config/manifest missing.
        RuntimeError: No position has both cases, or a run lacks the geometry block or used the legacy plane-wave angles.
    """

    # Ensure the study directory and result paths exist.
    study_dir = Path(study_dir)
    res_dir = study_dir / 'results_analysis'
    cfg_path = res_dir / 'scan_config.json'
    man_path = res_dir / 'manifest_moving_sample_line_array.csv'
    for p in (cfg_path, man_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"_msla_load_scan(): {p} not found - run MODE == 12 first.")
    with open(cfg_path) as fh:
        config = json.load(fh)

    # Load the scan configuration from the JSON file.
    manifest = pd.read_csv(man_path, dtype={'position_index': str, 'case': str,
                                             'results_folder': str})
    manifest = manifest[manifest['status'] == 'OK']
    has_empty = bool(config.get('has_empty'))
    has_base = bool(config.get('has_baseline', True))
    need = tuple(c for c, on in (('empty', has_empty), ('baseline', has_base), ('fo', True)) if on)

    # Filter the manifest to only include rows corresponding to the current scan configuration offsets.
    offs = np.asarray(config.get('offsets_mm', []), float)
    if offs.size:
        manifest = manifest[[int(i) < offs.size and bool(np.isclose(float(o), offs[int(i)], atol=1e-6)) for i, o in zip(manifest['position_index'], manifest['offset_mm'])]]
    positions = []

    # Group the manifest by position index and offset, and collect the cases for each position.
    for (pidx, _), grp in manifest.groupby(['position_index', 'offset_mm'], sort=False):
        cases = dict(zip(grp['case'], grp['results_folder']))
        miss = [c for c in need if c not in cases]
        if not miss:
            positions.append((float(grp['offset_mm'].iloc[0]), cases))
        else:
            print(f"_msla_load_scan(): position {pidx} lacks {miss} - skipped.")
    
    # Ensure that there is at least one position with all required cases.
    if not positions:
        raise RuntimeError(f"_msla_load_scan(): no position has all OK runs ({', '.join(need)}).")
    positions.sort(key=lambda t: t[0])

    # Iterate over each position and load the corresponding data for the required cases.
    out = dict(config=config, offsets_mm=np.array([p[0] for p in positions]))
    H_b, H_f, H_e, k_tx, k_rx, c_m, tau, rxh = [], [], [], [], [], [], [], []
    for off, cases in positions:

        # Load the data for the current case from the corresponding folder.
        loaded = {}
        for case in need:
            folder = _resolve_cookie_run_folder(study_dir, cases[case])
            with open(folder / 'data.pkl', 'rb') as fh:
                loaded[case] = pickle.load(fh)

        # Get the geometry information for each loaded case. Raise an error if the geometry is missing or incorrect.
        for case, data in loaded.items():
            geom = data.get('geometry')
            if geom is None or geom.get('plane_wave_angles') != 'CORRECT':
                raise RuntimeError(
                    f"_msla_load_scan(): the {case} run at offset {off:+.1f} mm has no "
                    "MODE == 12 geometry block (or used the legacy plane-wave angles) - "
                    "it was not produced by run_moving_sample_line_array_simulation().")

        # Append the relevant data to the output lists for the current position.
        if has_base:
            H_b.append(loaded['baseline']['H_tf'][0])
        H_f.append(loaded['fo']['H_tf'][0])
        if has_empty:
            H_e.append(loaded['empty']['H_tf'][0])
        g = loaded['fo']['geometry']
        k_tx.append(np.asarray(g['k_tx'])[:, 0])
        k_rx.append(np.asarray(g['k_rx']).T)
        c_m.append(np.asarray(g['target_center']))
        tau.append(float(np.asarray(g['tau_offset_m'])[0]))
        if g.get('rx_mode') == 'DIRECT':
            rxh.append(np.asarray(loaded['fo']['rx_pos']).T)
        else:
            rxh.append(np.asarray(g['rx_huygens']).T)
        out['f_hz'] = loaded['fo']['f_fft']
        out['delta_m'] = float(g['delta_z'])
        out['nf2ff_mode'] = 'EXACT' if g.get('rx_mode') == 'DIRECT' else g['nf2ff_mode']
        rx_pos = np.asarray(loaded['fo']['rx_pos'])
        out['rx_y_mm'] = (rx_pos[1] - g['l_grid'][1] / 2) * 1e3

    # Update the output dictionary with the collected data arrays.
    out.update(H_base=np.array(H_b) if has_base else None, 
               H_fo=np.array(H_f),
               H_empty=np.array(H_e) if has_empty else None, 
               k_tx=np.array(k_tx),
               k_rx=np.array(k_rx), 
               c_m=np.array(c_m), 
               tau_off=np.array(tau),
               rx_huygens=np.array(rxh))
    
    return out


def _msla_backproject(D, f_hz, k_tx, k_rx, c_m, tau_off, delta_m, rx_huygens,
                      nf2ff_mode, x_c, y_c, z_above_belt, window='hann',
                      chunk=48, progress=True) -> np.ndarray:
    """This function performs matched-filter back-projection imaging of the differential field D on a pixel grid defined by x_c, y_c, and z_above_belt based on the provided geometry and NF2FF mode.

    Args:
        D (ndarray): 
            (M, K, F) complex differential transfer function.
        f_hz (ndarray): 
            (F,) frequencies.
        k_tx, k_rx, c_m, tau_off, delta_m, rx_huygens: 
            Geometry from _msla_load_scan().
        nf2ff_mode (str): 
            'FARFIELD_CENTERED' or 'EXACT'.
        x_c, y_c (ndarray): 
            1D pixel coordinates in the cookie frame [m].
        z_above_belt (float): 
            Image plane height above the belt [m].
        window (str): 
            'hann' (default) or 'none' taper over m, k and f.
        chunk (int): 
            Pixels per batch (memory).
        progress (bool): 
            Flag to print progress.

    Returns:
        ndarray: (len(y_c), len(x_c)) complex image.

    Raises:
        NotImplementedError: nf2ff_mode other than the two above.
    """
    # Check if the NF2FF mode is supported.
    if nf2ff_mode not in ('FARFIELD_CENTERED', 'EXACT'):
        raise NotImplementedError(
            f"_msla_backproject(): imaging model for NF2FF_MODE={nf2ff_mode!r} not implemented.")

    # Extract the dimensions of the differential field and compute the wavenumber array.
    M, K, F = D.shape
    kf = 2 * np.pi * np.asarray(f_hz) / sp.constants.c

    # Define helper function to generate windowing arrays.
    def _w(n):
        return np.hanning(n + 2)[1:-1] if (window == 'hann' and n > 2) else np.ones(n)

    # Apply windowing to the differential field.
    Dw = D * _w(M)[:, None, None] * _w(K)[None, :, None] * _w(F)[None, None, :]

    # Generate the pixel grid and initialize the image array.
    xx, yy = np.meshgrid(x_c, y_c)                       # (Ny, Nx)
    P_rel = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_above_belt)], axis=1)  # (P, 3)
    img = np.zeros(len(P_rel), dtype=complex)

    # Loop over pixel chunks and perform back-projection.
    t0 = time.time()
    for s in range(0, len(P_rel), chunk):

        # Extract the current chunk of pixels.
        pr = P_rel[s:s + chunk]                           # (p, 3)
        
        # Compute the absolute positions of the scatterers relative to the current pixel chunk (p, M, 3).
        r_abs = c_m[None, :, :] + pr[:, None, :] 

        # Compute the incident field phase for the current pixel chunk.
        inc = np.einsum('pmi,mi->pm', r_abs, k_tx) - tau_off[None, :]

        # Compute the observed field phase for the current pixel chunk.
        if nf2ff_mode == 'FARFIELD_CENTERED':
            obs = delta_m + np.einsum('pi,mki->pmk', pr, k_rx)
        else:
            obs = np.linalg.norm(rx_huygens[None, :, :, :] - r_abs[:, :, None, :], axis=-1)

        # Compute the total phase for the current pixel chunk.
        psi = inc[:, :, None] + obs                       # (p, M, K)
        ph = np.exp(1j * psi[..., None] * kf[None, None, None, :])
        img[s:s + chunk] = np.einsum('pmkf,mkf->p', ph, Dw)
        
        # Print progress every 20 chunks if enabled.
        if progress and (s // chunk) % 20 == 0:
            print(f"_msla_backproject(): pixel {min(s + chunk, len(P_rel))}/{len(P_rel)} "
                  f"[{time.time() - t0:.0f} s]")
            
    return img.reshape(xx.shape)


def analyze_moving_sample_line_array(study_dir: str, output_dir: str = None,
                                     pixel_mm: float = 1.0, margin_mm: float = 10.0,
                                     z_above_belt_mm: float = None, freq_stride: int = 1,
                                     window: str = 'hann', cmap: str = 'jet',
                                     show: bool = False) -> dict:
    """MODE == 13: image the MODE == 12 scan from its empty-belt, baseline and foreign-object runs (no simulation).
    This function analyzes the moving sample line array data (MODE 12 study). It generates various images and summaries based on the empty-belt, baseline, and foreign-object runs.

    Args:
        study_dir (str): 
            MODE 12 study folder.
        output_dir (str): 
            Output folder (default study_dir/results_analysis).
        pixel_mm (float): 
            Image pixel size [mm] (default 1).
        margin_mm (float): 
            Image margin around the cookie outline in mm.
        z_above_belt_mm (float): 
            Image plane height above the belt [mm]; None = FO center height from scan_config.json.
        freq_stride (int): 
            Use every n-th frequency bin (speed).
        window (str): 
            'hann' or 'none' taper of the back-projection.
        cmap (str): 
            Colormap of all images (default 'jet').
        show (bool): 
            plt.show() the figures (default: only save).

    Returns:
        dict: Image arrays and peak summary (also saved as moving_sample_images.npz / moving_sample_peak_summary.csv).

    Raises:
        RuntimeError: If the required baseline or empty runs are not available.
    """
    # Prepare output directory and load scan data
    study_dir = Path(study_dir)
    out_dir = Path(output_dir) if output_dir else study_dir / 'results_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the scan data and check for required baseline and empty runs
    scan = _msla_load_scan(study_dir)

    # Check for required baseline and empty runs. Raise error if not available.
    if scan['H_base'] is None or scan['H_empty'] is None:
        raise RuntimeError("analyze_moving_sample_line_array(): MODE == 13 needs the 'baseline' and 'empty' runs, which this study did not "
                           "simulate - rerun MODE == 12 with MSLA_SIMULATE_BASELINE/MSLA_SIMULATE_EMPTY = True (MODE == 14 works without them).")

    # Extract configuration and offsets from the scan data
    cfg = scan['config']
    off_mm = scan['offsets_mm']
    f_hz = scan['f_hz'][::freq_stride]
    H_b = scan['H_base'][..., ::freq_stride]
    H_f = scan['H_fo'][..., ::freq_stride]
    M, K, F = H_f.shape
    print(f"analyze_moving_sample_line_array(): {M} position(s) "
          f"({off_mm[0]:+.1f} .. {off_mm[-1]:+.1f} mm), {K} RX, {F} frequency bins.")

    # Load the empty-belt data and check for availability
    H_e = scan['H_empty']
    if H_e is None:
        raise RuntimeError("analyze_moving_sample_line_array(): this study has no empty-belt runs "
                           "- rerun MODE == 12 (it simulates them now).")

    # Apply frequency stride to the empty-belt data
    H_e = H_e[..., ::freq_stride]

    # Define helper function for band-averaged complex ratio relative to the empty belt
    def _tr(H):
        return np.mean(H / H_e, axis=-1)

    # Compute the band-averaged complex ratios for the FO and baseline cases relative to the empty belt
    T_fo, T_base = _tr(H_f), _tr(H_b)
    ratio = np.mean(H_f / H_b, axis=-1)

    # Compute amplitude and phase of the FO vs. baseline ratio, and differential level relative to the baseline
    amp = np.abs(ratio)
    phase_deg = np.degrees(np.angle(ratio))
    D = H_f - H_b
    diff_db = 20 * np.log10(np.maximum(np.mean(np.abs(D), axis=-1), 1e-300)
                            / np.max(np.mean(np.abs(H_b), axis=-1)))
    print(f"analyze_moving_sample_line_array(): cookie+FO vs. empty belt: |T| {np.abs(T_fo).min():.4f} .. {np.abs(T_fo).max():.4f}; FO vs. baseline: |ratio| {amp.min():.4f} .. {amp.max():.4f}, phase {phase_deg.min():+.2f} .. {phase_deg.max():+.2f} deg, differential level {diff_db.min():.1f} .. {diff_db.max():.1f} dB re max|H_base|.")

    # Store the computed results in a dictionary for later use
    result = dict(offsets_mm=off_mm, 
                  rx_y_mm=scan['rx_y_mm'], 
                  ratio_amp=amp,
                  ratio_phase_deg=phase_deg, 
                  diff_db=diff_db,
                  T_fo_amp=np.abs(T_fo), 
                  T_fo_phase_deg=np.degrees(np.angle(T_fo)),
                  T_base_amp=np.abs(T_base), 
                  T_base_phase_deg=np.degrees(np.angle(T_base)))
    
    # If there are fewer than 3 positions, save the results and exit early
    if M < 3:
        print("analyze_moving_sample_line_array(): fewer than 3 positions (pilot run?) - no "
              "image can be formed; run the full scan (MSLA_PILOT = False) and repeat.")
        np.savez(out_dir / 'moving_sample_images.npz', **result)
        return result

    # Prepare the beam positions and sort them for plotting and analysis
    x_beam = -off_mm
    order = np.argsort(x_beam)

    # Define function to draw the cookie outline and FO footprint on the plots
    def _cookie_outline(ax, fo=True):
        if cfg.get('sut_outline_mm'):
            _msla_draw_freeform_outline(ax, cfg, color='k', lw=1.2)
        elif cfg['sut_shape'] == 'CYLINDER':
            ax.add_patch(plt.Circle((0, 0), 
                                    cfg['sut_size']['DIAMETER'] * 500, fill=False,
                                    ec='k', lw=1.2))
        else:
            sx, sy = cfg['sut_size'][0] * 1e3, cfg['sut_size'][1] * 1e3
            ax.add_patch(Rectangle((-sx / 2, -sy / 2), sx, sy, fill=False, ec='k', lw=1.2))
        if fo:
            fx, fy, _ = cfg['fo_center_cookie_frame_mm']
            s_ = cfg['fo_size_mm']
            ax.add_patch(Rectangle((fx - s_ / 2, fy - s_ / 2), s_, s_, fill=False, ec='k', lw=1.5, ls='--'))

    # Compute the pixel edges for the beam positions and RX positions
    xg = x_beam[order]
    yg = scan['rx_y_mm']

    # Define function to compute pixel edges for plotting
    def _edges(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        mid = (v[1:] + v[:-1]) / 2
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    xe, ye = _edges(xg), _edges(yg)

    # Define function to create the cookie image figure (normalized amplitude and phase; cookie + FO relative to the empty belt)
    def _t_figure(T, title, fname):
        
        A = np.abs(T)[order].T
        A_norm = A / A.max()
        P = np.degrees(np.angle(T))[order].T

        fig, axs_ = plt.subplots(1, 2, figsize=(13, 5.2), 
                                 sharex=True, sharey=True)
        im = axs_[0].pcolormesh(xe, ye, A_norm, 
                                cmap=cmap, 
                                vmin=0, vmax=1, 
                                shading='flat')
        fig.colorbar(im, ax=axs_[0], 
                     label='normalized amplitude |H / H_empty| (max = 1)')
        axs_[0].set_title('normalized amplitude')
        im = axs_[1].pcolormesh(xe, ye, P, 
                                cmap=cmap, 
                                vmin=-180, vmax=180, shading='flat')
        fig.colorbar(im, ax=axs_[1], 
                     label='phase(H / H_empty) / deg (band mean)')
        axs_[1].set_title('phase')
        
        for ax in axs_:
            _cookie_outline(ax, fo=(fname == 'moving_sample_cookie_image.png'))
            ax.set_xlabel('x in the cookie frame / mm  (= TX/RX axis position = -conveyor offset)')
            ax.set_aspect('equal')
        axs_[0].set_ylabel('y in the cookie frame / mm  (= RX position)')
        
        fig.suptitle(title)
        fig.tight_layout()
        _savefig(fig, out_dir / fname, dpi=150)

        return A_norm, P

    # Create the cookie image figure
    t_title = (f"{cfg['cookie_name']} with {cfg['fo_size_mm']:g} mm {cfg['fo_material']} object (dashed: object, black: cookie outline) - fixed TX/RX, moving sample, {M} positions x {K} RX")
    cookie_amp, cookie_phase = _t_figure(T_fo, t_title, 'moving_sample_cookie_image.png')
    _t_figure(T_base, f"{cfg['cookie_name']} without the object - same scan",
              'moving_sample_cookie_image_baseline.png')

    # Update the result dictionary with the cookie image data
    result.update(cookie_amp_norm=cookie_amp, cookie_phase_deg=cookie_phase,
                  x_scan_mm=xg, y_rx_mm=yg)

    # Create the object-only channel images figure (FO vs. same cookie without FO) 
    fig1, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, data, title, cm, label in (
            (axs[0], amp[order].T, '|H_FO / H_baseline| (1 = no object)', cmap, 'normalized amplitude'),
            (axs[1], phase_deg[order].T, 'phase(H_FO / H_baseline)', cmap, 'degrees'),
            (axs[2], diff_db[order].T, '|H_FO - H_baseline| re max|H_baseline|', cmap, 'dB')):
        im = ax.pcolormesh(xe, ye, data, shading='flat', cmap=cm)
        fig1.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        _cookie_outline(ax)
        ax.set_xlabel('x in the cookie frame / mm')

    # Set the y-axis label for the first subplot and adjust the layout of the figure. Save the figure afterward.
    axs[0].set_ylabel('y in the cookie frame / mm')
    fig1.suptitle(f"{cfg['cookie_name']} - object-only channel images, {cfg['fo_size_mm']:g} mm {cfg['fo_material']} object")
    fig1.tight_layout()
    _savefig(fig1, out_dir / 'moving_sample_channel_images.png', dpi=150)

    # Create the focused image (backprojected) figure
    half_x = (cfg['sut_size']['DIAMETER'] / 2 if cfg['sut_shape'] == 'CYLINDER'
              else cfg['sut_size'][0] / 2)
    half_y = (cfg['sut_size']['DIAMETER'] / 2 if cfg['sut_shape'] == 'CYLINDER'
              else cfg['sut_size'][1] / 2)
    px = pixel_mm / 1000.0
    x_c = np.arange(-(half_x + margin_mm / 1000), 
                    half_x + margin_mm / 1000 + px / 2, px)
    y_c = np.arange(-(half_y + margin_mm / 1000), 
                    half_y + margin_mm / 1000 + px / 2, px)
    z_img = (cfg['fo_center_cookie_frame_mm'][2] if z_above_belt_mm is None else z_above_belt_mm) / 1000.0

    # Backproject the focused image using the MSLA algorithm
    img = _msla_backproject(D, f_hz, scan['k_tx'], scan['k_rx'], scan['c_m'], 
                            scan['tau_off'], scan['delta_m'], 
                            scan['rx_huygens'], 
                            scan['nf2ff_mode'],
                            x_c, y_c, z_img, window=window)

    # Compute the magnitude, normalized amplitude, and phase relative to the peak pixel
    mag = np.abs(img)
    amp_norm = mag / mag.max()
    iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
    peak_x, peak_y = 1e3 * x_c[ix], 1e3 * y_c[iy]
    true_x, true_y, _ = cfg['fo_center_cookie_frame_mm']
    err = float(np.hypot(peak_x - true_x, peak_y - true_y))
    phase_rel = np.degrees(np.angle(img * np.conj(img[iy, ix])))
    print(f"analyze_moving_sample_line_array(): image peak at (x, y) = ({peak_x:+.1f}, {peak_y:+.1f}) mm; "
          f"FO center ({true_x:+.1f}, {true_y:+.1f}) mm; distance {err:.1f} mm "
          f"(plane z = {z_img * 1e3:.1f} mm above the belt).")

    # Plot the amplitude, amplitude in dB, and phase of the focused image
    fig2, axs = plt.subplots(1, 3, figsize=(17, 5.4), sharex=True, sharey=True)
    
    ext = [1e3 * (x_c[0] - px / 2), 1e3 * (x_c[-1] + px / 2), 1e3 * (y_c[0] - px / 2), 1e3 * (y_c[-1] + px / 2)]
    im0 = axs[0].imshow(amp_norm, origin='lower', extent=ext, cmap=cmap, vmin=0, vmax=1)

    fig2.colorbar(im0, ax=axs[0], label='normalized amplitude (max = 1)')
    amp_db = 20 * np.log10(np.maximum(amp_norm, 1e-6))
    im1 = axs[1].imshow(amp_db, origin='lower', extent=ext, cmap=cmap, vmin=-30, vmax=0)

    fig2.colorbar(im1, ax=axs[1], label='dB re peak')
    ph = np.ma.masked_where(amp_db < -15, phase_rel)
    im2 = axs[2].imshow(ph, origin='lower', extent=ext, cmap=cmap, vmin=-180, vmax=180)

    fig2.colorbar(im2, ax=axs[2], label='phase re peak pixel / deg  (masked below -15 dB)')

    # Mark the true FO position on the plots
    for ax, t in zip(axs, ('amplitude', 'amplitude, dB', 'phase')):
        _cookie_outline(ax)
        ax.plot([peak_x], [peak_y], 'k+', ms=12, mew=2)
        ax.set_title(t)
        ax.set_xlabel('x in the cookie frame / mm')

    # Set the y-axis label, title, and x-axis label for the first subplot. Save the figure afterward.
    axs[0].set_ylabel('y in the cookie frame / mm')
    fig2.suptitle(f"{cfg['cookie_name']} - focused image of the foreign object (H_FO - H_baseline). "
                  f"Black: cookie outline; dashed: true {cfg['fo_size_mm']:g} mm object; "
                  f"+: image peak ({err:.1f} mm from the object center)")
    fig2.tight_layout()
    _savefig(fig2, out_dir / 'moving_sample_focused_image.png', dpi=150)

    # Save the results and summary of the moving sample analysis
    result.update(x_c_mm=1e3 * x_c, 
                  y_c_mm=1e3 * y_c, 
                  image=img, 
                  amp_norm=amp_norm,
                  phase_rel_deg=phase_rel, 
                  peak_xy_mm=(peak_x, peak_y),
                  true_fo_xy_mm=(true_x, true_y), 
                  peak_error_mm=err)
    np.savez(out_dir / 'moving_sample_images.npz', **{k: np.asarray(v) for k, v in result.items()})
    pd.DataFrame([dict(cookie=cfg['cookie_name'], 
                       n_positions=M, 
                       n_rx=K, 
                       n_freq=F,
                       peak_x_mm=peak_x, 
                       peak_y_mm=peak_y, 
                       true_x_mm=true_x, 
                       true_y_mm=true_y,
                       peak_error_mm=err, 
                       image_z_mm=z_img * 1e3)]).to_csv(out_dir / 'moving_sample_peak_summary.csv', index=False)
    
    # Show or close the plots based on the flag
    if show:
        plt.show()
    else:
        plt.close('all')
    print(f"analyze_moving_sample_line_array(): done - see {out_dir}")

    return result


def _msla_draw_freeform_outline(ax, cfg: dict, **kw):
    """This function draws the freeform outline and its regions on the given axis (for MODE 12 data).
    
    Args:
        ax (matplotlib.axes.Axes): 
            The axis on which to draw the outline.
        cfg (dict): 
            The scan configuration containing the freeform outline and region outlines.
        **kw: 
            Additional keyword arguments passed to the plot function for customization.

    Returns:
        None
    
    Raises:
        None
    """
    ax.plot(*cfg['sut_outline_mm'], **kw)
    for rx_, ry_ in cfg.get('sut_region_outlines_mm', []):
        ax.plot(rx_, ry_, **{**kw, 'lw': 0.6 * kw.get('lw', 1.0)})


def _msla_camera_model(I_fo, I_base, I_empty, x_mm, y_mm, sheets=(0, 10, 20),
                       atten_db_per_sheet=0.5, aperture_mm=1.5, smear_x_mm=0.0,
                       pitch_mm=None, step_mm=None, interp='cubic', 
                       noise_rel=0.02, gain_spread=0.05, offset_rel=0.01, adc_bits=12, full_scale=1.0, seed=1) -> dict:
    """This function simulates the response of a THz line camera (power detector) to the given intensity maps, including resampling, pixel aperture, smear, paper-stack damping, and noise (for MODE 12 data).


    Args:
        I_fo, I_base, I_empty (ndarray): 
            Intensity maps (Ny, Nx) in units of the reference intensity; I_base/I_empty may be None (not simulated).
        x_mm, y_mm (ndarray): 
            Pixel centers of the maps (ascending), cookie frame.
        sheets (tuple): 
            Numbers of paper sheets, one figure column each.
        atten_db_per_sheet (float): 
            Power damping per sheet in dB.
        aperture_mm (float):
            Pixel aperture (FWHM) along y.
        smear_x_mm (float): 
            Smear along x.
        pitch_mm (float): 
            Camera pixel pitch (y) of the output grid; None = simulated grid.
        step_mm (float): 
            Conveyor step (x) of the output grid; None = simulated grid.
        interp (str): 
            'cubic' or 'linear' resampling (clipped at 0).
        noise_rel (float): 
            Std of the additive temporal noise [full-scale units].
        gain_spread (float): 
            Std of the relative per-pixel gain.
        offset_rel (float): 
            Std of the per-pixel offset [full-scale units].
        adc_bits (int): 
            ADC resolution.
        full_scale (float): 
            Detector full scale in empty-belt units.
        seed (int): 
            Seed of all random draws.

    Returns:
        dict: x_mm, y_mm (output grid), images {'fo'|'base': {n_sheets: (Ny', Nx') array}} and clean {'fo'|'base'|'empty': array}
            (after resampling/aperture, before paper, noise and ADC).

    Raises:
        ValueError: If the input arrays have incompatible shapes.
    """
    # Import necessary modules
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import gaussian_filter

    # Prepare the output grid and resampling flags
    x_mm, y_mm = np.asarray(x_mm, float), np.asarray(y_mm, float)
    x_c = np.arange(x_mm[0], x_mm[-1] + 1e-9, step_mm) if step_mm and len(x_mm) > 1 else x_mm
    y_c = np.arange(y_mm[0], y_mm[-1] + 1e-9, pitch_mm) if pitch_mm and len(y_mm) > 1 else y_mm
    resample = (len(x_c) != len(x_mm)) or (len(y_c) != len(y_mm)) \
        or not (np.allclose(x_c, x_mm) and np.allclose(y_c, y_mm))
    dx = float(np.mean(np.diff(x_c))) if len(x_c) > 1 else 1.0
    dy = float(np.mean(np.diff(y_c))) if len(y_c) > 1 else 1.0

    # Define a helper function to prepare each input array (resample, apply aperture and smear)
    def prep(A):
        B = np.asarray(A, float)
        if resample and min(len(x_mm), len(y_mm)) > 1:
            
            k = min(3 if interp == 'cubic' else 1, len(x_mm) - 1, len(y_mm) - 1)
            B = np.clip(RectBivariateSpline(y_mm, x_mm, B, kx=k, ky=k)(y_c, x_c), 0.0, None)

        elif resample: 
            B = np.stack([np.interp(y_c, y_mm, B[:, i]) for i in range(B.shape[1])], axis=1)
        sy = aperture_mm / 2.3548 / dy if aperture_mm and dy > 0 else 0.0
        sx = smear_x_mm / 2.3548 / dx if smear_x_mm and dx > 0 else 0.0
        if sx > 0 or sy > 0:
            B = gaussian_filter(B, sigma=(sy, sx), mode='nearest')
        return B

    # Prepare the clean images dictionary by applying the prep function to each input array
    clean = {k: prep(A) for k, A in (('fo', I_fo), ('base', I_base), ('empty', I_empty)) if A is not None}

    # Prepare the noisy images by applying gain, offset, and noise to the clean images
    rng_fp = np.random.default_rng(seed)
    gain = 1.0 + gain_spread * rng_fp.standard_normal(len(y_c))
    offs = offset_rel * rng_fp.standard_normal(len(y_c))
    levels = 2 ** int(adc_bits) - 1
    images = {c: {} for c in ('fo', 'base') if c in clean}

    # Loop over each case ('fo' and 'base') and each number of sheets to generate noisy images
    for ci, case in enumerate(('fo', 'base')):
        if case not in clean:
            continue
        for n in sheets:
            rng = np.random.default_rng([seed, 1 + ci, int(n)])
            S = clean[case] * 10 ** (-n * atten_db_per_sheet / 10)
            v = gain[:, None] * S + offs[:, None] + noise_rel * \
                rng.standard_normal(S.shape)
            v = np.clip(v, 0.0, full_scale)
            images[case][int(n)] = np.round(v / full_scale * levels) / levels * full_scale

    return dict(x_mm=x_c, y_mm=y_c, images=images, clean=clean)


def analyze_moving_sample_camera(study_dir: str, 
                                 output_dir: str = None,
                                 sheets=(0, 10, 20), 
                                 atten_db_per_sheet: float = 0.5,
                                 freq_ghz: float = None, 
                                 aperture_mm: float = 1.5,
                                 smear_x_mm: float = 0.0, 
                                 pitch_mm: float = None,
                                 step_mm: float = None, 
                                 interp: str = 'cubic',
                                 noise_rel: float = 0.02, 
                                 gain_spread: float = 0.05,
                                 offset_rel: float = 0.01, 
                                 adc_bits: int = 12,
                                 full_scale: float = 1.0, 
                                 seed: int = 1,
                                 display: str = 'fixed', 
                                 cmap: str = 'jet',
                                 show: bool = False, 
                                 scan: dict = None,
                                 reference_power: float = None, reference_power_db: float = None) -> dict:
    """This function simulates what a THz line camera with power detectors would capture based on the MODE 12 scan (no simulation, band-averaged power |H|^2 at the RX).

    Args:
        study_dir (str): 
            MODE 12 study folder:
        output_dir (str): 
            Output folder.
        sheets (tuple): 
            Numbers of paper sheets.
        atten_db_per_sheet (float): 
            Placeholder power damping per sheet IN dB
        freq_ghz (float): 
            Evaluate only this frequency (nearest bin); None = band-mean power of the simulated band (that of the MODE 12 parameter file).
        aperture_mm, smear_x_mm, pitch_mm, step_mm, interp, noise_rel, gain_spread, offset_rel, adc_bits, full_scale, seed: 
            See _msla_camera_model().
        display (str): 
            'fixed' = every image on 0 .. full_scale; 'auto' = every image scaled to its own min/max.
        cmap (str): 
            Colormap (default 'jet').
        show (bool): 
            plt.show() the figures.
        scan (dict): 
            Already loaded scan (tests); None = _msla_load_scan().
        reference_power (float): 
            Reference power in |H|^2 units of the transfer function (e.g. a measured empty-belt level); None = automatic.
        reference_power_db (float): 
            The same reference as 10*log10(|H|^2) [dB]; set at most one of reference_power / reference_power_db.

    Returns:
        dict: The _msla_camera_model() result plus the object contrast table.

    Raises:
        ValueError: If both reference_power and reference_power_db are set, or if reference_power is not positive.
    """

    # Prepare output directory and load scan data.
    study_dir = Path(study_dir)
    out_dir = Path(output_dir) if output_dir else study_dir / 'results_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scan data if not provided.
    scan = _msla_load_scan(study_dir) if scan is None else scan
    cfg = scan['config']
    f_hz = np.asarray(scan['f_hz'], float)

    # Select frequency bins based on the requested frequency.
    if freq_ghz is None:
        sel = np.arange(len(f_hz))
    else:
        sel = np.array([int(np.argmin(np.abs(f_hz - 1e9 * freq_ghz)))])
        if abs(f_hz[sel[0]] - 1e9 * freq_ghz) > 0.02 * 1e9 * freq_ghz:
            print(f"analyze_moving_sample_camera(): WARNING - nearest simulated frequency is {f_hz[sel[0]] / 1e9:.2f} GHz, not {freq_ghz:g} GHz (the MODE == 12 band is {f_hz[0] / 1e9:.1f}..{f_hz[-1] / 1e9:.1f} GHz).")
    print(f"analyze_moving_sample_camera(): power over {len(sel)} frequency bin(s), {f_hz[sel[0]] / 1e9:.2f}..{f_hz[sel[-1]] / 1e9:.2f} GHz.")

    # Define a helper function to compute the power for the forward, base, and empty scans (M, K).
    def power(H):
        return np.mean(np.abs(np.asarray(H)[..., sel]) ** 2, axis=-1)

    # Compute the power for the forward, base, and empty scans (M, K).
    P_f, P_b, P_e = (None if scan[k] is None else power(scan[k]) for k in ('H_fo', 'H_base', 'H_empty'))

    # Extract and sort the beam and receiver positions.
    x_beam = -np.asarray(scan['offsets_mm'], float)
    xo = np.argsort(x_beam)
    y_rx = np.asarray(scan['rx_y_mm'], float)
    yo = np.argsort(y_rx)

    # Handle the reference power for normalization.
    if reference_power_db is not None:
        if reference_power is not None:
            raise ValueError("analyze_moving_sample_camera(): set reference_power or reference_power_db, not both.")
        reference_power = 10 ** (float(reference_power_db) / 10)

    # Determine the reference power based on user input or empty-belt measurements.
    if reference_power is not None:

        if not reference_power > 0:
            raise ValueError(f"analyze_moving_sample_camera(): reference_power must be > 0, got {reference_power!r}.")
        ref, ref_label = float(reference_power), 'user-set reference power'
        
        if P_e is not None:
            print(f"analyze_moving_sample_camera(): user-set reference = {ref / float(np.median(P_e)):.4g} x median empty-belt power.")

    elif P_e is not None:

        ref, ref_label = float(np.median(P_e)), 'median empty-belt power'

    else:
        
        XX, YY = np.meshgrid(x_beam, y_rx, indexing='ij')
        if cfg.get('sut_outline_mm'):
            from matplotlib.path import Path as _MplPath
            belt = ~_MplPath(np.array(cfg['sut_outline_mm']).T).contains_points(
                np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        elif cfg.get('sut_shape') == 'CYLINDER':
            belt = np.hypot(XX, YY) > cfg['sut_size']['DIAMETER'] * 500
        else:
            belt = (np.abs(XX) > cfg['sut_size'][0] * 500) | (np.abs(YY) > cfg['sut_size'][1] * 500)
        ref, ref_label = ((float(np.median(P_f[belt])), f"median FO power outside the cookie ({int(belt.sum())} pixels)")
                          if belt.any() else (float(P_f.max()), 'maximum FO power (no pixel outside the cookie)'))
        print(f"analyze_moving_sample_camera(): no empty-belt runs - reference = {ref_label}.")
    print(f"analyze_moving_sample_camera(): reference power {ref:.4g} = {10 * 
    np.log10(ref):.2f} dB ({ref_label}).")

    # Define a helper function to convert power measurements to images normalized by the reference power (Ny, Nx).
    to_img = lambda P: None if P is None else (P / ref)[xo][:, yo].T            

    # Run the camera model simulation with the normalized images.
    res = _msla_camera_model(to_img(P_f), 
                             to_img(P_b), 
                             to_img(P_e), 
                             x_beam[xo], 
                             y_rx[yo],
                             sheets=sheets, atten_db_per_sheet=atten_db_per_sheet,
                             aperture_mm=aperture_mm, smear_x_mm=smear_x_mm, pitch_mm=pitch_mm,
                             step_mm=step_mm, interp=interp, noise_rel=noise_rel,
                             gain_spread=gain_spread, offset_rel=offset_rel, adc_bits=adc_bits,
                             full_scale=full_scale, seed=seed)
    xg, yg = res['x_mm'], res['y_mm']

    # Define a helper function to compute the edges of the grid for plotting.
    def edges(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        mid = (v[1:] + v[:-1]) / 2
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    xe, ye = edges(xg), edges(yg)
    fx, fy, _ = cfg['fo_center_cookie_frame_mm']
    fo_size = float(cfg['fo_size_mm'])

    # Define a helper function to draw the outline of the sample on the plot.
    def outline(ax):
        if cfg.get('sut_outline_mm'):
            _msla_draw_freeform_outline(ax, cfg, color='0.25', lw=0.6, ls=':')
        elif cfg.get('sut_shape') == 'CYLINDER':
            ax.add_patch(plt.Circle((0, 0), cfg['sut_size']['DIAMETER'] * 500, fill=False, ec='0.25', lw=0.6, ls=':'))
        else:
            sx, sy = cfg['sut_size'][0] * 1e3, cfg['sut_size'][1] * 1e3
            ax.add_patch(Rectangle((-sx / 2, -sy / 2), sx, sy, fill=False, ec='0.25', lw=0.6, ls=':'))

    # Determine the number of sheets and the cases to display in each row.
    nS = len(sheets)
    row_cases = [c for c in ('fo', 'base') if c in res['images']]

    # Create the figure and axes for the camera images.
    fig, axs = plt.subplots(len(row_cases), nS, figsize=(3.0 * nS + 1.2, 3.2 * len(row_cases)), squeeze=False)

    # Loop over each sheet and each case to plot the camera images.
    for j, n in enumerate(sheets):
        for i, case in enumerate(row_cases):
            ax = axs[i, j]
            img = res['images'][case][int(n)]
            vmin, vmax = (0.0, full_scale) if display == 'fixed' else (float(img.min()), float(img.max()))
            ax.pcolormesh(xe, ye, img, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-12), shading='flat')
            outline(ax)
            if case == 'fo':
                ax.add_patch(plt.Circle((fx, fy), fo_size / 2 + 2.0, fill=False, ec='k', lw=1.3))
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        axs[0, j].set_title(f"{int(n)}p")

    # Label the rows with the case names.
    for i, case in enumerate(row_cases):
        axs[i, 0].set_ylabel('with object' if case == 'fo' else 'without object')

    # Set the overall title and save the figure. Save the figure to the output directory.
    fig.suptitle(f"Camera model: {cfg['cookie_name']} | {fo_size:g} mm {cfg['fo_material']} | {atten_db_per_sheet:g} dB/sheet, aperture {aperture_mm:g} mm, noise {noise_rel:g}, gain {gain_spread:g}, {adc_bits} bit, display {display}", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out_dir / 'moving_sample_camera_images.png', dpi=170)

    # Determine the indices corresponding to the cookie center in the grid.
    iy = int(np.argmin(np.abs(yg))); ix = int(np.argmin(np.abs(xg)))
    
    fig2, a2 = plt.subplots(1, 2, figsize=(12, 4))

    # Loop over the different cases and plot the noise-free profiles through the cookie center.
    for key, lab in (('empty', 'empty belt'), ('base', 'cookie'), ('fo', 'cookie + object')):
        if key not in res['clean']:
            continue
        a2[0].plot(xg, res['clean'][key][iy, :], label=lab)
        a2[1].plot(yg, res['clean'][key][:, ix], label=lab)
    a2[0].set_xlabel(f'x in the cookie frame [mm]  (row y = {yg[iy]:+.1f} mm)')
    a2[1].set_xlabel(f'y in the cookie frame [mm]  (column x = {xg[ix]:+.1f} mm)')

    # Label the axes and add grid and legend for the noise-free profiles. Save the figure to the output directory.
    for a in a2:
        a.set_ylabel(f"power / {ref_label} (noise free, before paper and detector)")
        a.grid(alpha=0.3); a.legend()
    fig2.tight_layout()
    _savefig(fig2, out_dir / 'moving_sample_camera_profiles.png', dpi=150)

    # Create a mask for the object footprint and handle the case where the footprint is smaller than a pixel.
    XX, YY = np.meshgrid(xg, yg)
    mask = (np.abs(XX - fx) <= fo_size / 2 + 1e-9) & (np.abs(YY - fy) <= fo_size / 2 + 1e-9)
    
    # If the mask is empty (footprint smaller than a pixel), set the mask to cover the nearest pixel.
    if not mask.any():
        mask = np.zeros_like(XX, bool)
        mask[int(np.argmin(np.abs(yg - fy))), int(np.argmin(np.abs(xg - fx)))] = True    # footprint smaller than a pixel
    rows = []
    has_b = 'base' in res['images']

    # Loop over the different sheets and calculate the mean signal within the object footprint.
    for n in sheets:
        a_f = float(res['images']['fo'][int(n)][mask].mean())
        a_b = float(res['images']['base'][int(n)][mask].mean()) if has_b else np.nan
        img_sat = res['images']['base' if has_b else 'fo'][int(n)]
        rows.append(dict(sheets=int(n), atten_db=n * atten_db_per_sheet, mean_with_object=a_f,
                         mean_without_object=a_b, contrast_ratio=a_f / a_b if a_b > 0 else np.nan,
                         image_mean_without_object=float(img_sat.mean()) if has_b else np.nan,
                         saturated_fraction=float((img_sat >= full_scale - 1e-9).mean())))
    # Convert the list of dictionaries to a DataFrame for easier analysis and save it as a CSV file. Save the summary table as well.
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / 'moving_sample_camera_summary.csv', index=False)
    print("analyze_moving_sample_camera(): object contrast (mean over the object footprint, "
          "with / without object):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    np.savez(out_dir / 'moving_sample_camera_images.npz', x_mm=xg, y_mm=yg,
             **{f"{c}_{int(n)}p": res['images'][c][int(n)] for c in res['images'] for n in sheets},
             **{f"clean_{k}": v for k, v in res['clean'].items()})

    # Save the analysis parameters to a JSON file.
    with open(out_dir / 'moving_sample_camera_params.json', 'w') as fh:
        json.dump(dict(sheets=list(map(int, sheets)), 
                       atten_db_per_sheet=atten_db_per_sheet, 
                       freq_ghz=freq_ghz,
                       aperture_mm=aperture_mm, 
                       smear_x_mm=smear_x_mm, 
                       pitch_mm=pitch_mm, 
                       step_mm=step_mm,
                       interp=interp, 
                       noise_rel=noise_rel, 
                       gain_spread=gain_spread, 
                       offset_rel=offset_rel,
                       adc_bits=adc_bits, 
                       full_scale=full_scale, 
                       seed=seed, 
                       display=display,
                       reference_power=ref, 
                       reference_power_db=float(10 * np.log10(ref)), reference=ref_label), 
                       fh, 
                       indent=1)

    # Show or close the plots based on the 'show' flag.
    if show:
        plt.show(block=False)
    else:
        plt.close('all')
    res['contrast'] = table
    
    return res
