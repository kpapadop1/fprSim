"""Running gprMax and turning its output into results (SolverMixin): run folders, gprMax calls, NF2FF field reconstruction, H_tf, saving, protocols, simulate()."""
import copy
import json
import os
import pickle
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import scipy as sp


class SolverMixin:
    """This mixing class provides methods of EMSimulation with regard to running gprMax simulations and handling their results."""

    def _resolve_existing_results_folder(self) -> None:
        """This function determines the folder containing existing simulation results based on the provided parameters or the most recently modified subfolder.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: If no folder can be resolved or it lacks the expected .in/.out files.
        """
        # Determine the candidate folder for existing results. If self.existing_results_folder is set, use it; otherwise, use the most recently modified subfolder.
        if self.existing_results_folder:

            candidate = Path(self.existing_results_folder)
            if not candidate.is_absolute():
                candidate = Path(self.path_input_output_files) / candidate

        else:

            root = Path(self.path_input_output_files)
            if not root.is_dir():
                raise RuntimeError(
                    "_resolve_existing_results_folder(): "
                    f"{root} does not exist - set parms['EXISTING_RESULTS_FOLDER'] "
                    "or run with SIMULATE_BEFORE_ANALYSIS=True first.")
            
            subfolders = [d for d in root.iterdir() if d.is_dir()]
            if not subfolders:
                raise RuntimeError(
                    "_resolve_existing_results_folder(): no run folders found in {root} - set parms['EXISTING_RESULTS_FOLDER'] or run with SIMULATE_BEFORE_ANALYSIS=True first.")
            candidate = max(subfolders, key=lambda d: d.stat().st_mtime)
            print("_resolve_existing_results_folder(): no EXISTING_RESULTS_FOLDER "
                  f"given - using most recently modified folder: {candidate}")

        if not candidate.is_dir():

            raise RuntimeError(
                f"_resolve_existing_results_folder(): {candidate} is not a "
                "folder - check parms['EXISTING_RESULTS_FOLDER'].")

        # Get the list of .in and output files in the candidate folder.
        in_files = self._get_files_folder(candidate, 'in')
        out_files = self._get_output_files_folder(candidate)

        # Check that the number of .in and output files matches the expected number of positions (n_pos).
        if len(in_files) != self.n_pos or len(out_files) != self.n_pos:
            raise RuntimeError(
                f"_resolve_existing_results_folder(): expected exactly {self.n_pos} .in and {self.n_pos} output (.out/.h5) file(s) in {candidate} (from parms['N_TX'] - one gprMax run per TX firing), found {len(in_files)} .in and {len(out_files)} output - this folder doesn't match the current N_TX (or is a partial/failed simulation).")
        
        # Check that the first .in file matches the current simulation parameters.
        self._check_in_file_matches_current_parms(in_files[0])

        self.path_folder = candidate
        print(f"_resolve_existing_results_folder(): using {self.path_folder}")

    def _check_in_file_matches_current_parms(self, path_inputfile: str) -> None:
        """This function reads the .in file and extracts the "#domain" and "#dx_dy_dz" header lines. Then, it compares them to the current simulation parameters.

        Args:
            path_inputfile (str): 
                Path to one .in file of the candidate results folder.
        
        Returns:
            None

        Raises:
            RuntimeError: If the header can't be parsed or doesn't match the current parms within tolerance.
        """
        # Read the .in file and extract the #domain: and #dx_dy_dz: header lines.
        text = Path(path_inputfile).read_text()
        domain_match = re.search(
            r'^#domain:\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', 
            text, re.M)
        dxdydz_match = re.search(
            r'^#dx_dy_dz:\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', 
            text, re.M)

        # Check that both header lines were found.
        if not domain_match or not dxdydz_match:
            raise RuntimeError(
                "_check_in_file_matches_current_parms(): could not find #domain:/#dx_dy_dz: header lines in {path_inputfile} - not a gprMax input file written by _generate_simulation_input_file(), or its format changed.")

        # Convert the found header values to tuples of floats for comparison.
        found_domain = tuple(float(v) for v in domain_match.groups())
        found_dxdydz = tuple(float(v) for v in dxdydz_match.groups())
        expected_domain = (self.l_x_grid, self.l_y_grid, self.l_z_grid)
        expected_dxdydz = (self.dx, self.dy, self.dz)

        # Compare the found values with the expected values using tolerances.
        domain_ok = np.allclose(found_domain, 
                                expected_domain, 
                                rtol=1e-3, 
                                atol=1e-4)
        dxdydz_ok = np.allclose(found_dxdydz, 
                                expected_dxdydz, 
                                rtol=1e-6, 
                                atol=1e-12)
        
        # Raise an error if the domain or dx_dy_dz values do not match the expected values.
        if not domain_ok or not dxdydz_ok:
            raise RuntimeError(
                f"_check_in_file_matches_current_parms(): {path_inputfile} was generated with #domain: {found_domain} #dx_dy_dz: {found_dxdydz}, but the current parms give domain {expected_domain} dx_dy_dz {expected_dxdydz} - this results folder was built with different L_X_GRID/L_Y_GRID/L_Z_GRID/CELLS_PER_WAVELENGTH/FREQUENCY_RANGE settings than the ones in effect now. Fix parms['EXISTING_RESULTS_FOLDER'] to point at the right folder, or re-run with SIMULATE_BEFORE_ANALYSIS=True.")

    def _get_files_folder(self, folder: str = None, ending: str = 'txt') -> list:
        """This function helps in retrieving all files with a specific ending from a given folder (sorted alphabetically).

        Args:
            folder (str): 
                Path to folder (default: None).
            ending (str): 
                File ending, without the dot (default: 'txt').

        Returns:
            list: Sorted full paths (as strings).

        Raises:
            None
        """
        folder = Path(folder)
        results = [str(file) for file in sorted(folder.glob("*." + ending))]

        return results

    def _get_output_files_folder(self, folder: str = None) -> list:
        """This function first tries to find '.out' files in the folder. If none are found, it falls back to '.h5' files.

        Args:
            folder (str): Path to folder (default: None).

        Returns:
            list: Sorted full paths (as strings) to the output files.

        Raises:
            None
        """
        out_files = self._get_files_folder(folder, 'out')
        if out_files:
            return out_files
        
        return self._get_files_folder(folder, 'h5')

    def _find_msvc_bin_dir(self) -> str:
        """This function attempts to locate the MSVC compiler required by nvcc on Windows.

        Args:
            None

        Returns:
            str: Directory containing cl.exe.

        Raises:
            RuntimeError: If MSVC cannot be located (raised before launching gprMax).
        """
        # Return the user-specified MSVC bin path if available.
        if self.msvc_bin_path is not None:
            return self.msvc_bin_path

        # Check if vswhere.exe exists at the standard location. Otherwise raise an error.
        vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
        if not vswhere.exists():
            raise RuntimeError(
                f"_find_msvc_bin_dir(): vswhere.exe not found at the standard location ({vswhere}) - Visual Studio Installer doesn't appear to be present. Install Visual Studio Build Tools with the 'Desktop development with C++' workload (https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio), or set parms['MSVC_BIN_PATH'] explicitly if MSVC is installed somewhere vswhere can't find it.")

        # Use vswhere.exe to find the latest Visual Studio installation with the required C++ tools.
        result = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True)

        # Extract the installation path from vswhere.exe output.
        vs_install_path = result.stdout.strip()
        if not vs_install_path:
            raise RuntimeError(
                "_find_msvc_bin_dir(): vswhere.exe found no Visual Studio installation with the C++ build tools component (Microsoft.VisualStudio.Component.VC.Tools.x86.x64). Install it via the Visual Studio Installer ('Desktop development with C++' workload), or set parms['MSVC_BIN_PATH'] explicitly.")

        # Look for cl.exe under the MSVC installation path.
        candidates = sorted(Path(vs_install_path).glob(
            "VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if not candidates:
            raise RuntimeError(
                f"_find_msvc_bin_dir(): found Visual Studio at '{vs_install_path}' but no cl.exe under its VC/Tools/MSVC/*/bin/Hostx64/x64/ tree - the C++ build tools component may be incomplete. Set parms['MSVC_BIN_PATH'] explicitly if you know the correct path.")

        # Pick the highest version of cl.exe found.
        msvc_bin_dir = str(candidates[-1].parent)
        print(f"_find_msvc_bin_dir(): using MSVC compiler at "
              f"'{msvc_bin_dir}' (from VS install '{vs_install_path}').")
        
        return msvc_bin_dir

    def _simulate_model(self) -> None:
        """This method runs the gprMax simulation for each input file, optionally using the GPU solver if self.use_gpu is True.

        Args:
            None

        Returns:
            None

        Raises:
            FileNotFoundError: If the gprMax virtual environment interpreter is not found.
            subprocess.CalledProcessError: If the gprMax simulation fails for any input file.
        """
        # Check if the gprMax virtual environment interpreter exists.
        if not Path(self.gprmax_venv_python).is_file():
            raise FileNotFoundError(
                f"_simulate_model(): gprMax venv interpreter not found at {self.gprmax_venv_python} - set GPRMAX_PYTHON in EMSimulationSettings.py or use the folder layout <parent>/'EM Simulation'/.venv/Scripts/python.exe next to this script's folder.")

        # Get current date and time for adding it to the input and output filename
        input_files = self._get_files_folder(self.path_folder, 'in')

        # Prepare GPU-related settings if GPU mode is enabled.
        gpu_suffix = ""
        popen_env = None
        if self.use_gpu:

            gpu_suffix = f" [GPU device(s): {self.gpu_device if self.gpu_device is not None else 0}]"

            # Required, if using GPU mode: nvcc needs cl.exe on PATH as host compiler
            msvc_bin_dir = self._find_msvc_bin_dir()
            popen_env = os.environ.copy()
            popen_env['PATH'] = msvc_bin_dir + os.pathsep + popen_env.get('PATH', '')

        # Run the simulation for each input file.
        for i in range(self.n_pos):

            print(f"simulate_model(): running simulation {i+1}/{self.n_pos}{gpu_suffix}...")

            # Construct the command to run gprMax with the current input file.
            cmd = [self.gprmax_venv_python, "-m", "gprMax", input_files[i]]
            if self.use_gpu:
                cmd.append("-gpu")
                if self.gpu_device is not None:
                    if isinstance(self.gpu_device, (list, tuple)):
                        cmd.extend(str(device_id) for device_id in self.gpu_device)
                    else:
                        cmd.append(str(self.gpu_device))

            # Launch the gprMax simulation as a subprocess with the constructed command.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=popen_env)

            # Stream gprMax's output
            for line in process.stdout:
                print(line, end='')

            # Wait for the simulation process to complete.
            process.wait()

            # Check if the simulation process exited successfully.
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode,
                                                     process.args)

            print(f"simulate_model(): simulation {i+1}/{self.n_pos} completed.")

        print("simulate_model(): Results saved.")

    def _sync_time_parameters_from_output(self) -> None:
        """This function reads the actual time step and iteration count from the first output file and updates the corresponding attributes of the solver if they differ from the assumed values.

        Args:
            None
        
        Returns:
            None

        Raises:
            RuntimeError: If no .out or .h5 files are found in self.path_folder.
        """
        # Get the list of output files in the folder. If none are found, raise an error.
        out_files = self._get_output_files_folder(self.path_folder)
        if not out_files:
            raise RuntimeError(
                f"_sync_time_parameters_from_output(): no .out or .h5 files found in {self.path_folder} - run _simulate_model() first.")

        # Open the first output file to read the actual time step and iteration count.
        with h5py.File(out_files[0], "r") as f:
            actual_dt = float(f.attrs["dt"])
            actual_n_time = int(f.attrs["Iterations"])

        # Compare the actual values with the assumed ones and update if necessary. Then regenerate the excitation waveform.
        if actual_dt != self.dt or actual_n_time != self.n_time:
            print(f"_sync_time_parameters_from_output(): assumed dt={self.dt:.6e}, n_time={self.n_time} but {out_files[0]} actually used dt={actual_dt:.6e}, n_time={actual_n_time} - resyncing and rebuilding the excitation waveform.")
            self.dt = actual_dt
            self.n_time = actual_n_time
            self.time = np.arange(self.n_time) * self.dt
            self._generate_waveform()

    def _determine_electrical_fields_at_receiver(self, show_results: bool = False, reuse_tangential_fields: bool = False) -> None:
        """This function reconstructs the electric field at the receiver antennas using the equivalence principle and the simulated Huygens-surface currents by building equivalent surface currents and evaluating them at the RX positions.

        Args:
            show_results (bool): 
                Whether to show the results (default: False).
            reuse_tangential_fields (bool): 
                Skip reading the .out files and use the E_tan_per_tx/H_tan_per_tx/f_huygens/band_mask of the previous call (only the RX geometry changed, see _save_mirrored_rx_results()); default False.

        Returns:
            None

        Raises:
            RuntimeError: If the required .out files are not found and reuse_tangential_fields is False.
        """

        def _equivalent_currents(E_tan, H_tan, n_hat):
            """This function implements the equivalence principle to compute the equivalent surface currents from the tangential electric and magnetic fields and the surface normal (Js = n_hat x H_tan, Ms = -n_hat x E_tan).

            Args:
                E_tan (array): 
                    Tangential electric field in V/m.
                H_tan (array): 
                    Tangential magnetic field in A/m.
                n_hat (array): 
                    Normal vector (dimensionless).

            Returns:
                Js (array): 
                    Electric surface current density in A/m.
                Ms (array): 
                    Magnetic surface current density in V/m.
            """
            E_tan = np.asarray(E_tan, dtype=complex)
            H_tan = np.asarray(H_tan, dtype=complex)
            n_hat = np.asarray(n_hat, dtype=float)
            Js = np.cross(n_hat, H_tan)
            Ms = -np.cross(n_hat, E_tan)

            return Js, Ms

        def _radiated_field_geometry(r_obs, r_src, mode="exact"):
            """This function precomputes the geometry between observation points and source points, which is independent of frequency.

            Args:
                r_obs (array): 
                    (3,) or (M,3) array, observation (receiver) point(s) in m.
                r_src (array): 
                    (N,3) array, source points on the Huygens surface in m.
                mode (str): 
                    "exact" or "farfield".

            Returns:
                list of per-observation-point tuples: 
                    "exact": (Rhat (N,3), R (N,)); 
                    "farfield": (Rhat (N,3), R_phase (N,), r0 (float)).

            Raises:
                ValueError: If mode is not "exact" or "farfield".
            """
            # Ensure input arrays are of correct shape and type.
            r_obs = np.atleast_2d(np.asarray(r_obs, dtype=float))
            r_src = np.asarray(r_src, dtype=float)

            # For each observation point, compute the geometry relative to all source points.
            geometry = []
            for robs in r_obs:
                if mode == "exact":
                    Rvec = robs[None, :] - r_src
                    R = np.linalg.norm(Rvec, axis=1)
                    Rhat = Rvec / R[:, None]
                    geometry.append((Rhat, R))
                elif mode == "farfield":
                    r0 = np.linalg.norm(robs)
                    r_hat = robs / r0
                    Rhat = np.tile(r_hat, (r_src.shape[0], 1))
                    R_phase = r0 - r_src @ r_hat
                    geometry.append((Rhat, R_phase, r0))
                else:
                    raise ValueError("mode must be 'exact' or 'farfield'")

            return geometry

        def _radiated_E_field(geometry, Js, Ms, dS, k, eta, mode="exact"):
            """This function computes the radiated E field at the observation points using the precomputed geometry and the surface currents (for a wavenumber k) - in exact or farfield mode.

            Args:
                geometry (list of tuples): 
                    List of geometry tuples from _radiated_field_geometry(), one per observation point.
                Js (complex array): 
                    (N,3) complex array, electric surface current density J_s = n_hat x H [A/m].
                Ms (complex array): 
                    (N,3) complex array, magnetic surface current density M_s = -n_hat x E [V/m].
                dS (array): 
                    (N,) array, area element per source patch [m^2].
                k (float): 
                    wavenumber 2*pi*f/c [rad/m].
                eta (float): 
                    wave impedance of the medium at the receiver [Ohm].
                mode (str): 
                    "exact" or "farfield".

            Returns:
                E: (3,) or (M,3) complex array, radiated E field at r_obs inV/m.

            Raises:
                ValueError: If mode is not "exact" or "farfield".
            """
            Js = np.asarray(Js, dtype=complex)
            Ms = np.asarray(Ms, dtype=complex)
            dS = np.asarray(dS, dtype=float)

            # Loop over observation points and compute the radiated E field using the precomputed geometry - exact or farfield depending on the mode.
            E_out = np.zeros((len(geometry), 3), dtype=complex)
            for i, geom in enumerate(geometry):

                if mode == "exact":

                    # Exact field computation: Rhat is the unit vector pointing from the source to the observation point, R is the distance.
                    Rhat, R = geom
                    kR = k * R
                    green = np.exp(-1j * kR) / (4 * np.pi * R)

                    # Compute the electric dipole moments for each patch.
                    pJ = Js * dS[:, None]
                    pJ_R = np.sum(pJ * Rhat, axis=1)[:, None] * Rhat
                    pJ_T = pJ - pJ_R
                    bracket_T = 1 + 1 / (1j * kR) - 1 / kR**2
                    bracket_L = 1 + 1 / (1j * kR)

                    # Compute the electric field contribution from the electric dipole moments.
                    E_J = (eta * green)[:, None] * ((2.0 / R * bracket_L)[:, None] * pJ_R - (1j * k * bracket_T)[:, None] * pJ_T)

                    # Compute the magnetic field contribution from the magnetic dipole moments.
                    pM = Ms * dS[:, None]
                    E_M = (1j * k * green * bracket_L)[:, None] * np.cross(Rhat, pM)

                    integrand_sum = E_J + E_M

                elif mode == "farfield":

                    # Far-field approximation: Rhat is the unit vector pointing from the source to the observation point, R_phase is the phase distance, r0 is the reference distance for the Green's function.
                    Rhat, R_phase, r0 = geom
                    green = np.exp(-1j * k * R_phase) / (4 * np.pi * r0)

                    # Compute the far-field contributions from the electric and magnetic dipole moments.
                    Js_cross_R = np.cross(Js, Rhat)
                    JJ = np.cross(Rhat, Js_cross_R)

                    # Compute M-ter (R_hat x Ms)
                    MM = np.cross(Rhat, Ms)

                    # Compute the far-field integrand as the combination of M-term and J-term contributions.
                    integrand_sum = 1j * k * (MM - eta * JJ) * green[:, None] * dS[:, None]

                else:
                    raise ValueError("mode must be 'exact' or 'farfield'")

                E_out[i] = integrand_sum.sum(axis=0)

            return E_out if E_out.shape[0] > 1 else E_out[0]

        # Synchronize the time step, number of time steps, and transmitted field to match what gprMax actually used.
        self._sync_time_parameters_from_output()

        # Determine the list of output files to process, depending on whether tangential fields are being reused.
        output_files = [] if reuse_tangential_fields else self._get_output_files_folder(self.path_folder)
        if reuse_tangential_fields and (not getattr(self, 'E_tan_per_tx', None) or getattr(self, 'f_fft_huygens', None) is None):
            raise RuntimeError("_determine_electrical_fields_at_receiver(): reuse_tangential_fields=True needs a previous full call.")

        # Use the per-point patch-area array for the Huygens surface directly.
        dS = self.dS_huygens

        # Initialize the per-TX storage for tangential fields if not reusing them.
        if not reuse_tangential_fields:
            self.E_tan_per_tx = []
            self.H_tan_per_tx = []
            self.f_fft_huygens = None  # frequency axis, same for every file/rx (set on first pass)
        n_files = len(output_files)
        print(f"_determine_electrical_fields_at_receiver(): reading and FFTing "
              f"{n_files} output file(s)...")
        t_start_read = time.time()

        # Loop over each output file to read the tangential fields and perform FFT.
        for file_idx, path in enumerate(output_files):

            E_tan_this_file = []
            H_tan_this_file = []

            # Read the tangential fields for each Huygens point in this file and perform FFT.
            with h5py.File(path, "r") as f:

                rx_names = sorted(f['rxs'].keys(),
                                key=lambda name: int(name.replace('rx', '')))
                
                # Get the number of Huygens recording points for this file.
                n_huygens_pts = len(rx_names)
                rx_progress_step = max(1, n_huygens_pts // 100)

                # Loop over each Huygens recording point within this file.
                for rx_idx, rx_name in enumerate(rx_names):

                    # Extract the tangential electric and magnetic fields for this Huygens point.
                    grp = f['rxs'][rx_name]
                    Ex, Ey, Ez = grp['Ex'][:], grp['Ey'][:], grp['Ez'][:]
                    Hx, Hy, Hz = grp['Hx'][:], grp['Hy'][:], grp['Hz'][:]

                    # Perform FFT on the extracted fields to obtain frequency-domain representation.
                    if self.f_fft_huygens is None:
                        self.f_fft_huygens = np.fft.rfftfreq(len(Ex), d=self.dt)
                        # Restrict to bins within the excitation bandwidth
                        self.band_mask = ((self.f_fft_huygens >= self.frequency_range[0]) &
                                        (self.f_fft_huygens <= self.frequency_range[1]))
                        if self.frequency_bands:
                            self.band_mask = np.zeros_like(self.band_mask)
                            for f_lo, f_hi in self.frequency_bands:
                                self.band_mask |= (self.f_fft_huygens >= f_lo) & (self.f_fft_huygens <= f_hi)

                    # Apply the frequency band mask to keep only the relevant frequency bins.
                    E_fft = np.array([np.fft.rfft(Ex)[self.band_mask],
                                    np.fft.rfft(Ey)[self.band_mask],
                                    np.fft.rfft(Ez)[self.band_mask]]).T   # (n_freq, 3)
                    H_fft = np.array([np.fft.rfft(Hx)[self.band_mask],
                                    np.fft.rfft(Hy)[self.band_mask],
                                    np.fft.rfft(Hz)[self.band_mask]]).T   # (n_freq, 3)

                    E_tan_this_file.append(E_fft)
                    H_tan_this_file.append(H_fft)

                    if (rx_idx + 1) % rx_progress_step == 0 or rx_idx == n_huygens_pts - 1:
                        print(f"_determine_electrical_fields_at_receiver(): file {file_idx + 1}/{n_files} - Huygens point {rx_idx + 1}/{n_huygens_pts} ({100 * (rx_idx + 1) / n_huygens_pts:.0f}%) FFT'd [{time.time() - t_start_read:.1f} s elapsed]")

            # shape (n_huygens_pts, n_freq, 3)
            self.E_tan_per_tx.append(np.array(E_tan_this_file))
            self.H_tan_per_tx.append(np.array(H_tan_this_file))

        self.f_huygens = self.f_fft_huygens[self.band_mask]
        n_freq = len(self.f_huygens)

        # If RX mode is 'DIRECT', the recorded fields are used as-is without further processing.
        if self.rx_mode == 'DIRECT':
            
            # E_rx has shape (n_tx, n_rx, n_freq, 3)
            for arr in self.E_tan_per_tx:
                if arr.shape[0] != self.n_rx:
                    raise RuntimeError(
                        f"_determine_electrical_fields_at_receiver(): RX_MODE='DIRECT' expects {self.n_rx} recorded receivers per file, found {arr.shape[0]}.")
            
            self.E_rx = np.array(self.E_tan_per_tx)
            
            if show_results:
                print(f"_determine_electrical_fields_at_receiver(): DIRECT receivers, E_rx shape = {self.E_rx.shape}")
            return

        # Build equivalent surface currents and propagate them to every RX for every TX firing and frequency bin
        self.E_rx = np.zeros((self.n_tx, self.n_rx, n_freq, 3), dtype=complex)

        # Determine the effective near-field to far-field transformation mode.
        nf_eff = self._resolved_nf2ff_mode()
        mode = "farfield" if self.farfield else "exact"
        if nf_eff == 'EXACT':
            mode = "exact"
        elif nf_eff == 'FARFIELD_CENTERED':
            mode = "farfield"

        print(f"_determine_electrical_fields_at_receiver(): propagating fields to {self.n_rx} receiver(s) for {self.n_tx} TX firing(s) x {n_freq} frequency bin(s) ({self.n_tx * self.n_rx * n_freq} evaluations total, mode='{mode}')...")

        # Loop over all TX firings to propagate fields to all RX positions.
        t_start_propagate = time.time()
        progress_step = max(1, n_freq // 100)
        for tx_idx in range(self.n_tx):
            t_start_tx = time.time()

            # Determine the observation (n_rx, 3) and source points based on the near-field to far-field mode.
            r_obs = self.rx_pos.T if nf_eff == 'EXACT' else self.rx_huygens.T
            r_src = self.rx_positions
            if nf_eff == 'FARFIELD_CENTERED':
                c_ref = np.asarray(self.target_center).reshape(3)
                r_obs = r_obs - c_ref
                r_src = r_src - c_ref

            # Compute the geometry for the radiated field calculation based on the observation and source points.
            geometry = _radiated_field_geometry(r_obs, r_src, mode=mode)

            # Determine the electric field for the differential surface element for the Huygens surface.
            for fi, freq in enumerate(self.f_huygens):

                # Determine the electric field for the differential surface element for the current frequency bin (n_huygens_pts, 3).
                E_tan_f = self.E_tan_per_tx[tx_idx][:, fi, :]
                H_tan_f = self.H_tan_per_tx[tx_idx][:, fi, :]
                Js, Ms = _equivalent_currents(E_tan_f, H_tan_f, self.rx_normals)
                k = 2 * np.pi * freq / sp.constants.c
                E_field = _radiated_E_field(geometry, Js, Ms, 
                                            dS, k, self.eta_0, mode=mode)
                E_field = np.atleast_2d(E_field)
                self.E_rx[tx_idx, :, fi, :] = E_field

                if (fi + 1) % progress_step == 0 or fi == n_freq - 1:
                    print(f"_determine_electrical_fields_at_receiver(): TX {tx_idx + 1}/{self.n_tx} - frequency bin {fi + 1}/{n_freq} ({100 * (fi + 1) / n_freq:.0f}%, {freq:.3e} Hz) -> {self.n_rx} RX antenna(s) [{time.time() - t_start_propagate:.1f} s elapsed]")

            print(f"_determine_electrical_fields_at_receiver(): TX {tx_idx + 1}/{self.n_tx} done in {time.time() - t_start_tx:.1f} s")

        if show_results:
            print(f"determine_electrical_fields_at_receiver(): extracted {self.n_tx} TX firing(s) x {self.n_rx} RX antenna(s), {n_freq} frequency bins in range [{self.f_huygens[0]:.3e}, {self.f_huygens[-1]:.3e}] Hz; E_rx shape = {self.E_rx.shape}")

    def _postprocess_results(self, show_results: bool = False) -> None:
        """This function postprocesses the results obtained from the electrical field simulations at the receiver. It computes the MIMO transfer function H_tf (E_rx / E_tx, shape (n_tx, n_rx, n_freq)) by dividing the received field by the transmitted field in the frequency domain.

        Args:
            show_results (bool): 
                Show results (default: False).

        Returns:
            None
        
        Raises:
            RuntimeError: If self.band_mask is not set or if RX_PROJECTION='K_RX' is used with an incompatible mode.
        """
        # The FFT of the transmitted field is computed over the full time series, but only the frequency bins specified by self.band_mask are retained for further processing.
        E_tx_fft_full = np.fft.rfft(self.E_tx)

        if not hasattr(self, 'band_mask') or self.band_mask is None:
            raise RuntimeError("analyze_results(): self.band_mask is not set - run determine_electrical_fields_at_receiver() first.")

        self.E_tx_fft = E_tx_fft_full[self.band_mask]
        self.f_fft = self.f_huygens

        # The received field is projected onto the appropriate direction depending on self.rx_projection. 'K_RX' projects onto the propagation direction (only valid in LEGACY_AUTO mode), while 'POLARIZATION' projects onto the RX antenna polarization vector.
        if self.rx_projection == 'K_RX' and self._resolved_nf2ff_mode() != 'LEGACY_AUTO':
            raise RuntimeError(
                "_postprocess_results(): RX_PROJECTION='K_RX' projects the received field onto the propagation direction, which is ~zero for a correctly directed far field (E is transverse: E . k_rx = 0, measured ~1e-17 of the field). Use RX_PROJECTION='POLARIZATION' (co-polar component along RX_POLARIZATION, the default); 'K_RX' only makes sense together with NF2FF_MODE='LEGACY_AUTO' to re-analyze old results.")

        # Project the received field onto the appropriate direction based on self.rx_projection.
        if self.rx_projection == 'POLARIZATION':
            pol_vec = np.zeros(3)
            pol_vec['xyz'.index(str(self.rx_polarization).lower())] = 1.0
            self.E_rx_fft = np.sum(self.E_rx * pol_vec[None, None, None, :], axis=-1)
        else:
            self.E_rx_fft = np.sum(self.E_rx * self.k_rx.T[None, :, None, :], axis=-1)

        # Compute the transfer function H_tf as the ratio of the received field to the transmitted field for each TX-RX-frequency combination and frequency.
        self.H_tf = self.E_rx_fft / self.E_tx_fft[None, None, :]

        # Compute the amplitude and phase of the transfer function.
        self.A_tf = np.abs(self.H_tf)
        self.phi_tf = np.unwrap(np.angle(self.H_tf), axis=-1)

        if show_results:
            print(f"analyze_results(): H_tf shape = {self.H_tf.shape} (n_tx={self.H_tf.shape[0]}, n_rx={self.H_tf.shape[1]}, n_freq={self.H_tf.shape[2]}); frequency band = [{self.f_fft[0]:.3e}, {self.f_fft[-1]:.3e}] Hz")

    def _save_results(self, filename: str = 'data.pkl') -> None:
        """This function saves the relevant simulation results and dielectric properties to a file.

        Args:
            filename (str): 
                The name of the file to save the results to. Defaults to 'data.pkl'.

        Returns:
            None

        Raises:
            None
        """
        # Prepare the dielectric properties dictionary for saving.
        dielectric = {
            'sut_mode': self.sut_mode,
            'fo_mode': self.fo_mode,
            'debye_model_sut': self.debye_model_sut,
            'conductivity_sut': self.conductivity_sut,
            'debye_model_fo': self.debye_model_fo,
            'permittivity_relative_fo': self.permittivity_relative_fo,
            'conductivity_fo': self.conductivity_fo,
            'fo_layer_index': getattr(self, 'fo_layer_index', 0),
            'sut_inclusions': (None if self.sut_inclusions is None else {
                'n': len(self.sut_inclusions['CENTERS']),
                'diameter': self.sut_inclusions['DIAMETER'],
                'debye_model': self.debye_model_inclusion,
                'conductivity': self.conductivity_inclusion}),
            'sut_layers': (None if self.debye_model_sut_layers is None else [
                {'NAME': layer['NAME'], 'THICKNESS': layer['THICKNESS'],
                 'debye_model': self.debye_model_sut_layers[i],
                 'conductivity': self.conductivity_sut_layers[i]}
                for i, layer in enumerate(self.sut_layers)]),}

        # Save the results dictionary to the specified file.
        path_results = Path(self.path_folder) / filename
        with open(path_results, 'wb') as f:
            pickle.dump({'H_tf': self.H_tf, 'f_fft': self.f_fft,
                         'rx_side': 'reflection' if float(self.rx_pos_range[2]) > 0 else 'transmission',
                         'tx_pos': getattr(self, 'tx_pos', None),
                         'rx_pos': getattr(self, 'rx_pos', None),
                         'dielectric': dielectric,
                         'geometry': {
                             'l_grid': (self.l_x_grid, self.l_y_grid, self.l_z_grid),
                             'sut_offset': self.sut_offset,
                             'target_center': np.asarray(self.target_center).reshape(3),
                             'k_tx': getattr(self, 'k_tx', None),
                             'k_rx': getattr(self, 'k_rx', None),
                             'rx_huygens': (getattr(self, 'rx_pos', None) if self._resolved_nf2ff_mode() == 'EXACT'
                                            else getattr(self, 'rx_huygens', None)),
                             'delta_z': self.l_z_grid / 2,
                             'rx_projection': self.rx_projection,
                             'rx_polarization': self.rx_polarization,
                             'nf2ff_mode': self._resolved_nf2ff_mode(),
                             'rx_mode': self.rx_mode,
                             'plane_wave_angles': self.plane_wave_angles,
                             'tau_offset_m': (None if self.plane_wave_tau_offset_m is None
                                              else np.asarray(self.plane_wave_tau_offset_m))}}, f)
        print(f"_save_results(): results saved to {path_results}")

    def _save_mirrored_rx_results(self) -> None:
        """This function mirrors the RX array in the z-direction and evaluates the corresponding NF2FF results using the already-read Huygens-surface fields.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: If RX_MODE='DIRECT', RX z is 0, or a consistency assertion fails (the run is then reported as failed and its .out files are kept).
        """
        # Check for unsupported RX mode
        if self.rx_mode == 'DIRECT':
            raise RuntimeError("_save_mirrored_rx_results(): ALSO_MIRRORED_RX is not supported with RX_MODE='DIRECT'.")
        
        # Check the primary RX z-position
        z_primary = float(self.rx_pos_range[2])
        if z_primary == 0.0:
            raise RuntimeError("_save_mirrored_rx_results(): RX_POS_RANGE z = 0, nothing to mirror.")
        name = 'transmission' if z_primary > 0 else 'reflection'

        keys = ['rx_pos_range', 'rx_pos', 'rx_huygens', 'k_rx','received_wave_alpha', 'received_wave_beta', 'tx_pos', 'tx_huygens', 'k_tx', 'incident_wave_alpha', 'incident_wave_beta', 'target_center', 'domain_center', 'E_rx', 'E_rx_fft', 'H_tf', 'A_tf', 'phi_tf','E_tx_fft', 'f_fft']

        # Backup the current state of the relevant attributes before mirroring the RX array
        backup = {k: copy.deepcopy(getattr(self, k)) for k in keys if hasattr(self, k)}
        try:
            # Mirror the RX array by flipping the z-coordinate of the RX position range
            rpr = list(self.rx_pos_range)
            rpr[2] = -rpr[2]
            self.rx_pos_range = rpr
            self._determine_antenna_location_huygens()

            # Check that the transmitter-related attributes have not changed after mirroring the RX array
            for k in ('tx_pos', 'tx_huygens', 'k_tx', 'incident_wave_alpha', 'incident_wave_beta', 'target_center'):
                if not np.array_equal(np.asarray(backup[k]), np.asarray(getattr(self, k))):
                    raise RuntimeError(f"_save_mirrored_rx_results(): '{k}' changed with the mirrored RX - the recorded fields would not be valid for both arrays.")

            # Check that the mirrored RX position is consistent with a z-mirror of the primary RX position
            zc = float(self.l_z_grid) / 2.0
            rx0, rx1 = np.asarray(backup['rx_pos']), np.asarray(self.rx_pos)
            if not (np.allclose(rx1[:2], rx0[:2], atol=1e-12) and np.allclose(rx1[2] - zc, -(rx0[2] - zc), atol=1e-12)):
                raise RuntimeError("_save_mirrored_rx_results(): mirrored RX array is not the z-mirror of the primary array.")

            # Check that the mirrored k_rx is consistent with a z-mirror of the primary k_rx
            k0, k1 = np.asarray(backup['k_rx']), np.asarray(self.k_rx)
            if not (np.allclose(k1[:2], k0[:2], atol=1e-9) and np.allclose(k1[2], -k0[2], atol=1e-9)):
                raise RuntimeError("_save_mirrored_rx_results(): k_rx of the mirrored array is not the z-mirror of the primary k_rx.")

            # Print information about the mirrored RX array relative to the sample centre
            print(f"_save_mirrored_rx_results(): evaluating the same Huygens-surface fields for the mirrored RX array ({name}): "
                  f"RX z = {(rx1[2, 0] - zc) * 1e3:+.1f} mm relative to the sample centre (primary {(rx0[2, 0] - zc) * 1e3:+.1f} mm).")
            self._determine_electrical_fields_at_receiver(show_results=False, reuse_tangential_fields=True)
            self._postprocess_results()
            self._save_results(filename=f'data_{name}.pkl')
            self.mirrored_results_file = Path(self.path_folder) / f'data_{name}.pkl'
            
        finally:

            for k, v in backup.items():
                setattr(self, k, v)

    def _delete_output_files(self) -> None:
        """This method deletes the large .out files generated by gprMax simulations to save disk space.

        Args:
            None
        
        Returns:
            None

        Raises:
            None
        """
        # Get all .out and .h5 files in the output folder. Return early if none found.
        out_files = self._get_output_files_folder(self.path_folder)
        if not out_files:
            print(f"_delete_output_files(): no .out or .h5 files found in {self.path_folder} - nothing to delete.")
            return

        # Delete each output file and keep track of the total size and number of deleted files.
        total_bytes = 0
        n_deleted = 0
        for f in out_files:
            path = Path(f)
            try:
                total_bytes += path.stat().st_size
                path.unlink()
                n_deleted += 1
            except OSError as e:
                print(f"_delete_output_files(): failed to delete {f}: {e}")

        print(f"_delete_output_files(): deleted {n_deleted}/{len(out_files)} .out file(s) in {self.path_folder}, freeing {total_bytes / 1e9:.2f} GB. The .in files were kept.")

    def _resolved_nf2ff_mode(self) -> str:
        """This function resolves the NF2FF mode based on the current settings and the far-field condition.

        Args:
            None

        Returns:
            str: The resolved NF2FF mode ('EXACT', 'FARFIELD_CENTERED', or 'LEGACY_AUTO').
        
        Raises:
            None
        """
        if self.nf2ff_mode == 'AUTO':
            return 'FARFIELD_CENTERED' if self.farfield else 'EXACT'
        return self.nf2ff_mode

    def _write_run_protocol(self) -> None:
        """This function writes the run protocol files before the gprMax simulation starts, including protocol.txt, protocol_parms.json, and a row in run_protocol.csv.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        try:
            self._write_run_protocol_impl()
        except Exception as e:
            print(f"_write_run_protocol(): WARNING - protocol not written: {e!r}")

    def _write_run_protocol_impl(self) -> None:
        """This function implements the run protocol writing logic.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If writing the protocol files fails.
        """
        import csv
        deg = np.degrees

        # Definitions of helper functions for angle and spherical coordinate calculations
        def _ang(a, b):
            """Angle [deg] between the columns of a and b (3, n)."""
            a = a / np.linalg.norm(a, axis=0)
            b = b / np.linalg.norm(b, axis=0)
            return deg(np.arccos(np.clip(np.sum(a * b, axis=0), -1.0, 1.0)))

        def _sph(v):
            """theta (from +z), phi (from +x) [deg] and |v| of the columns of v (3, n)."""
            r = np.linalg.norm(v, axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                th = deg(np.arccos(np.clip(v[2] / r, -1, 1)))
            ph = deg(np.arctan2(v[1], v[0]))
            return th, ph, r

        # Compute the relative positions of the TX with respect to the domain center and the geometric direction of travel
        center = self.domain_center.reshape(3, 1)
        target = self.target_center.reshape(3, 1)
        tx, rx = np.asarray(self.tx_pos, float), np.asarray(self.rx_pos, float)
        tx_rel = tx - center
        tx_th, tx_ph, tx_r = _sph(tx_rel)
        d_tx = target - tx
        k_geo = d_tx / np.linalg.norm(d_tx, axis=0)

        if self.plane_wave_angles == 'CORRECT':
            a_eff, b_eff = self.incident_wave_alpha, self.incident_wave_beta
        else:
            a_eff, b_eff = np.radians(self.incident_wave_alpha), np.radians(self.incident_wave_beta)

        k_eff = np.array([np.sin(a_eff) * np.cos(b_eff), np.sin(a_eff) * np.sin(b_eff), np.cos(a_eff)])
        err_pw = _ang(k_eff, k_geo)
        k_th, k_ph, _ = _sph(k_eff)
        kg_th, kg_ph, _ = _sph(k_geo)
        tilt_geo = np.minimum(kg_th, 180 - kg_th)
        tilt_eff = np.minimum(k_th, 180 - k_th)
        tau_off = (np.asarray(self.plane_wave_tau_offset_m) if self.plane_wave_tau_offset_m is not None
                   else np.full(self.n_tx, np.nan))

        # Compute the relative positions of the RX with respect to the domain center and the geometric direction of travel
        rx_rel = rx - center
        rx_th, rx_ph, rx_r = _sph(rx_rel)
        r_true = (rx - target) / np.linalg.norm(rx - target, axis=0)
        direct = (self.rx_mode == 'DIRECT')

        # Determine the far-field direction used for the RX based on the NF2FF mode and whether the RX is in direct mode
        if direct:
            ff_note = "RX_MODE='DIRECT': real receivers, no far-field direction (total field recorded at the position)"
            r_used, err_ff = None, np.full(self.n_rx, np.nan)
        else:
            nf = self._resolved_nf2ff_mode()
            
            if nf == 'FARFIELD_CENTERED':
                mode_res = 'farfield (centered on the sample)'
                v = self.rx_huygens - target
            elif nf == 'EXACT' or (nf == 'LEGACY_AUTO' and not self.farfield):
                mode_res = 'exact (true distances, no direction assumption)'
                v = None
            else:
                mode_res = 'farfield (direction taken from the DOMAIN ORIGIN, corner of the grid - legacy)'
                v = np.asarray(self.rx_huygens, float)

            ff_note = f"NF2FF_MODE={self.nf2ff_mode} -> {mode_res}"
            if v is None:
                r_used, err_ff = None, np.zeros(self.n_rx)
            else:
                r_used = v / np.linalg.norm(v, axis=0)
                err_ff = _ang(r_used, r_true)
        rx_dir_th, rx_dir_ph, _ = _sph(r_true)

        # Apply grid snapping of the receivers as performed by gprMax (it rounds #rx to the nearest cell)
        snapped = np.floor(rx / self.dx + 0.5) * self.dx
        snap_err = np.linalg.norm(snapped - rx, axis=0)
        ysort = np.sort(snapped[1])
        pitch_snapped = np.diff(ysort) if self.n_rx > 1 else np.array([])

        # Check if the plane-wave and far-field directions are within the specified tolerance
        tol = 1e-6
        pw_ok = bool(np.all(err_pw < tol))
        ff_ok = bool(np.all(np.nan_to_num(err_ff) < tol))
        checks = [
            f"PLANE-WAVE DIRECTION written to the .in vs. geometric TX->sample direction ({self.plane_wave_angles}): max error {np.max(err_pw):.4f} deg -> {'OK' if pw_ok else 'MISMATCH'}",
            f"FAR-FIELD DIRECTION used for the RX vs. geometric sample->RX direction: "
            + (f"n/a ({ff_note})" if direct or r_used is None else
               f"max error {np.max(err_ff):.4f} deg -> {'OK' if ff_ok else 'MISMATCH'}  [{ff_note}]"),
            f"RX grid snapping: max position shift {1e3 * np.max(snap_err):.4f} mm"
            + (f"; snapped y pitch {1e3 * pitch_snapped.min():.4f}..{1e3 * pitch_snapped.max():.4f} mm (dx = {1e3 * self.dx:.4f} mm)" if len(pitch_snapped) and pitch_snapped.max() > 0 else ""),
            f"propagation sign: k_z of the applied plane wave = {np.round(k_eff[2], 4).tolist()} "
            f"(geometric {np.round(k_geo[2], 4).tolist()})",]

        # Write the per-run protocol to a list of strings (L) for later output
        folder = Path(self.path_folder)
        ctx = self.protocol_context or {}
        L = []
        L.append("RUN PROTOCOL  (written before gprMax starts)")
        L.append(f"time          : {datetime.now().isoformat(timespec='seconds')}")
        L.append(f"run folder    : {folder}")
        L.append(f"parms source  : {self._parms_source}")
        L.append(f"consts source : {self._consts_source}")
        for k, v in ctx.items():
            L.append(f"context       : {k} = {v}")
        L.append("settings      : "
                 f"RX_MODE={self.rx_mode}, NF2FF_MODE={self.nf2ff_mode} (used: {self._resolved_nf2ff_mode()}), PLANE_WAVE_ANGLES={self.plane_wave_angles}, "
                 f"RX_PROJECTION={self.rx_projection}, N_TX={self.n_tx}, N_RX={self.n_rx}, "
                 f"SUT_OFFSET={[round(1e3 * v, 3) for v in self.sut_offset]} mm, far-field condition met={getattr(self, 'farfield', None)}")
        L.append(f"band [GHz]    : {self.frequency_range[0] / 1e9:g} .. {self.frequency_range[-1] / 1e9:g}; dx = {self.dx * 1e3:.4f} mm")
        L.append(f"domain [mm]   : {self.l_x_grid * 1e3:g} x {self.l_y_grid * 1e3:g} x {self.l_z_grid * 1e3:g}; "
                 f"domain center = {np.round(1e3 * center.ravel(), 3).tolist()}; sample/reference center = {np.round(1e3 * target.ravel(), 3).tolist()}")
        L.append("conventions   : theta = polar angle from +z (0 = up, 180 = down), phi = azimuth in the xy-plane from +x toward +y [deg] "
                 "(spherical angles, NOT rotations about axes); (x,y,z) = unit direction vector; positions relative to the DOMAIN CENTER; "
                 "k = direction of propagation")
        L.append("")
        L.append("CHECKS")
        L += ["  " + c for c in checks]
        L.append("")
        L.append("TX  (idx | abs pos [mm] | rel. to domain center [mm] | dist | theta | phi || geometric k theta/phi | geometric k (kx,ky,kz) | "
                 "alpha=theta,beta=phi intended [rad] | alpha,beta written to .in [rad] | applied k theta/phi | applied k (kx,ky,kz) | "
                 "applied k - geometric [deg] | tilt geo/applied [deg] | tau_off [m])")
        for i in range(self.n_tx):
            L.append(f"  TX{i:<3d} {np.round(1e3 * tx[:, i], 3).tolist()} | {np.round(1e3 * tx_rel[:, i], 3).tolist()} | "
                     f"{1e3 * tx_r[i]:.3f} mm | {tx_th[i]:.3f} | {tx_ph[i]:.3f} || {kg_th[i]:.4f}/{kg_ph[i]:.4f} | "
                     f"({k_geo[0, i]:+.5f},{k_geo[1, i]:+.5f},{k_geo[2, i]:+.5f}) | "
                     f"{self.incident_wave_alpha[i]:.6f},{self.incident_wave_beta[i]:.6f} | "
                     f"{a_eff[i]:.6f},{b_eff[i]:.6f} | {k_th[i]:.4f}/{k_ph[i]:.4f} | "
                     f"({k_eff[0, i]:+.5f},{k_eff[1, i]:+.5f},{k_eff[2, i]:+.5f}) | {err_pw[i]:.4f} | "
                     f"{tilt_geo[i]:.4f}/{tilt_eff[i]:.4f} | {tau_off[i]:.6g}")
        L.append("")
        L.append("RX  (idx | abs pos [mm] | rel. to domain center [mm] | dist | theta | phi || geometric sample->RX theta/phi | "
                 "geometric sample->RX (x,y,z) | far-field direction used (x,y,z) | far-field direction error [deg] | grid-snapped pos [mm])")
        for j in range(self.n_rx):
            L.append(f"  RX{j:<3d} {np.round(1e3 * rx[:, j], 3).tolist()} | {np.round(1e3 * rx_rel[:, j], 3).tolist()} | "
                     f"{1e3 * rx_r[j]:.3f} mm | {rx_th[j]:.3f} | {rx_ph[j]:.3f} || {rx_dir_th[j]:.4f}/{rx_dir_ph[j]:.4f} | "
                     f"({r_true[0, j]:+.5f},{r_true[1, j]:+.5f},{r_true[2, j]:+.5f}) | "
                     + ("n/a" if (direct or r_used is None) else f"({r_used[0, j]:+.5f},{r_used[1, j]:+.5f},{r_used[2, j]:+.5f})") + " | "
                     f"{'n/a' if (direct or r_used is None) else f'{err_ff[j]:.4f}'} | {np.round(1e3 * snapped[:, j], 3).tolist()}")
        (folder / 'protocol.txt').write_text("\n".join(L) + "\n", encoding='utf-8')
        with open(folder / 'protocol_parms.json', 'w', encoding='utf-8') as fh:
            json.dump(self._parms_snapshot, fh, indent=1, default=str)

        # Prepare the study-wide protocol row for the current run, including metadata and error metrics
        csv_path = Path(self.path_input_output_files) / 'run_protocol.csv'
        row = dict(
            time=datetime.now().isoformat(timespec='seconds'), run_folder=folder.name,
            parms_source=self._parms_source, consts_source=self._consts_source,
            context=json.dumps(ctx, default=str), rx_mode=self.rx_mode, nf2ff_mode=self.nf2ff_mode,
            plane_wave_angles=self.plane_wave_angles, rx_projection=self.rx_projection,
            n_tx=self.n_tx, n_rx=self.n_rx,
            sut_offset_mm=json.dumps([round(1e3 * v, 3) for v in self.sut_offset]),
            tx0_rel_center_mm=json.dumps(np.round(1e3 * tx_rel[:, 0], 3).tolist()),
            rx0_rel_center_mm=json.dumps(np.round(1e3 * rx_rel[:, 0], 3).tolist()),
            tx_tilt_geometric_deg=json.dumps(np.round(tilt_geo, 4).tolist()),
            tx_tilt_applied_deg=json.dumps(np.round(tilt_eff, 4).tolist()),
            plane_wave_max_err_deg=round(float(np.max(err_pw)), 4), plane_wave_check='OK' if pw_ok else 'MISMATCH',
            farfield_max_err_deg=('n/a' if (direct or r_used is None) else round(float(np.max(err_ff)), 4)),
            farfield_check=('n/a' if (direct or r_used is None) else ('OK' if ff_ok else 'MISMATCH')),
            rx_snap_max_mm=round(1e3 * float(np.max(snap_err)), 4))

        # Ensure the parent directory exists and write the protocol row to the CSV file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.is_file()
        with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

        # Print a summary of the run to the console, including TX/RX positions, angles, and error metrics
        print("\n--- RUN PROTOCOL (angles w.r.t. the domain center; full table in protocol.txt) ---")
        for i in range(min(self.n_tx, 6)):
            print(f"  TX{i}: pos rel. center {np.round(1e3 * tx_rel[:, i], 2).tolist()} mm, theta/phi {tx_th[i]:.2f}/{tx_ph[i]:.2f} deg; "
                  f"geometric k theta {kg_th[i]:.3f} deg ({k_geo[0, i]:+.4f},{k_geo[1, i]:+.4f},{k_geo[2, i]:+.4f}), applied k theta {k_th[i]:.3f} deg "
                  f"({k_eff[0, i]:+.4f},{k_eff[1, i]:+.4f},{k_eff[2, i]:+.4f}) (error {err_pw[i]:.3f} deg)")
        for j in sorted({0, self.n_rx // 2, self.n_rx - 1}):
            print(f"  RX{j}: pos rel. center {np.round(1e3 * rx_rel[:, j], 2).tolist()} mm, theta/phi {rx_th[j]:.2f}/{rx_ph[j]:.2f} deg"
                  + ("" if direct or r_used is None else f"; far-field direction error {err_ff[j]:.3f} deg"))
        for c in checks:
            print("  " + c)
        print(f"  protocol: {folder / 'protocol.txt'}; study log: {csv_path}\n")

    def _write_setup_summary(self) -> None:
        """Write a summary of the TX/RX setup and parameters to a text file in the run folder. It includes positions, directions, angles, and the effective parameter dictionary.
        
        Args:
            None

        Returns:
            None

        Raises:
            Exception: If writing the setup summary fails.
        """
        try:
            self._write_setup_summary_impl()
        except Exception as e:
            print(f"_write_setup_summary(): WARNING - setup summary not written: {e!r}")

    def _write_setup_summary_impl(self) -> None:
        """This function implements writing the setup summary to a text file. 
        
        Args:
            None

        Returns:
            None

        Raises:
            Exception: If writing the setup summary fails.
        """
        # Convert angles from radians to degrees
        deg = np.degrees
        folder = Path(self.path_folder)
        c = np.asarray(self.target_center, float).reshape(3, 1)
        tx = np.asarray(self.tx_pos, float)
        rx = np.asarray(self.rx_pos, float)

        # Define helper functions for spherical coordinates and table formatting
        def sph(v):
            v = np.asarray(v, float)
            r = np.linalg.norm(v, axis=0)
            th = deg(np.arccos(np.clip(v[2] / r, -1, 1)))
            ph = deg(np.arctan2(v[1] + 0.0, v[0]))
            return th, ph

        def table(header, rows):
            w = [max(len(str(h)), *(len(str(r[k])) for r in rows)) for k, h in enumerate(header)]
            fmt = lambda r: "  " + "  ".join(str(x).rjust(w[k]) if k else str(x).ljust(w[k]) for k, x in enumerate(r))
            return [fmt(header), "  " + "  ".join("-" * x for x in w)] + [fmt(r) for r in rows]

        # Helper function for formatting floating-point numbers with fixed decimal places
        f3 = lambda a, d=5: [f"{x:.{d}f}" for x in a]

        # Read the alpha and beta values (angles) as written in the .in files for each transmitter
        written = []
        for i in range(self.n_tx):
            a_val = b_val = None
            legacy = False
            fin = folder / f'em_sim_{i}.in'
            if fin.is_file():
                with open(fin, 'r', errors='replace') as fh:
                    for n_line, line in enumerate(fh):
                        ma = re.match(r'\s*alpha\s*=\s*(math\.radians\()?\s*([-+0-9.eE]+)', line)
                        mb = re.match(r'\s*beta\s*=\s*(math\.radians\()?\s*([-+0-9.eE]+)', line)
                        if ma:
                            a_val, legacy = float(ma.group(2)), bool(ma.group(1))
                        if mb:
                            b_val = float(mb.group(2))
                        if (a_val is not None and b_val is not None) or n_line > 400:
                            break
            written.append((a_val, b_val, legacy))

        k_applied = np.full((3, self.n_tx), np.nan)
        for i, (a_val, b_val, legacy) in enumerate(written):
            if a_val is None or b_val is None:
                continue
            a_e, b_e = (np.radians(a_val), np.radians(b_val)) if legacy else (a_val, b_val)
            k_applied[:, i] = [np.sin(a_e) * np.cos(b_e), np.sin(a_e) * np.sin(b_e), np.cos(a_e)]

        # Compute the applied wave vectors for each transmitter based on the read angles
        k_tx = np.asarray(self.k_tx, float)
        k_geo = (c - tx) / np.linalg.norm(c - tx, axis=0)
        th_k, ph_k = sph(k_tx)
        err_pw = deg(np.arccos(np.clip(np.sum(k_applied * k_geo, axis=0), -1, 1)))

        # Compute the error between the applied wave vectors and the geometric wave vectors
        k_rx = np.asarray(self.k_rx, float)
        th_r, ph_r = sph(k_rx)
        rxh = np.asarray(self.rx_huygens, float)
        nf = self._resolved_nf2ff_mode()
        if self.rx_mode == 'DIRECT':
            v, ff_note = None, "RX_MODE='DIRECT': no far-field direction (field recorded at the receiver position)"
        elif nf == 'FARFIELD_CENTERED':
            v, ff_note = rxh - c, "far field centered on the sample: direction = (rx_huygens - sample center)"
        elif nf == 'EXACT':
            v, ff_note = None, "exact near-to-far-field evaluation: true distances, no direction assumption"
        else:
            v, ff_note = rxh, "LEGACY_AUTO: direction taken from the DOMAIN ORIGIN (legacy, wrong)"

        # Normalize the vector to get the far-field direction used for comparison with the geometric direction
        r_used = None if v is None else v / np.linalg.norm(v, axis=0)
        r_geo = (rx - c) / np.linalg.norm(rx - c, axis=0)
        if self.rx_projection == 'POLARIZATION':
            pol = np.zeros(3)
            pol['xyz'.index(str(self.rx_polarization).lower())] = 1.0
            proj_note = (f"RX_PROJECTION='POLARIZATION': received scalar = E_far . p with p = "
                         f"({pol[0]:g}, {pol[1]:g}, {pol[2]:g}) (RX_POLARIZATION = '{self.rx_polarization}')")
        else:
            proj_note = "RX_PROJECTION='K_RX': received scalar = E_far . k_rx (per RX, see k_rx table)"

        # Initialize the list for storing the setup summary lines
        L = []
        L.append("TX/RX SETUP AND PARAMETERS OF THIS RUN  (written automatically before gprMax starts)")
        L.append(f"time          : {datetime.now().isoformat(timespec='seconds')}")
        L.append(f"run folder    : {folder}")
        L.append(f"parms source  : {self._parms_source}")
        L.append(f"consts source : {self._consts_source}")
        for k, val in (self.protocol_context or {}).items():
            L.append(f"context       : {k} = {val}")
        L.append(f"settings      : PLANE_WAVE_ANGLES={self.plane_wave_angles}, NF2FF_MODE={self.nf2ff_mode} (used: {nf}), "
                 f"RX_MODE={self.rx_mode}, RX_PROJECTION={self.rx_projection}, N_TX={self.n_tx}, N_RX={self.n_rx}")
        L.append(f"domain [mm]   : {self.l_x_grid * 1e3:g} x {self.l_y_grid * 1e3:g} x {self.l_z_grid * 1e3:g}; "
                 f"sample/reference center [mm] = ({', '.join(f3(1e3 * c.ravel(), 5))}); dx = {self.dx * 1e3:.5f} mm; "
                 f"band [GHz] = {self.frequency_range[0] / 1e9:g} .. {self.frequency_range[-1] / 1e9:g}")
        L.append("conventions   : domain coordinates have their origin in the grid corner; 'rel.' = relative to the sample center; "
                 "theta = polar angle from +z (0 = up, 180 = down), phi = azimuth in the xy-plane from +x toward +y [deg]; "
                 "k_tx = travel direction TX -> sample, k_rx = direction RX -> sample, r_hat = far-field direction sample -> RX (= -k_rx)")
        L.append("")
        L.append("=== TX POSITIONS [mm] ===")
        L += table(["TX", "x dom", "y dom", "z dom", "x rel", "y rel", "z rel"],
                   [[f"TX{i}"] + f3(1e3 * tx[:, i]) + f3(1e3 * (tx[:, i] - c.ravel())) for i in range(self.n_tx)])
        L.append("")
        L.append("=== TX PLANE-WAVE DIRECTION k_tx (TX -> sample) ===")
        L += table(["TX", "theta [deg]", "phi [deg]", "kx", "ky", "kz"],
                   [[f"TX{i}", f"{th_k[i]:.6f}", f"{ph_k[i]:.6f}"] + f3(k_tx[:, i], 6) for i in range(self.n_tx)])
        L.append("")
        L.append("=== WRITTEN INTO THE .in FILES (embedded python block) AND THE k THAT RESULTS ===")
        L += table(["TX", "alpha in .in", "beta in .in", "radians() applied", "kx", "ky", "kz", "error vs k_tx [deg]"],
                   [[f"TX{i}", "n/a" if written[i][0] is None else f"{written[i][0]:.10f}",
                     "n/a" if written[i][1] is None else f"{written[i][1]:.10f}",
                     "yes (LEGACY)" if written[i][2] else "no"] + f3(k_applied[:, i], 6) + [f"{err_pw[i]:.4f}"]
                    for i in range(self.n_tx)])
        L.append("")
        L.append("=== RX POSITIONS [mm] ===")
        L += table(["RX", "x dom", "y dom", "z dom", "x rel", "y rel", "z rel"],
                   [[f"RX{j}"] + f3(1e3 * rx[:, j]) + f3(1e3 * (rx[:, j] - c.ravel())) for j in range(self.n_rx)])
        L.append("")
        L.append("=== RX DIRECTION k_rx (RX -> sample) ===")
        L += table(["RX", "theta [deg]", "phi [deg]", "kx", "ky", "kz"],
                   [[f"RX{j}", f"{th_r[j]:.6f}", f"{ph_r[j]:.6f}"] + f3(k_rx[:, j], 6) for j in range(self.n_rx)])
        L.append("")
        L.append(f"=== FAR-FIELD DIRECTION USED r_hat (sample -> RX) - {ff_note} ===")
        if r_used is None:
            L.append("  n/a")
        else:
            th_u, ph_u = sph(r_used)
            err_ff = deg(np.arccos(np.clip(np.sum(r_used * r_geo, axis=0), -1, 1)))
            L += table(["RX", "theta [deg]", "phi [deg]", "x", "y", "z", "error vs geometric [deg]"],
                       [[f"RX{j}", f"{th_u[j]:.6f}", f"{ph_u[j]:.6f}"] + f3(r_used[:, j], 6) + [f"{err_ff[j]:.4f}"]
                        for j in range(self.n_rx)])
        L.append("")
        L.append("=== RX OBSERVATION POINTS rx_huygens [mm, domain coordinates] ===")
        L += table(["RX", "x", "y", "z"], [[f"RX{j}"] + f3(1e3 * rxh[:, j]) for j in range(self.n_rx)])
        L.append("")
        L.append(f"=== RX PROJECTION ===")
        L.append("  " + proj_note)
        L.append("")

        # Collect the paths of parameter files used in this simulation run, avoiding duplicates
        paths, seen = [], set()
        cand = [self._parms_source, self._consts_source] + [v for k, v in (self.protocol_context or {}).items() if str(k).endswith('_file')]
        for pth in cand:
            if isinstance(pth, (str, Path)) and str(pth) != '<dict>' and str(pth) not in seen:
                seen.add(str(pth))
                paths.append(Path(pth))
        for pth in paths:
            L.append(f"=== PARAMETER FILE USED: {pth.as_posix()} ===")
            try:
                L.append(pth.read_text(encoding='utf-8').rstrip("\n"))
            except Exception as e:
                L.append(f"  (could not be read: {e!r})")
            L.append("")
        L.append("=== EFFECTIVE PARAMETERS OF THIS RUN (complete dict passed to EMSimulation) ===")
        L.append(json.dumps(self._parms_snapshot, indent=1, default=str))
        (folder / 'tx_rx_setup_and_parameters.txt').write_text("\n".join(L) + "\n", encoding='utf-8')
        print(f"_write_setup_summary(): {folder / 'tx_rx_setup_and_parameters.txt'}")

    def simulate(self) -> None:
        """This function orchestrates the entire simulation workflow, including setup, execution, and post-processing of results.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        # Apply preprocessing before the main simulation loop, such as determining the number of elements, time parameters, and incident wave parameters
        self._determine_number_of_elements()
        self._determine_time_parameters(show_results=True)
        self._check_farfield_condition()
        self._determine_complexity_and_resources(show_results=True)
        self._determine_material_properties()
        self._determine_antenna_location_huygens()
        self._generate_waveform(show_results=True)
        
        # Execute the main simulation loop, where the electromagnetic fields are updated at each time step based on the FDTD method and the interactions with the sample under test
        if self.simulate_before_analysis:
            self._generate_simulation_input_file()
            self._write_run_protocol()
            self._write_setup_summary()
            self._simulate_model()
        else:
            self._compute_receiver_geometry()
            self._resolve_existing_results_folder()
        self._determine_electrical_fields_at_receiver(show_results=True)
        
        # Post-processing steps after the main simulation loop (analyzing the results and visualizing the field distribution of the electromagnetic fields)
        self._postprocess_results()
        self._save_results()

        # Save the mirrored receiver results to a separate file, depending on the receiver position (reflection or transmission)
        if self.also_mirrored_rx:
            self._save_results(filename='data_reflection.pkl' if float(self.rx_pos_range[2]) > 0 else 'data_transmission.pkl')
            self._save_mirrored_rx_results()

        # Delete the .out files generated by the fresh simulation run if requested
        if self.simulate_before_analysis and self.delete_output_files_after_save:
            self._delete_output_files()

    @classmethod
    def preflight_resource_check(cls, 
                                 parms: Union[str, dict], 
                                 consts_file: str, 
                                 target_dir: str,
                                 model_gb: float = 3.5, 
                                 disk_margin_gb: float = 10.0) -> dict:
        """This function helps prevent wasting computational resources on simulations that are likely to fail due to insufficient disk or GPU memory.

        Args:
            parms (str | dict): 
                Parameter file path or dict.
            consts_file (str): 
                Constants file.
            target_dir (str): 
                Folder where the runs are written (its drive is checked; created if missing).
            model_gb (float): 
                Assumed GPU memory of the gprMax model besides the receivers [GB].
            disk_margin_gb (float): 
                Extra free disk required [GB].

        Returns:
            dict: n_rx_points, n_time, n_tx, receiver_gb, disk_need_gb, disk_free_gb, vram_need_gb, vram_free_gb (None if unknown).

        Raises:
            RuntimeError: If free disk or (for USE_GPU) free GPU memory is below the estimate.
        """
        # Dependencies
        import shutil, io, contextlib

        # Load and prepare parameters
        parms_d = copy.deepcopy(cls._load_config(parms) if isinstance(parms, str) else parms)
        if isinstance(parms_d.get('FREQUENCY_RANGE'), list):
            parms_d['FREQUENCY_RANGE'] = tuple(parms_d['FREQUENCY_RANGE'])
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        parms_d['PATH_INPUT_OUTPUT_FILES'] = str(target_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            em = cls(parms=parms_d, consts=consts_file)
            em._determine_number_of_elements(); em._determine_time_parameters(show_results=False); em._check_farfield_condition()
            em._determine_material_properties(); em._determine_antenna_location_huygens(); em._generate_waveform(show_results=False)
            em._compute_receiver_geometry()

        # Compute resource requirements based on loaded parameters
        n_rx = len(em.rx_positions)
        rec_gb = n_rx * 6 * em.n_time * 4 / 1e9
        disk_need = em.n_tx * rec_gb + disk_margin_gb
        disk_free = shutil.disk_usage(str(target_dir)).free / 1e9
        vram_need = rec_gb + model_gb
        vram_free = vram_total = None

        # Check if GPU is used and gather VRAM information if available
        if em.use_gpu:
            try:
                q = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
                vram_total, vram_free = (float(v) * 1.048576e6 / 1e9 for v in q.split(','))  # MiB -> GB (1e9 B)
            except Exception as exc:
                print(f"preflight_resource_check(): Warning - could not read GPU memory with nvidia-smi ({exc}); GPU memory NOT checked.")
        print(f"preflight_resource_check(): {n_rx} Huygens points, n_time = {em.n_time}, N_TX = {em.n_tx}: receiver buffers {rec_gb:.1f} GB; GPU need ~{vram_need:.1f} GB" + (f", free {vram_free:.1f} of {vram_total:.1f} GB" if vram_free is not None else "") + f"; disk need ~{disk_need:.0f} GB, free {disk_free:.0f} GB on the drive of '{target_dir}'.")
        problems = []
        if disk_free < disk_need:
            problems.append(f"free disk {disk_free:.0f} GB < ~{disk_need:.0f} GB needed ({em.n_tx} x {rec_gb:.1f} GB .out files + {disk_margin_gb:g} GB margin)")
        if vram_free is not None and vram_free < vram_need:
            problems.append(f"free GPU memory {vram_free:.1f} GB < ~{vram_need:.1f} GB needed (receivers {rec_gb:.1f} GB + model ~{model_gb:g} GB); "
                            f"close other GPU programs, lower RX_HUYGENS_WALL_SAMPLES_PER_WAVELENGTH, or shorten the time window")
            
        # Raise an error if any resource requirements are not met
        if problems:
            raise RuntimeError("EMSimulation.preflight_resource_check(): " + "; ".join(problems) + ".")
        return dict(n_rx_points=n_rx, 
                    n_time=em.n_time, 
                    n_tx=em.n_tx, 
                    receiver_gb=rec_gb, 
                    disk_need_gb=disk_need,
                    disk_free_gb=disk_free, 
                    vram_need_gb=vram_need, 
                    vram_free_gb=vram_free)
