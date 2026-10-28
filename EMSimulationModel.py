import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import least_squares

from EMSimulationHelpers import _freeform_columns
from EMSimulationParameters import _VACUUM_PERMITTIVITY_F_PER_M


class ModelMixin:
    """This mixing class provides methods for setting up and managing the EMSimulation model (materials, grid, waveform etc.), including loading configuration files, initializing simulation parameters, and handling grid and waveform settings.
    """

    @staticmethod
    def _load_config(config: Union[dict, str]) -> dict:
        """This function resolves a configuration argument into a dictionary. If a dictionary is provided, it is returned unchanged. If a string is provided, it is interpreted as a path to a JSON file, which is then loaded and returned as a dictionary.

        Args:
            config (dict | str): 
                Dict (returned unchanged) or path to a JSON file; a relative path is resolved against this script's directory.

        Returns:
            dict: The loaded configuration.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is not valid JSON.
        """
        # If the config is already a dictionary, return it as-is. Otherwise, treat it as a path to a JSON file.
        if isinstance(config, dict):
            return config
        given = config

        # Resolve the config path relative to this script if it's not absolute.
        path = Path(config)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(
                f"EMSimulation._load_config(): no such file: '{given}' (resolved to '{path}'). Relative paths are resolved against this script's own directory ({Path(__file__).resolve().parent}), not the current working directory - check the folder/filename (e.g. 'parameters/parms.json' or 'constants/consts.json') and that the file actually exists there.")
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"EMSimulation._load_config(): '{path}' is not valid JSON ({e}). Fix the syntax error in that file (e.g. a trailing comma or unquoted key) and try again.") from e

    def __init__(self, parms: Union[dict, str], consts: Union[dict, str]) -> None:
        """This constructor initializes the model.

        Args:
            parms (dict | str): 
                Simulation parameters, or path to a JSON file (relative to this script).
            consts (dict | str): 
                Material-property constants, or path to a JSON file (same rules).

        Returns:
            None

        Raises:
            FileNotFoundError: If the path to parms or consts does not exist.
            ValueError: If the JSON file for parms or consts is not valid.
        """
        # Store the sources of the parameters and constants for provenance.
        self._parms_source = str(parms) if isinstance(parms, (str, Path)) else '<dict>'
        self._consts_source = str(consts) if isinstance(consts, (str, Path)) else '<dict>'
        self.protocol_context = {}

        # Load the parameter and constant configurations.
        parms = self._load_config(parms)
        consts = self._load_config(consts)
        self._parms_snapshot = copy.deepcopy(parms)

        # Convert any list-based frequency range to a tuple for consistency.
        if isinstance(parms.get('FREQUENCY_RANGE'), list):
            parms['FREQUENCY_RANGE'] = tuple(parms['FREQUENCY_RANGE'])

        self.simulate_before_analysis = parms['SIMULATE_BEFORE_ANALYSIS']

        self.l_x_grid = parms['L_X_GRID']
        self.l_y_grid = parms['L_Y_GRID']
        self.l_z_grid = parms['L_Z_GRID']
        self.cells_per_wavelength = parms['CELLS_PER_WAVELENGTH']
        
        self.frequency_range = parms['FREQUENCY_RANGE']
        self.frequency_bands = parms.get('FREQUENCY_BANDS')
        self.sweeping_time = parms['SWEEPING_TIME']

        self.aperture = parms['APERTURE']
        self.tx_pos_range = parms['TX_POS_RANGE']

        self.n_tx = parms['N_TX']
        self.n_pos = self.n_tx
        self.tx_amplitude = parms['TX_AMPLITUDE']
        self.tx_polarization = parms['TX_POLARIZATION']
        self.rx_pos_range = parms['RX_POS_RANGE']
        self.n_rx = parms['N_RX']
        self.n_pml = parms['N_PML']
        self.n_pml_margin = parms['N_PML_MARGIN']
        self.rx_polarization = parms['RX_POLARIZATION']

        sut_offset = parms.get('SUT_OFFSET', [0.0, 0.0])
        self.sut_offset = (float(sut_offset[0]), float(sut_offset[1]))
        self.sut_cx = self.l_x_grid / 2 + self.sut_offset[0]
        self.sut_cy = self.l_y_grid / 2 + self.sut_offset[1]

        self.rx_projection = parms.get('RX_PROJECTION', 'POLARIZATION')
        if self.rx_projection not in ('K_RX', 'POLARIZATION'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['RX_PROJECTION'] must be 'K_RX' or 'POLARIZATION', got {self.rx_projection!r}.")
        if self.rx_projection == 'POLARIZATION' and str(self.rx_polarization).lower() not in ('x', 'y', 'z'):
            raise ValueError(
                f"EMSimulation.__init__(): RX_PROJECTION='POLARIZATION' needs RX_POLARIZATION in ('x', 'y', 'z'), got {self.rx_polarization!r}.")

        self.huygens_x_window = parms.get('HUYGENS_X_WINDOW', None)
        self.nf2ff_mode = parms.get('NF2FF_MODE', 'AUTO')
        if self.nf2ff_mode not in ('AUTO', 'EXACT', 'FARFIELD_CENTERED', 'LEGACY_AUTO'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['NF2FF_MODE'] must be 'AUTO', 'EXACT', 'FARFIELD_CENTERED' or 'LEGACY_AUTO', got {self.nf2ff_mode!r}.")

        self.plane_wave_angles = parms.get('PLANE_WAVE_ANGLES', 'CORRECT')
        if self.plane_wave_angles not in ('LEGACY', 'CORRECT'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['PLANE_WAVE_ANGLES'] must be 'LEGACY' or 'CORRECT', got {self.plane_wave_angles!r}.")
        self.plane_wave_tau_offset_m = None
        self.rx_mode = parms.get('RX_MODE', 'HUYGENS')
        if self.rx_mode not in ('HUYGENS', 'DIRECT'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['RX_MODE'] must be 'HUYGENS' or 'DIRECT', got {self.rx_mode!r}.")

        self.rx_huygens_samples_per_wavelength = parms.get('RX_HUYGENS_SAMPLES_PER_WAVELENGTH', 10)
        self.huygens_closed_surface = parms.get('HUYGENS_CLOSED_SURFACE', False)
        self.huygens_top_cap_offset_cells = int(parms.get('HUYGENS_TOP_CAP_OFFSET_CELLS', 0))

        self.also_mirrored_rx = bool(parms.get('ALSO_MIRRORED_RX', False))
        self.mirrored_results_file = None
        if self.huygens_top_cap_offset_cells < 0:
            raise ValueError("EMSimulation.__init__(): parms['HUYGENS_TOP_CAP_OFFSET_CELLS'] must be >= 0, got {self.huygens_top_cap_offset_cells!r}.")

        # Samples per wavelength for the four side walls and only used if huygens_closed_surface is True.
        self.rx_huygens_wall_samples_per_wavelength = parms.get(
            'RX_HUYGENS_WALL_SAMPLES_PER_WAVELENGTH', self.rx_huygens_samples_per_wavelength)

        self.belt = parms['BELT']
        self.sut = parms['SUT']
        self.foreign_object = parms['FOREIGN_OBJECT']

        self.fo_may_overlap_inclusions = bool(self.foreign_object.get('ALLOW_INCLUSION_OVERLAP', False))

        self.sut_mode = self.sut.get('MODE', 'INGREDIENTS')
        if self.sut_mode not in ('INGREDIENTS', 'SURROUNDING'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['SUT']['MODE'] must be 'INGREDIENTS' or 'SURROUNDING', got {self.sut_mode!r}.")

        self.fo_mode = self.foreign_object.get('MODE', 'MATERIAL')
        if self.fo_mode not in ('MATERIAL', 'SUT'):
            raise ValueError(
                f"EMSimulation.__init__(): parms['FOREIGN_OBJECT']['MODE'] must be 'MATERIAL' or 'SUT', got {self.fo_mode!r}.")
        if self.fo_mode == 'SUT' and self.sut_mode == 'SURROUNDING':
            raise ValueError(
                "EMSimulation.__init__(): parms['FOREIGN_OBJECT']['MODE'] = 'SUT' requires parms['SUT']['MODE'] = 'INGREDIENTS' - there is no SUT material to copy properties from when the SUT itself is 'SURROUNDING'.")

        self.sut_is_layered = 'LAYERS' in self.sut
        self.sut_layers = self._normalize_sut_layers(self.sut)
        self.sut_inclusions = self._normalize_sut_inclusions(self.sut)
        self.sut_freeform = self._normalize_sut_freeform(self.sut)

        self.path_input_output_files = parms['PATH_INPUT_OUTPUT_FILES']
        self.existing_results_folder = parms.get('EXISTING_RESULTS_FOLDER', None)

        self.delete_output_files_after_save = parms.get('DELETE_OUTPUT_FILES_AFTER_SAVE', False)

        self.amp_noise_floor_db = parms.get('AMP_NOISE_FLOOR_DB', None)
        self.phase_noise_floor_deg = parms.get('PHASE_NOISE_FLOOR_DEG', None)

        self.material_properties = consts['MATERIAL_PROPERTIES']

        from EMSimulationSettings import GPRMAX_PYTHON
        self.gprmax_venv_python = (Path(GPRMAX_PYTHON) if GPRMAX_PYTHON else Path(__file__).resolve().parent.parent
                                    / "EM Simulation" / ".venv" / "Scripts" / "python.exe")

        self.use_gpu = parms.get('USE_GPU', False)
        self.gpu_device = parms.get('GPU_DEVICE', None)

        self.msvc_bin_path = parms.get('MSVC_BIN_PATH', None)

        air_props = self.material_properties['AIR']
        eps_r_air = np.mean(air_props['PERMITTIVITY_RELATIVE'])
        mu_r_air = air_props['PERMEABILITY_RELATIVE']
        self.eta_0 = np.sqrt((sp.constants.mu_0 * mu_r_air) / (sp.constants.epsilon_0 * eps_r_air))

    @staticmethod
    def _normalize_sut_layers(sut: dict) -> list:
        """This function normalizes the SUT layers into a consistent list format, ensuring each layer has the required keys and valid values.

        Args:
            sut (dict): 
                parms['SUT'].

        Returns:
            list: [{'NAME': str, 'THICKNESS': float [m], 'INGREDIENTS_FRACTIONS': dict or None}, ...]

        Raises:
            ValueError: LAYERS empty/malformed, non-positive thickness, LAYERS together with INGREDIENTS_FRACTIONS, or thicknesses not summing to SIZE's thickness.
        """

        # Extract the total thickness from the SUT's SIZE field.
        size = sut['SIZE']
        total = size['THICKNESS'] if isinstance(size, dict) else size[2]
        if 'LAYERS' not in sut:
            return [{'NAME': 'sut', 'THICKNESS': float(total),
                     'INGREDIENTS_FRACTIONS': sut.get('INGREDIENTS_FRACTIONS')}]

        # Validate and normalize each layer in the SUT's LAYERS field.
        layers = sut['LAYERS']
        if not isinstance(layers, (list, tuple)) or len(layers) == 0:
            raise ValueError(
                "EMSimulation._normalize_sut_layers(): parms['SUT']['LAYERS'] must be a non-empty list of {'THICKNESS', 'INGREDIENTS_FRACTIONS'} dicts, top layer first.")
        
        # Ensure that INGREDIENTS_FRACTIONS is not specified at the top level when LAYERS are present.
        if 'INGREDIENTS_FRACTIONS' in sut:
            raise ValueError(
                "EMSimulation._normalize_sut_layers(): parms['SUT'] has both 'LAYERS' and 'INGREDIENTS_FRACTIONS' - ambiguous; give the fractions per layer.")

        # Process each layer, validating required keys and positive thickness.
        out = []
        for i, layer in enumerate(layers):
            for key in ('THICKNESS', 'INGREDIENTS_FRACTIONS'):
                if key not in layer:
                    raise ValueError(
                        f"EMSimulation._normalize_sut_layers(): SUT layer {i} has no '{key}'.")
            if not layer['THICKNESS'] > 0:
                raise ValueError(
                    f"EMSimulation._normalize_sut_layers(): SUT layer {i} has non-positive THICKNESS {layer['THICKNESS']!r}.")
            out.append({'NAME': layer.get('NAME', f'layer_{i}'),
                        'THICKNESS': float(layer['THICKNESS']),
                        'INGREDIENTS_FRACTIONS': layer['INGREDIENTS_FRACTIONS']})
        layer_sum = sum(layer['THICKNESS'] for layer in out)
        if abs(layer_sum - total) > 1e-9:
            raise ValueError(
                f"EMSimulation._normalize_sut_layers(): the layer thicknesses sum to {layer_sum * 1e3:.4f} mm but parms['SUT']['SIZE'] says {total * 1e3:.4f} mm - make them equal.")
        return out

    @staticmethod
    def _normalize_sut_inclusions(sut: dict):
        """Validate parms['SUT']['INCLUSIONS'] and return it (or None).
        This function checks the validity of the 'INCLUSIONS' entry in the SUT dictionary and ensures that it is consistent with the SUT's size and structure.

        Args:
            sut (dict): The SUT dictionary containing 'INCLUSIONS' and other relevant keys.

        Returns:
            dict or None: The validated 'INCLUSIONS' dictionary if present, otherwise None.

        Raises:
            ValueError: Missing keys, unsupported shape, sphere outside the SUT thickness, or combination with LAYERS.
        """
        # Get the 'INCLUSIONS' entry from the SUT dictionary and return None if it is not present. Check for conflicts with LAYERS and validate required keys.
        inc = sut.get('INCLUSIONS')
        if inc is None:
            return None
        if 'LAYERS' in sut:
            raise ValueError("EMSimulation._normalize_sut_inclusions(): INCLUSIONS are only supported in a plain (non-LAYERS) SUT.")
        for key in ('SHAPE', 'DIAMETER', 'CENTERS', 'INGREDIENTS_FRACTIONS'):
            if key not in inc:
                raise ValueError(
                    f"EMSimulation._normalize_sut_inclusions(): parms['SUT']['INCLUSIONS'] has no '{key}'.")
        if str(inc['SHAPE']).upper() != 'SPHERE':
            raise ValueError(
                f"EMSimulation._normalize_sut_inclusions(): only SHAPE 'SPHERE' is supported, got {inc['SHAPE']!r}.")

        # Validate the DIAMETER and ensure it is positive. Calculate the radius and check that each sphere is within the SUT thickness.
        if not inc['DIAMETER'] > 0:
            raise ValueError("EMSimulation._normalize_sut_inclusions(): DIAMETER must be > 0.")
        
        # Determine the total thickness of the SUT from the SIZE entry.
        size = sut['SIZE']
        total = size['THICKNESS'] if isinstance(size, dict) else size[2]
        radius = inc['DIAMETER'] / 2
        for c in inc['CENTERS']:
            if len(c) != 3:
                raise ValueError("EMSimulation._normalize_sut_inclusions(): every CENTERS entry needs [dx, dy, z].")
            if c[2] - radius < -1e-9 or c[2] + radius > total + 1e-9:
                raise ValueError(
                    f"EMSimulation._normalize_sut_inclusions(): a sphere at z = {c[2] * 1e3:.3f} mm (radius {radius * 1e3:.3f} mm) sticks out of the {total * 1e3:.3f} mm thick SUT.")
        
        return inc

    @staticmethod
    def _normalize_sut_freeform(sut: dict):
        """This function validates the FREEFORM shape of the SUT and returns its FREEFORM block with default values.

        Args:
            sut (dict): The SUT dictionary containing the FREEFORM shape and its parameters.

        Returns:
            dict: The normalized FREEFORM block with default values.

        Raises:
            ValueError: LAYERS/INCLUSIONS combined with FREEFORM, bad SIZE, edge radii above the thickness, or an incomplete region.
        """
        # Return None immediately if the SUT is not of shape 'FREEFORM'.
        if sut.get('SHAPE') != 'FREEFORM':
            return None

        # Ensure that LAYERS and INCLUSIONS are not present for FREEFORM shapes.
        if 'LAYERS' in sut or 'INCLUSIONS' in sut:
            raise ValueError("EMSimulation._normalize_sut_freeform(): SHAPE 'FREEFORM' supports neither LAYERS nor INCLUSIONS (use FREEFORM['REGIONS']).")

        # Validate the SIZE of the FREEFORM shape.
        size = sut.get('SIZE')
        if not isinstance(size, (list, tuple)) or len(size) != 3 or min(size) <= 0:
            raise ValueError(f"EMSimulation._normalize_sut_freeform(): SIZE must be [Lx, Ly, thickness] > 0, got {size!r}.")
        ff = dict(SCALLOPS=0, SCALLOP_DEPTH=0.0, EDGE_RADIUS_TOP=0.0, 
                  EDGE_RADIUS_BOTTOM=0.0, ROUGHNESS_AMPLITUDE=0.0,
                  ROUGHNESS_CORRELATION=0.003, ROUGHNESS_SEED=1, TILE=0.0005, REGIONS=[])

        # Update the default FREEFORM block with the values from the SUT.
        ff.update(sut.get('FREEFORM', {}))

        # Validate that the sum of the top and bottom edge radii does not exceed the thickness.
        if ff['EDGE_RADIUS_TOP'] + ff['EDGE_RADIUS_BOTTOM'] > size[2] + 1e-12:
            raise ValueError("EMSimulation._normalize_sut_freeform(): EDGE_RADIUS_TOP + EDGE_RADIUS_BOTTOM exceed the thickness.")
        
        # Validate each region in the FREEFORM block.
        for i, reg in enumerate(ff['REGIONS']):
            for key in ('SIZE', 'DEPTH', 'INGREDIENTS_FRACTIONS'):
                if key not in reg:
                    raise ValueError(f"EMSimulation._normalize_sut_freeform(): REGIONS[{i}] has no '{key}'.")
            if not 0 < reg['DEPTH'] or reg['DEPTH'] + reg.get('TOP_OFFSET', 0.0) + reg.get('TOP_SAG', 0.0) > size[2] + 1e-12:
                raise ValueError(f"EMSimulation._normalize_sut_freeform(): REGIONS[{i}] needs DEPTH > 0 and DEPTH + TOP_OFFSET + TOP_SAG <= thickness.")

        # Validate the SKIN section if it exists.
        sk = ff.get('SKIN')
        if sk and (not 0 < sk.get('THICKNESS', 0) < size[2] or 'INGREDIENTS_FRACTIONS' not in sk):
            raise ValueError("EMSimulation._normalize_sut_freeform(): SKIN needs 0 < THICKNESS < thickness and INGREDIENTS_FRACTIONS.")

        # Validate the RIM section if it exists.
        rb = ff.get('RIM')
        if rb and (not 0 < rb.get('THICKNESS', 0) or 'INGREDIENTS_FRACTIONS' not in rb):
            raise ValueError("EMSimulation._normalize_sut_freeform(): RIM needs THICKNESS > 0 and INGREDIENTS_FRACTIONS.")
        
        sut['FREEFORM'] = ff
        
        return ff

    def _freeform_geometry_lines(self) -> str:
        """This function generates the gprMax #box lines for the 'FREEFORM' SUT, including the main 'sut' columns and each 'region_<i>' column. Runs of equal columns are merged along the x-axis.

        Args:
            None

        Returns:
            str: The gprMax #box lines for the 'FREEFORM' SUT.
        
        Raises:
            ValueError: If the 'FREEFORM' SUT is not properly defined.
        """

        # Determine the tile size and generate the freeform columns.
        tile = max(1, int(round(self.sut_freeform['TILE'] / self.dx))) * self.dx
        col = _freeform_columns(self.sut, tile, self.dz)
        x0 = self.sut_cx + col['x'] - tile / 2
        y0 = self.sut_cy + col['y'] - tile / 2
        n_belt = int(np.floor(self.l_z_grid / 2 / self.dz + 0.5))
        lines = []

        # Define a helper function to emit #box lines for a given mask and z-range.
        def emit(mask, n_lo, n_hi, tag):
            for j in range(mask.shape[0]):
                i = 0
                while i < mask.shape[1]:
                    if not mask[j, i]:
                        i += 1
                        continue
                    k = i + 1
                    while k < mask.shape[1] and mask[j, k] and n_lo[j, k] == n_lo[j, i] and n_hi[j, k] == n_hi[j, i]:
                        k += 1
                    lines.append(f"#box: {x0[i]:.6f} {y0[j]:.6f} {(n_belt + n_lo[j, i]) * self.dz:.6f} "
                                 f"{x0[k - 1] + tile:.6f} {y0[j] + tile:.6f} {(n_belt + n_hi[j, i]) * self.dz:.6f} {tag}\n")
                    i = k

        # Emit #box lines for the main 'sut' columns and each type of region.
        emit(col['inside'], col['n_lo'], col['n_hi'], 'sut')
        for m, n_lo, n_hi in col['skin']:
            emit(m, n_lo, n_hi, 'skin')
        for m, n_lo, n_hi in col['rim']:
            emit(m, n_lo, n_hi, 'rim')
        for m, n_lo, n_hi in col['gaps']:
            emit(m, n_lo, n_hi, 'free_space')
        for m, n_lo, n_hi in col['shell']:
            emit(m, n_lo, n_hi, 'fo_shell')
        for r, (m, n_lo, n_hi) in enumerate(col['regions']):
            emit(m, n_lo, n_hi, f'region_{r}')
        print(f"_freeform_geometry_lines(): {len(lines)} boxes (tile {tile * 1e3:.3f} mm, {int(col['inside'].sum())} columns, {len(col['regions'])} region(s)).")
        return ''.join(lines) + '\n'

    def _sut_layer_z_ranges(self) -> list:
        """This function calculates the absolute z-range for each SUT layer, starting from the top layer down to the bottom layer.

        Args:
            None

        Returns:
            list of tuples: Each tuple contains the (z_lo, z_hi) range for a SUT layer, top-first.

        Raises:
            None
        """
        # Start from the bottom of the SUT stack and calculate the z-range for each layer up to the top.
        z_bottom = self.l_z_grid / 2
        z_hi = z_bottom + sum(layer['THICKNESS'] for layer in self.sut_layers)
        ranges = []
        for i, layer in enumerate(self.sut_layers):
            z_lo = z_bottom if i == len(self.sut_layers) - 1 else z_hi - layer['THICKNESS']
            ranges.append((z_lo, z_hi))
            z_hi = z_lo

        return ranges

    def _fo_layer_index(self) -> int:
        """This function calculates the absolute z-coordinate of the foreign object's center and determines which SUT layer it falls into.

        Args:
            None

        Returns:
            int: Index of the SUT layer containing the foreign object's center. If the center is outside the stack, returns the nearest end layer.

        Raises:
            None
        """
        # Calculate the absolute z-coordinate of the foreign object's center. Then determine which SUT layer it falls into.
        total = sum(layer['THICKNESS'] for layer in self.sut_layers)
        z_center = self.l_z_grid / 2 + self.foreign_object['POSITION_RELATIVE'][2] * total
        ranges = self._sut_layer_z_ranges()
        for i, (z_lo, z_hi) in enumerate(ranges):
            if z_lo - 1e-9 <= z_center <= z_hi + 1e-9:
                return i
            
        return 0 if z_center > ranges[0][1] else len(ranges) - 1

    def _determine_number_of_elements(self) -> None:
        """This function determines the number of grid cells per axis from the spatial resolution and grid dimensions.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        # Determine highest relative permittivity, which is required to define the spatial discretization.
        eps_max = self.get_max_relative_permittivity(self.frequency_range[-1])

        # Determine discretization of Yee's grid based on smallest frequency
        self.dx = (sp.constants.c/(self.frequency_range[-1]*eps_max))/self.cells_per_wavelength
        self.dy = self.dx
        self.dz = self.dx

        # Determine the number of elements in each dimension based on the grid dimensions and spatial resolution
        self.nx = int(self.l_x_grid / self.dx)
        self.ny = int(self.l_y_grid / self.dy)
        self.nz = int(self.l_z_grid / self.dz)
        self.n_elements = self.nx * self.ny * self.nz

        # Determine the error in the number of elements due to rounding and ensure that the grid dimensions are properly covered
        error_nx = self.l_x_grid - self.nx * self.dx
        error_ny = self.l_y_grid - self.ny * self.dy
        error_nz = self.l_z_grid - self.nz * self.dz

        # Print the errors with regard to the number of elements
        if (error_nx > 0 or error_ny > 0 or error_nz > 0):
            print(f"_determine_number_of_elements(): Warning - the grid dimensions are not perfectly divisible by the spatial resolution. Consider adjusting the grid dimensions or spatial resolution to ensure proper coverage. Error in x: {error_nx}, Error in y: {error_ny}, Error in z: {error_nz}")

    def _determine_time_parameters(self, show_results: bool = False) -> None:
        """This function determines the CFL-limited time step, time window and number of time steps.

        Args:
            show_results (bool): 
                Show relevant results of the computation.

        Returns:
            None

        Raises:
            None
        """
        # Determine time discretization based on CFL condition
        self.dt_max = 1 / (min(self.frequency_range) * 10)
        self.dt = 1/(sp.constants.c * np.sqrt((1/self.dx)**2 + 
                                              (1/self.dy)**2 + 
                                              (1/self.dz)**2))
        
        # Check if time step is below boundary set by the highest frequency
        if self.dt > self.dt_max:
            print(f"self._determine_time_parameters(): Warning: The time step determined by the CFL condition ({self.dt:.2e} s) exceeds the maximum time step allowed by the highest excitation frequency ({self.dt_max:.2e} s). Consider reducing the spatial resolution or adjusting the frequency range to ensure stability of the simulation.")
            self.dt = self.dt_max

        # The time window is set to 1.5 times the round trip distance between the TX and RX antennas.
        self.time_window = 1.5 * (abs(self.tx_pos_range[2]) + abs(self.rx_pos_range[2]))/sp.constants.c 

        # Determine total number of time steps
        self.n_time = int(self.time_window / self.dt) + 1
        self.time = np.arange(self.n_time) * self.dt

        # Show results if desired
        if show_results:
            print(f'self._determine_time_parameters(): dt = {self.dt} s; n_time = {self.n_time}')

    def _check_farfield_condition(self) -> None:
        """This function checks whether the farfield (Fraunhofer) condition is satisfied for the current TX/RX geometry.The Fraunhofer distance is calculated based on the aperture size and the speed of light.The condition is considered satisfied if both the TX and RX antennas are located beyond the Fraunhofer distance.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        # Get distance to the farfield region
        distance_farfield = 2*self.aperture**2/(sp.constants.c)

        # Check if both TX and RX are beyond the Fraunhofer distance to determine farfield condition.
        if distance_farfield < abs(self.tx_pos_range[2]) and distance_farfield < abs(self.rx_pos_range[2]):
            print(f'self._check_farfield_condition(): farfield condition applies (fraunhofer distance = {distance_farfield}; tx = {self.tx_pos_range[2]}; rx = {self.rx_pos_range[2]})')
            self.farfield = True
        else:
            print(f'self._check_farfield_condition(): farfield condition does not fully apply (fraunhofer distance = {distance_farfield}; tx = {self.tx_pos_range[2]}; rx = {self.rx_pos_range[2]})')
            self.farfield = False

    def _generate_waveform(self, show_results: bool = False) -> np.ndarray:
        """This function generates the excitation waveform based on the configured sweeping time and frequency range. It computes the chirp signal and the Gaussian pulse used as the excitation source in the simulation.

        Args:
            show_results (bool): Show results (default: False).

        Returns:
            np.ndarray: Complex waveform samples.

        Raises:
            None
        """
        # Determine center frequency of the chirp signal
        f0, f1 = self.frequency_range
        self.f_center = (f0 + f1) / 2
        
        # Show results if desired
        if show_results:

            # Determine chirp time, number of time points and chirp signal waveform
            self.dt_chirp = 1 / (2 * self.frequency_range[-1])
            self.t_chirp = np.arange(0, self.sweeping_time, self.dt_chirp)
            self.n_chirp = len(self.t_chirp)

            print(f'self._generate_waveform(): t_chirp = {self.t_chirp} s; n_chirp = {self.n_chirp}; farfield: {self.farfield}')

            # Determine chirp waveform by computing sweep rate, phase, and center frequency as well. This is only required for convolution with pulsed signals after the simulation to reconstruct the received signal since the simulation will be only carried out for a pulsed signal (reduces overall simulation time).
            sweep_rate = (f1 - f0) / self.sweeping_time
            phase = 2 * np.pi * (f0 * self.t_chirp + 
                                0.5 * sweep_rate * self.t_chirp**2)
            self.chirp_signal = np.exp(1j * phase)

        # Reproduce gprMax's 'gaussian' waveform
        t = self.time
        chi = 1 / self.f_center
        zeta = 2 * np.pi**2 * self.f_center**2
        self.E_tx = self.tx_amplitude * np.exp(-zeta * (t - chi)**2)

        # Plot if desired
        if show_results:

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))
            fig.canvas.manager.set_window_title("Waveforms")
            ax = ax.flatten()

            ax[0].plot(self.time, self.E_tx)
            ax[0].set_title("Pulse (used as excitation source in the simulation)")
            ax[0].set_xlabel("Time / s")
            ax[0].set_ylabel("Amplitude / 1")
            ax[0].set_xlim(0, min(8 * chi, self.time[-1]))

            # Compute the FFT of the pulse (power spectrum) to analyze its frequency content.
            E_tx_fft = np.fft.rfft(self.E_tx)
            freqs = np.fft.rfftfreq(self.n_time, d=self.dt)
            mag = np.abs(E_tx_fft)
            power_db = 20 * np.log10(np.maximum(mag, np.finfo(float).tiny) / mag.max())

            # Plot the power spectrum in dB relative to its peak.
            ax[1].plot(freqs, power_db)
            ax[1].axvspan(self.frequency_range[0], self.frequency_range[1],
                          color='steelblue', alpha=0.15, label='retained band (band_mask)')
            ax[1].axvline(self.f_center, color='gray', 
                          linestyle=':', linewidth=1,
                          label=f'f_center = {self.f_center / 1e9:.0f} GHz (waveform\'s f2, not the spectral peak)')
            ax[1].set_xlim(0, 3 * self.frequency_range[-1])
            ax[1].set_ylim(bottom=max(power_db.min(), -80.0))
            ax[1].legend(fontsize=7)
            ax[1].set_title("Pulse power spectrum")
            ax[1].set_xlabel("Frequency / Hz")
            ax[1].set_ylabel("Power / dB (rel. to peak)")

            # Plot the real and imaginary parts of the chirp signal.
            ax[2].plot(self.time, np.real(self.chirp_signal[0:self.n_time]), label='Real')
            ax[2].plot(self.time, np.imag(self.chirp_signal[0:self.n_time]), label='Imag')
            ax[2].legend()
            ax[2].set_title("Chirp (to be convolved with impulse response")
            ax[2].set_xlabel("Time / s")
            ax[2].set_ylabel("Amplitude / 1")

            plt.tight_layout()
            plt.show(block=False)

    def _determine_complexity_and_resources(self, 
                                            n_eq_per_cell: int = 6, n_bytes_per_eq: int = 8,
                                            show_results: bool = False) -> None:
        """This function estimates the memory required for the simulation grid based on the number of equations per cell and the bytes per equation.

        Args:
            n_eq_per_cell (int): 
                Equations per grid cell.
            n_bytes_per_eq (int): 
                Bytes per equation.

        Returns:
            None

        Raises:
            None
        """
        # Estimate storage requirements for simulation (not for saving results after simulation!)
        self.memory_sample_under_test_gb = self.n_elements * n_eq_per_cell * n_bytes_per_eq / (1000**3) * 1.5
        if show_results:
            print(f"self._determine_complexity_and_resources(): approximately {self.memory_sample_under_test_gb} GB of memory required.")

        pass

    def get_max_relative_permittivity(self, frequency):
        """This function evaluates the maximum relative permittivity of the materials at the specified frequency. This is useful for determining the effective permittivity in dispersive and non-dispersive materials.

        Args:
            frequency (float): Frequency [Hz].

        Returns:
            float: Maximum relative permittivity.
        """
        # Loop over all materials to determine the maximum relative permittivity at the given frequency.
        omega = 2 * np.pi * frequency
        eps_max = 0.0
        for name, material in self.material_properties.items():

            # Non-dispersive material
            if 'PERMITTIVITY_RELATIVE' in material:
                eps = max(material['PERMITTIVITY_RELATIVE'])

            # Debye material
            elif material.get('DISPERSION_MODEL', '').upper() == 'DEBYE':

                eps_s = material['EPS_STATIC']
                eps_inf = material['EPS_INF']
                tau = material['TAU']
                eps_complex = (eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau))
                eps = eps_complex.real

            else:

                continue

            eps_max = max(eps_max, eps)

        return eps_max

    @staticmethod
    def _debye_model_by_spectral_mixing(fractions, materials, type='looyenga', freq=None) -> dict:
        """This function takes a set of constituent materials with known Debye parameters and their volume/mass fractions, and computes an effective single-pole Debye model that approximates the mixture over the specified frequency range.

        Args:
            fractions (dict): 
                Ingredient name -> volume/mass fraction (sum ~1).
            materials (dict): 
                Full material property dictionary (self.material_properties).
            type (str): 
                Mixing rule, 'looyenga' (default) or 'linear'.
            freq (np.ndarray): 
                Frequency axis to fit over [Hz] (default: 50e9-300e9, 200 pts).

        Returns:
            dict: EPS_INF, DELTA_EPS, EPS_STATIC, TAU, FIT_RELATIVE_RESIDUAL

        Raises:
            ValueError: If 'type' is not 'looyenga' or 'linear'.
        """

        # Define helper function for constituent Debye model
        def debye_eps(mat, omega):
            eps_s = mat['EPS_STATIC']
            eps_inf = mat['EPS_INF']
            tau = mat['TAU']
            return eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

        # Define helper function for single-pole Debye model to fit the effective permittivity
        def debye_model(x, omega):
            eps_inf, delta_eps, tau = x
            return eps_inf + delta_eps / (1 + 1j * omega * tau)

        # Define helper function for residual function for least-squares fitting
        def residuals(x):
            eps_fit = debye_model(x, omega)
            return np.concatenate([
                np.real(eps_fit - eps_eff),
                np.imag(eps_fit - eps_eff)])

        # Set default frequency axis if not provided
        if freq is None:
            freq = np.linspace(50e9, 300e9, 200)
        omega = 2 * np.pi * freq

        # Compute the effective permittivity using the specified mixing rule.
        eps_eff = np.zeros_like(freq, dtype=complex)
        if type == 'looyenga':
            for k, frac in fractions.items():
                eps_i = debye_eps(materials[k], omega)
                eps_eff += frac * eps_i ** (1 / 3)
            eps_eff = eps_eff ** 3
        elif type == 'linear':
            for k, frac in fractions.items():
                eps_i = debye_eps(materials[k], omega)
                eps_eff += frac * eps_i
        else:
            raise ValueError("Unknown mixing type")

        # Generate initial guess for Debye parameters based on effective permittivity data
        eps_inf_guess = eps_eff[-1].real
        eps_static_guess = eps_eff[0].real
        delta_guess = max(eps_static_guess - eps_inf_guess, 1e-6)
        tau_guess = 1 / (2 * np.pi * np.median(freq))
        x0 = [max(eps_inf_guess, 1e-6), delta_guess, tau_guess]

        # Fit single Debye pole with physically valid values as bounds.
        results_ls = least_squares(residuals, x0, 
                                    bounds=([0.0, 0.0, 0.0], 
                                    [np.inf, np.inf, np.inf]))
        eps_inf, delta_eps, tau = results_ls.x

        # Compute fit-quality metric as the relative residual norm.
        residual_norm = np.linalg.norm(results_ls.fun)
        reference_norm = np.linalg.norm(np.concatenate([eps_eff.real, 
                                                        eps_eff.imag]))
        fit_relative_residual = residual_norm / reference_norm if reference_norm > 0 else np.nan

        return {'EPS_INF': eps_inf,
                'DELTA_EPS': delta_eps,
                'EPS_STATIC': eps_inf + delta_eps,
                'TAU': tau,
                'FIT_RELATIVE_RESIDUAL': fit_relative_residual}

    @staticmethod
    def _mix_conductivity(fractions, materials):
        """This function computes the effective conductivity of a mixture using a fraction-weighted average of the individual conductivities.

        Args:
            fractions (dict): 
                Ingredient name -> fraction.
            materials (dict): 
                Full material property dictionary.

        Returns:
            float: Effective conductivity in S/m.

        Raises:
            KeyError: If a fraction references a material not present in 'materials'.
        """
        sigma_eff = 0.0
        for k, f in fractions.items():
            sigma_i = materials[k].get('CONDUCTIVITY', 0.0)
            if isinstance(sigma_i, (list, tuple)):
                sigma_i = sum(sigma_i) / len(sigma_i)
            sigma_eff += f * sigma_i

        return sigma_eff

    @staticmethod
    def _complex_permittivity_at(frequency: float, 
                                 debye_model: dict,
                                 permittivity_relative: float,
                                 conductivity: float) -> complex:
        """This function computes the complex relative permittivity of a material at a given frequency, considering both dispersive and conductive contributions (eps_r(f) = eps' - j*eps'').

        Args:
            frequency (float): 
                Frequency [Hz] > 0.
            debye_model (dict or None): 
                EPS_INF/DELTA_EPS/TAU dict, or None for a non-dispersive material.
            permittivity_relative (float): 
                Non-dispersive relative permittivity (only used if debye_model is None).
            conductivity (float): 
                Conductivity [S/m] (0 for none).

        Returns:
            complex: eps_r(f); lossy materials have a negative imaginary part.

        Raises:
            ValueError: If frequency <= 0.
        """
        omega = 2 * np.pi * frequency
        if debye_model is not None:
            eps_r = (debye_model['EPS_INF']
                      + debye_model['DELTA_EPS'] / (1 + 1j * omega * debye_model['TAU']))
        else:
            eps_r = complex(permittivity_relative, 0.0)
        if conductivity:
            eps_r = eps_r - 1j * conductivity / (omega * _VACUUM_PERMITTIVITY_F_PER_M)

        return eps_r

    @staticmethod
    def _dielectric_contrast(frequency: float, dielectric: dict) -> dict:
        """This function calculates the dielectric contrast between the foreign object and the SUT at the specified frequency.

        Args:
            frequency (float): 
                Frequency in Hz at which both permittivities are evaluated.
            dielectric (dict or None): 
                The run's 'dielectric' snapshot from data.pkl; None returns None.

        Returns:
            dict, or None if dielectric is None/incomplete or sut_mode == 'SURROUNDING':
                eps_fo, eps_sut (complex): eps_r(frequency) of each material.
                contrast_real (float): (eps_fo.real - eps_sut.real) / eps_sut.real.
                contrast_complex_mag (float): abs(eps_fo - eps_sut) / abs(eps_sut).
                contrast_complex_phase (float): Phase difference between eps_fo and eps_sut in radians.

        Raises:
            ValueError: If frequency <= 0.
        """
        # Return None if dielectric information is missing or if SUT is not in 'INGREDIENTS' mode.
        if dielectric is None:
            return None
        if (dielectric.get('sut_mode') != 'INGREDIENTS'
                or dielectric.get('debye_model_sut') is None):
            return None

        # Calculate complex permittivity of SUT and foreign object at the specified frequency.
        eps_sut = ModelMixin._complex_permittivity_at(
            frequency, dielectric['debye_model_sut'], None,
            dielectric['conductivity_sut'])
        eps_fo = ModelMixin._complex_permittivity_at(
            frequency, dielectric['debye_model_fo'],
            dielectric['permittivity_relative_fo'],
            dielectric['conductivity_fo'])

        # Calculate contrasts between SUT and foreign object.
        contrast_real = ((eps_fo.real - eps_sut.real) / eps_sut.real
                          if eps_sut.real != 0 else np.nan)
        contrast_complex_mag = (abs(eps_fo - eps_sut) / abs(eps_sut)
                                 if abs(eps_sut) != 0 else np.nan)
        
        return {'eps_fo': eps_fo, 'eps_sut': eps_sut,
                'contrast_real': contrast_real,
                'contrast_complex_mag': contrast_complex_mag}

    def _determine_material_properties(self) -> None:
        """This function calculates and sets the material properties (permittivity, permeability and conductivity etc.) for the belt, SUT, and foreign object based on the specified modes and layers.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If any required material property is missing or invalid.
        """

        # Determine belt properties (assumption: non-dispersive)
        belt_parameters = self.material_properties[self.belt['MATERIAL']]
        self.permittivity_relative_belt = np.mean(belt_parameters['PERMITTIVITY_RELATIVE'])
        self.conductivity_belt = np.mean(belt_parameters['CONDUCTIVITY'])
        self.permeability_relative_belt = belt_parameters['PERMEABILITY_RELATIVE']
        self.magnetic_loss_belt = belt_parameters['MAGNETIC_LOSS']
        self.thickness_belt = self.belt['THICKNESS']

        # Determine SUT properties only if in 'INGREDIENTS' mode.
        if self.sut_mode == 'INGREDIENTS':

            # Fit the Debye model over the specified frequency range.
            fit_freq = np.linspace(self.frequency_range[0], 
                                   self.frequency_range[1], 200)
            if self.frequency_bands:
                fit_freq = np.concatenate([np.linspace(f_lo, f_hi, 40) for f_lo, f_hi in self.frequency_bands])

            # Iterate over each layer to compute and store the Debye model, conductivity, permeability, and magnetic loss for each SUT layer
            self.debye_model_sut_layers = []
            self.conductivity_sut_layers = []
            self.permeability_relative_sut_layers = []
            self.magnetic_loss_sut_layers = []
            for layer in self.sut_layers:
                debye_model = self._debye_model_by_spectral_mixing(
                                        layer['INGREDIENTS_FRACTIONS'],
                                        self.material_properties,
                                        type='looyenga',
                                        freq=fit_freq)

                # Check the quality of the Debye fit and warn if the relative residual is too high.
                if debye_model['FIT_RELATIVE_RESIDUAL'] > 0.05:
                    print(f"determine_material_properties(): Warning - single-pole Debye fit "
                        f"for SUT layer '{layer['NAME']}' has a relative residual of {debye_model['FIT_RELATIVE_RESIDUAL']:.3f} over [{fit_freq[0]:.3e}, {fit_freq[-1]:.3e}] Hz. Consider a two-pole model or a narrower fitting range.")

                # Compute the effective conductivity for this layer using the specified mixing rule.
                conductivity = self._mix_conductivity(
                    layer['INGREDIENTS_FRACTIONS'],
                    self.material_properties)

                # Compute the effective permeability and magnetic loss for this layer.
                mu_static = 0.0
                mu_loss = 0.0
                for k, f in layer['INGREDIENTS_FRACTIONS'].items():
                    mu_static += f * self.material_properties[k]['PERMEABILITY_RELATIVE']
                    mu_loss += f * self.material_properties[k]['MAGNETIC_LOSS']

                # Round the static permeability to avoid floating-point issues with gprMax.
                self.debye_model_sut_layers.append(debye_model)
                self.conductivity_sut_layers.append(conductivity)
                self.permeability_relative_sut_layers.append(round(mu_static, 12))
                self.magnetic_loss_sut_layers.append(mu_loss)

            # The reference layer's properties are extracted from the corresponding layer in the SUT stack.
            self.fo_layer_index = self._fo_layer_index()
            i_ref = self.fo_layer_index
            self.debye_model_sut = self.debye_model_sut_layers[i_ref]
            self.conductivity_sut = self.conductivity_sut_layers[i_ref]
            self.permeability_relative_sut = self.permeability_relative_sut_layers[i_ref]
            self.magnetic_loss_sut = self.magnetic_loss_sut_layers[i_ref]

            # Compute the inclusion's Debye model, conductivity, and permeability if it exists.
            self.debye_model_inclusion = None
            if self.sut_inclusions is not None:
                inc_fractions = self.sut_inclusions['INGREDIENTS_FRACTIONS']
                self.debye_model_inclusion = self.      _debye_model_by_spectral_mixing(
                    inc_fractions, 
                    self.material_properties,
                    type='looyenga', 
                    freq=fit_freq)
                self.conductivity_inclusion = self._mix_conductivity(
                    inc_fractions, 
                    self.material_properties)
                self.permeability_relative_inclusion = round(sum(
                    f * self.material_properties[k]['PERMEABILITY_RELATIVE']
                    for k, f in inc_fractions.items()), 12)
                self.magnetic_loss_inclusion = sum(
                    f * self.material_properties[k]['MAGNETIC_LOSS']
                    for k, f in inc_fractions.items())

            # Each freeform region gets its own Debye mixture, conductivity, and permeability based on its ingredient fractions.
            self.region_materials = []
            for reg in (self.sut_freeform or {}).get('REGIONS', []):
                fr = reg['INGREDIENTS_FRACTIONS']
                self.region_materials.append(dict(
                    debye=self._debye_model_by_spectral_mixing(fr, self.material_properties, type='looyenga', freq=fit_freq),
                    sigma=self._mix_conductivity(fr, self.material_properties),
                    mu=round(sum(f * self.material_properties[k]['PERMEABILITY_RELATIVE'] for k, f in fr.items()), 12),
                    mu_loss=sum(f * self.material_properties[k]['MAGNETIC_LOSS'] for k, f in fr.items())))

            # FREEFORM moisture skin and punched rim: own Debye mixtures, written as 'skin' / 'rim'.
            # Determine Debye model, conductivity, and permeability for the FREEFORM skin, rim, and fo_shell
            self.skin_material = self.rim_material = self.fo_shell_material = None
            for key in ('SKIN', 'RIM', 'FO_SHELL'):
                blk = (self.sut_freeform or {}).get(key)
                if not blk:
                    continue
                fr = blk['INGREDIENTS_FRACTIONS']
                mat = dict(
                    debye=self._debye_model_by_spectral_mixing(fr, self.material_properties, type='looyenga', freq=fit_freq),
                    sigma=self._mix_conductivity(fr, self.material_properties),
                    mu=round(sum(f * self.material_properties[k]['PERMEABILITY_RELATIVE'] for k, f in fr.items()), 12),
                    mu_loss=sum(f * self.material_properties[k]['MAGNETIC_LOSS'] for k, f in fr.items()))
                setattr(self, {'SKIN': 'skin_material', 
                               'RIM': 'rim_material', 
                               'FO_SHELL': 'fo_shell_material'}[key], mat)
        else:
            print("determine_material_properties(): SUT mode is 'SURROUNDING' - no SUT material will be written; that region of the domain stays whatever surrounds it (air).")

            self.debye_model_sut = None
            self.conductivity_sut = None
            self.permeability_relative_sut = None
            self.magnetic_loss_sut = None
            self.debye_model_sut_layers = None
            self.conductivity_sut_layers = None
            self.permeability_relative_sut_layers = None
            self.magnetic_loss_sut_layers = None
            self.debye_model_inclusion = None
            self.region_materials = []
            self.skin_material = self.rim_material = self.fo_shell_material = None
            self.fo_layer_index = 0

        # The shape and size of the SUT are needed to define the bounding box for the foreign object placement.
        self.shape_sut = self.sut['SHAPE']

        # Determine the electromagnetic properties of the foreign object based on the selected mode.
        if self.fo_mode == 'MATERIAL':
            fo_parameters = self.material_properties[self.foreign_object['MATERIAL']]
            self.permittivity_relative_fo = np.mean(fo_parameters['PERMITTIVITY_RELATIVE'])
            self.conductivity_fo = np.mean(fo_parameters['CONDUCTIVITY'])
            self.permeability_relative_fo = np.mean(fo_parameters['PERMEABILITY_RELATIVE'])
            self.magnetic_loss_fo = np.mean(fo_parameters['MAGNETIC_LOSS'])
            self.debye_model_fo = None
        else:
            print("determine_material_properties(): foreign-object mode is 'SUT' - the foreign object will use the SUT's own electromagnetic properties (including its Debye dispersion), i.e. it will not be detectable.")
            self.permittivity_relative_fo = self.debye_model_sut['EPS_INF']
            self.conductivity_fo = self.conductivity_sut
            self.permeability_relative_fo = self.permeability_relative_sut
            self.magnetic_loss_fo = self.magnetic_loss_sut
            self.debye_model_fo = self.debye_model_sut

    def _determine_antenna_location_huygens(self):
        """This function determines the positions of the TX and RX antennas on the Huygens surface based on the specified position ranges and the number of antennas.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # Get radius from the domain's centroid to the Huygens surface (along z)
        delta_z = self.l_z_grid / 2

        # Construct the (3, n_tx) TX position array
        tx_x = np.linspace(self.tx_pos_range[0][0] + self.l_x_grid / 2,
                            self.tx_pos_range[0][-1] + self.l_x_grid / 2,
                            self.n_tx)

        # Define a helper function to determine y positions based on the specification (scalar or range) and number of antennas.
        def _y_positions(y_spec, n):
            if np.ndim(y_spec) == 0:
                return np.full(n, y_spec + self.l_y_grid / 2)
            return np.linspace(y_spec[0], y_spec[-1], n) + self.l_y_grid / 2

        # Determine the y positions for the TX antennas using the helper function.
        tx_y = _y_positions(self.tx_pos_range[1], self.n_tx)
        tx_z = np.full(self.n_tx, self.tx_pos_range[2] + self.l_z_grid / 2)
        tx_pos = np.vstack([tx_x, tx_y, tx_z])          # shape (3, n_tx)
        self.tx_pos = tx_pos  # kept for visualization (_draw_scene())

        # Construct the (3, n_rx) RX position array using the same approach as for the TX array.
        rx_x = np.linspace(self.rx_pos_range[0][0] + self.l_x_grid / 2,
                            self.rx_pos_range[0][-1] + self.l_x_grid / 2,
                            self.n_rx)
        rx_y = _y_positions(self.rx_pos_range[1], self.n_rx)
        rx_z = np.full(self.n_rx, self.rx_pos_range[2] + self.l_z_grid / 2)
        rx_pos = np.vstack([rx_x, rx_y, rx_z])          # shape (3, n_rx)
        self.rx_pos = rx_pos  # kept for visualization (_draw_scene())

        # Determine the domain center based on the grid dimensions.
        self.domain_center = np.array([[self.l_x_grid / 2],
                                  [self.l_y_grid / 2],
                                  [self.l_z_grid / 2]])

        # Determine the target center based on the SUT offsets and grid dimensions.
        self.target_center = np.array([[self.sut_cx],
                                       [self.sut_cy],
                                       [self.l_z_grid / 2]])

        # Compute the distance vectors from the TX and RX antennas to the target center and normalize them.
        d_tx = self.target_center - tx_pos  # (3, n_tx)
        d_tx /= np.linalg.norm(d_tx, axis=0)
        d_rx = self.target_center - rx_pos  # (3, n_rx)
        d_rx /= np.linalg.norm(d_rx, axis=0)

        # Compute the incident and received wave angles (alpha: elevation, beta: azimuth) based on the normalized distance vectors.
        self.incident_wave_alpha = np.arccos(d_tx[2])
        self.incident_wave_beta = np.arctan2(d_tx[1], d_tx[0])
        self.received_wave_alpha = np.arccos(d_rx[2])
        self.received_wave_beta = np.arctan2(d_rx[1], d_rx[0])

        # Compute the wave vectors for the incident and received waves based on the angles.
        self.k_tx = np.array([
            np.sin(self.incident_wave_alpha) * np.cos(self.incident_wave_beta),
            np.sin(self.incident_wave_alpha) * np.sin(self.incident_wave_beta),
            np.cos(self.incident_wave_alpha)])
        self.k_rx = np.array([
            np.sin(self.received_wave_alpha) * np.cos(self.received_wave_beta),
            np.sin(self.received_wave_alpha) * np.sin(self.received_wave_beta),
            np.cos(self.received_wave_alpha)])

        # Determine points on the Huygens surface
        self.tx_huygens = self.target_center - delta_z * self.k_tx
        self.rx_huygens = self.target_center - delta_z * self.k_rx

    def _validate_geometry(self) -> None:
        """This function checks whether the belt, SUT, and foreign object fit within the simulation domain and remain clear of the PML region.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        # Define a helper function to check if a box is within the domain bounds and PML region.
        def _check_box(name, p0, p1, lo, hi):
            for axis, label in enumerate(('x', 'y', 'z')):
                if p0[axis] < lo[axis] - 1e-9:
                    print(f"_validate_geometry(): Warning - {name} extends "
                          f"{lo[axis] - p0[axis]:.4f} m below the {label}_min = "
                          f"{lo[axis]:.4f} m boundary ({name} {label}_min = "
                          f"{p0[axis]:.4f} m).")
                if p1[axis] > hi[axis] + 1e-9:
                    print(f"_validate_geometry(): Warning - {name} extends "
                          f"{p1[axis] - hi[axis]:.4f} m beyond the {label}_max = "
                          f"{hi[axis]:.4f} m boundary ({name} {label}_max = "
                          f"{p1[axis]:.4f} m).")

        # Define the domain bounds and the inner region (PML-free zone) for validation.
        domain_lo = (0.0, 0.0, 0.0)
        domain_hi = (self.l_x_grid, 
                     self.l_y_grid, 
                     self.l_z_grid)
        inner_lo = (self.domain_inner[0], 
                    self.domain_inner[2], 
                    self.domain_inner[4])
        inner_hi = (self.domain_inner[1], 
                    self.domain_inner[3], 
                    self.domain_inner[5])

        # The belt is assumed to span the full width of the domain in x and y, and is centered in z.
        p_0_belt = [0.0, 0.0, self.l_z_grid / 2 - self.thickness_belt]
        p_1_belt = [self.l_x_grid, self.l_y_grid, self.l_z_grid / 2]
        _check_box("belt", p_0_belt, p_1_belt, domain_lo, domain_hi)

        # SUT bounding box definition depends on the shape of the SUT. Must stay within the inner region (PML-free zone).
        if self.shape_sut == "CYLINDER":

            radius_sut = self.sut['SIZE']['DIAMETER'] / 2
            thickness_sut = self.sut['SIZE']['THICKNESS']
            p_sut_bbox = [
                [self.sut_cx - radius_sut,
                 self.sut_cy - radius_sut,
                 self.l_z_grid / 2],
                [self.sut_cx + radius_sut,
                 self.sut_cy + radius_sut,
                 self.l_z_grid / 2 + thickness_sut]]
            
        elif self.shape_sut in ("RECTANGULAR", "FREEFORM"):

            length_x_sut, length_y_sut, thickness_sut = self.sut['SIZE']
            p_sut_bbox = [
                [self.sut_cx - length_x_sut / 2,
                 self.sut_cy - length_y_sut / 2,
                 self.l_z_grid / 2],
                [self.sut_cx + length_x_sut / 2,
                 self.sut_cy + length_y_sut / 2,
                 self.l_z_grid / 2 + thickness_sut]]
            
        else:

            print(f"_validate_geometry(): Warning - unknown SUT shape "
                  f"'{self.shape_sut}', skipping SUT/foreign-object bounds check.")
            
            return

        # Check the SUT bounding box against the inner region to ensure it does not intersect the PML.
        if self.sut_mode == 'INGREDIENTS':
            _check_box("SUT", p_sut_bbox[0], p_sut_bbox[1], inner_lo, inner_hi)

        # Foreign-object bounding box definition depends on its size and relative position to the SUT. Must stay within the inner region (PML-free zone).
        fo_size = self.foreign_object['SIZE']
        fo_rel_pos = self.foreign_object['POSITION_RELATIVE']
        fo_center = [p_sut_bbox[0][axis] + fo_rel_pos[axis] *
                     (p_sut_bbox[1][axis] - p_sut_bbox[0][axis])
                     for axis in range(3)]
        p_0_fo = [fo_center[axis] - fo_size[axis] / 2 for axis in range(3)]
        p_1_fo = [fo_center[axis] + fo_size[axis] / 2 for axis in range(3)]
        _check_box("foreign object", p_0_fo, p_1_fo, inner_lo, inner_hi)

        # For a cylindrical SUT, ensure that all corners of the foreign-object bounding box lie within the cylinder's cross-sectional radius.
        if self.shape_sut == "CYLINDER":
            axis_x, axis_y = self.sut_cx, self.sut_cy
            corners_xy = [(p_0_fo[0], p_0_fo[1]), (p_0_fo[0], p_1_fo[1]),
                          (p_1_fo[0], p_0_fo[1]), (p_1_fo[0], p_1_fo[1])]
            max_corner_radius = max(np.hypot(x - axis_x, y - axis_y) for x, y in corners_xy)
            if max_corner_radius > radius_sut:
                print(f"_validate_geometry(): Warning - the foreign object (POSITION_RELATIVE={fo_rel_pos}, SIZE={fo_size}) extends {max_corner_radius:.4f} m from the cylinder axis, outside the SUT radius of {radius_sut:.4f} m. Consider reducing SIZE or moving POSITION_RELATIVE closer to 0.5 in x/y.")

    def _compute_source_array_stride(self) -> int:
        """This function computes the maximum allowable stride for the Huygens-surface dipole source array based on the grating-lobe criterion (d_max = lambda_min / (1 + max|sin(alpha)|)) and a safety margin (20 % by default).

        Args:
            None

        Returns:
            int: Number of Yee cells (self.dx) between adjacent dipole sources along x and y (1 = one per cell).

        Raises:
            None
        """
        # Compute the minimum wavelength and the maximum sine of the incident angle for the grating-lobe criterion.
        lambda_min = sp.constants.c / self.frequency_range[-1]
        sin_theta_max = np.max(np.abs(np.sin(self.incident_wave_alpha)))

        # Compute the grating-lobe limit and the safe target spacing with a 20% safety margin.
        d_grating_limit = lambda_min / (1 + sin_theta_max)
        d_max_safe = min(d_grating_limit, lambda_min / 2)
        d_target = 0.8 * d_max_safe

        # Convert the target spacing to a stride in terms of the number of Yee cells.
        stride = max(1, int(np.floor(d_target / self.dx)))

        print(f"_compute_source_array_stride(): lambda_min={lambda_min * 1e3:.3f} mm, theta_max={np.degrees(np.arcsin(sin_theta_max)):.2f} deg off broadside, grating-lobe-free spacing <= {d_max_safe * 1e3:.3f} mm, using stride={stride} (source spacing={stride * self.dx * 1e3:.3f} mm)")

        return stride

    def _compute_receiver_geometry(self) -> None:
        """This function sets up the receiver geometry, including the PML boundaries, the inner domain, and the Huygens receiver surface.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        # Define PML boundary conditions and inner domain (same for every TX position,
        # so this only needs to be computed once)
        self.pml_thickness = [self.n_pml * self.dx,
                              self.n_pml * self.dy,
                              self.n_pml * self.dz]
        self.domain_inner = [
            0.0 + self.pml_thickness[0] + self.n_pml_margin * self.dx,
            self.l_x_grid - self.pml_thickness[0] - self.n_pml_margin * self.dx,
            0.0 + self.pml_thickness[1] + self.n_pml_margin * self.dy,
            self.l_y_grid - self.pml_thickness[1] - self.n_pml_margin * self.dy,
            0.0 + self.pml_thickness[2] + self.n_pml_margin * self.dz,
            self.l_z_grid - self.pml_thickness[2] - self.n_pml_margin * self.dz]

        # Compute the number of grid cells along each axis within the inner domain.
        self.n_x_domain_inner = (self.domain_inner[1] -
                                 self.domain_inner[0]) / self.dx
        self.n_y_domain_inner = (self.domain_inner[3] -
                                 self.domain_inner[2]) / self.dy
        self.n_z_domain_inner = (self.domain_inner[5] -
                                 self.domain_inner[4]) / self.dz

        # Validate that the belt, SUT, and any foreign objects fit within the inner domain and do not intersect the PML region.
        self._validate_geometry()

        # Determine the subsampling stride for the receiver grid based on the shortest wavelength.
        d = self.dx
        lambda_min = sp.constants.c / (self.frequency_range[1] * 
                                       np.sqrt(self.get_max_relative_permittivity(self.frequency_range[1])))

        # Define helper function to build a receiver grid on a face of the inner-domain box.
        def _face_grid(fixed_axis, fixed_value, normal_sign, 
                       axis_a, range_a, axis_b, range_b, stride):
            """Build a regularly subsampled receiver grid on one axis-aligned plane of the inner-domain box.
            Args:
                fixed_axis (int): 
                    coordinate (0=x, 1=y, 2=z) held constant.
                fixed_value (float): 
                    value of the fixed coordinate.
                normal_sign (float): 
                    +1.0 or -1.0, the outward normal along fixed_axis.
                axis_a (int): 
                    first spanned coordinate index.
                range_a (tuple): 
                    (lo, hi) bounds for axis_a.
                axis_b (int): 
                    second spanned coordinate index.
                range_b (tuple): 
                    (lo, hi) bounds for axis_b.
                stride (int): subsampling stride (grid cells) along both spanned axes.

            Returns:
                (positions (N,3), normals (N,3), dS (N,) per-point patch area).

            Raises:
                ValueError: if the specified ranges are invalid or result in zero cells.
            """
            # Determine the number of cells along each spanned axis based on the stride and cell size d.
            n_a = int(round((range_a[1] - range_a[0]) / d))
            n_b = int(round((range_b[1] - range_b[0]) / d))

            # Define a helper function to compute node positions and quadrature weights along one axis.
            def _nodes_and_weights(n_cells):

                idx = list(range(0, max(n_cells, 1), stride))
                if idx[-1] != n_cells and n_cells > 0:
                    idx.append(n_cells)
                
                idx = np.asarray(idx, dtype=float)
                if len(idx) == 1:
                    return idx, np.array([max(n_cells, 1) * d])
                
                w = np.empty(len(idx))
                w[1:-1] = (idx[2:] - idx[:-2]) / 2.0
                w[0] = (idx[1] - idx[0]) / 2.0
                w[-1] = (idx[-1] - idx[-2]) / 2.0

                return idx, w * d

            # Compute the node positions and quadrature weights along both spanned axes.
            i_a, w_a = _nodes_and_weights(n_a)
            i_b, w_b = _nodes_and_weights(n_b)
            a_vals = range_a[0] + i_a * d
            b_vals = range_b[0] + i_b * d
            aa, bb = np.meshgrid(a_vals, b_vals, indexing='ij')
            wa, wb = np.meshgrid(w_a, w_b, indexing='ij')
            aa = aa.ravel()
            bb = bb.ravel()
            n_pts = len(aa)

            # Allocate arrays for the positions, normals, and differential surface elements.
            positions = np.zeros((n_pts, 3))
            positions[:, fixed_axis] = fixed_value
            positions[:, axis_a] = aa
            positions[:, axis_b] = bb

            # Compute the normal vector for the face and replicate it for all points.
            normal = np.zeros(3)
            normal[fixed_axis] = normal_sign
            normals = np.tile(normal, (n_pts, 1))

            # Compute the differential surface element for each point as the product of the quadrature weights along both axes.
            dS = (wa * wb).ravel()

            return positions, normals, dS

        # Determine the target spacing and stride for the top/bottom caps based on the desired Huygens sampling density.
        target_spacing_cap = lambda_min / self.rx_huygens_samples_per_wavelength
        stride_cap = max(1, int(round(target_spacing_cap / d)))

        # Determine the x-range for the Huygens surface, taking into account any user-specified window.
        x_range = (self.domain_inner[0], self.domain_inner[1])
        if self.huygens_x_window is not None:
            if self.huygens_closed_surface:
                raise ValueError(
                    "EMSimulation._compute_receiver_geometry(): HUYGENS_X_WINDOW cannot be combined with HUYGENS_CLOSED_SURFACE (the side walls would no longer close the surface).")
            x_range = (max(x_range[0], self.sut_cx - self.huygens_x_window),
                       min(x_range[1], self.sut_cx + self.huygens_x_window))
        
        # Determine the y-range for the Huygens surface.
        y_range = (self.domain_inner[2], self.domain_inner[3])
        z_lo, z_hi = self.domain_inner[4], self.domain_inner[5] - self.huygens_top_cap_offset_cells * d
        if z_hi <= z_lo + 0.5 * (self.domain_inner[5] - self.domain_inner[4]):
            raise ValueError(
                f"EMSimulation._compute_receiver_geometry(): HUYGENS_TOP_CAP_OFFSET_CELLS = {self.huygens_top_cap_offset_cells} moves the top cap to z = {z_hi * 1e3:.2f} mm, below the middle of the inner domain - reduce it.")
        if self.huygens_top_cap_offset_cells:
            print(f"_compute_receiver_geometry(): top cap moved {self.huygens_top_cap_offset_cells * d * 1e3:.3f} mm below the source plane (z = {z_hi * 1e3:.3f} mm; source plane z_src = {self.domain_inner[5] * 1e3:.3f} mm).")

        # Generate the grid for the top and bottom caps of the Huygens surface.
        pos_lo, nrm_lo, dS_lo = _face_grid(2, z_lo, 
                                           -1.0, 0, 
                                           x_range, 1, 
                                           y_range, 
                                           stride_cap)
        pos_hi, nrm_hi, dS_hi = _face_grid(2, z_hi, 
                                           +1.0, 0, 
                                           x_range, 1, 
                                           y_range, 
                                           stride_cap)

        # Interleave the positions, normals, and differential areas of the top and bottom caps (for every (x, y) point [lo, hi, lo, hi, ...]).
        n_cap_pts = len(pos_lo)
        cap_positions = np.empty((2 * n_cap_pts, 3))
        cap_normals = np.empty((2 * n_cap_pts, 3))
        cap_dS = np.empty(2 * n_cap_pts)
        cap_positions[0::2], cap_positions[1::2] = pos_lo, pos_hi
        cap_normals[0::2], cap_normals[1::2] = nrm_lo, nrm_hi
        cap_dS[0::2], cap_dS[1::2] = dS_lo, dS_hi

        # Compute the number of receiver points for the native grid (without subsampling).
        n_rx_native = (int(self.n_x_domain_inner) * 
                       int(self.n_y_domain_inner) * 2)
        print(f"_compute_receiver_geometry(): caps subsampled at {stride_cap}x grid spacing ({stride_cap * d * 1e3:.3f} mm vs. native {d * 1e3:.3f} mm). RX points: {n_rx_native} -> {len(cap_positions)} (reduced by a factor of {n_rx_native / max(len(cap_positions), 1):.1f}x).")

        # If the Huygens surface is closed, compute the side walls as well.
        if self.huygens_closed_surface:

            target_spacing_wall = lambda_min / self.rx_huygens_wall_samples_per_wavelength
            stride_wall = max(1, int(round(target_spacing_wall / d)))

            # Define the ranges for the side walls of the Huygens surface.
            z_range = (self.domain_inner[4], z_hi)
            x_lo, x_hi = self.domain_inner[0], self.domain_inner[1]
            y_lo, y_hi = self.domain_inner[2], self.domain_inner[3]

            # Compute the positions, normals, and differential areas for each side wall.
            pos_xlo, nrm_xlo, dS_xlo = _face_grid(0, x_lo, 
                                                  -1.0, 1, 
                                                  y_range, 2, 
                                                  z_range, stride_wall)
            pos_xhi, nrm_xhi, dS_xhi = _face_grid(0, x_hi, 
                                                  +1.0, 1, 
                                                  y_range, 2, 
                                                  z_range, stride_wall)
            pos_ylo, nrm_ylo, dS_ylo = _face_grid(1, y_lo, 
                                                  -1.0, 0, 
                                                  x_range, 2, 
                                                  z_range, stride_wall)
            pos_yhi, nrm_yhi, dS_yhi = _face_grid(1, y_hi, 
                                                  +1.0, 0, 
                                                  x_range, 2, 
                                                  z_range, stride_wall)

            # Concatenate the positions, normals, and differential areas of all side walls.
            wall_positions = np.concatenate([pos_xlo, pos_xhi, 
                                             pos_ylo, pos_yhi], axis=0)
            wall_normals = np.concatenate([nrm_xlo, nrm_xhi, 
                                           nrm_ylo, nrm_yhi], axis=0)
            wall_dS = np.concatenate([dS_xlo, dS_xhi, 
                                      dS_ylo, dS_yhi], axis=0)

            print(f"_compute_receiver_geometry(): walls subsampled at {stride_wall}x grid spacing ({stride_wall * d * 1e3:.3f} mm). RX points added: {len(wall_positions)} (x-walls: {len(pos_xlo) + len(pos_xhi)}, y-walls: {len(pos_ylo) + len(pos_yhi)}).")

            # Combine cap and wall positions, normals, and differential areas to form the full Huygens surface.
            self.rx_positions = np.concatenate([cap_positions, 
                                                wall_positions], axis=0)
            self.rx_normals = np.concatenate([cap_normals, 
                                              wall_normals], axis=0)
            self.dS_huygens = np.concatenate([cap_dS, wall_dS], axis=0)
            
            print(f"_compute_receiver_geometry(): closed Huygens surface (caps + walls) - {len(self.rx_positions)} RX points total ({len(cap_positions)} caps + {len(wall_positions)} walls, {len(self.rx_positions) / len(cap_positions):.2f}x the cap-only count).")

        else:

            # If the Huygens surface is not closed, only use the cap positions, normals, and differential areas.
            self.rx_positions = cap_positions
            self.rx_normals = cap_normals
            self.dS_huygens = cap_dS

        # If RX_MODE is 'DIRECT', override the Huygens surface with real receiver positions.
        if self.rx_mode == 'DIRECT':

            # Define the lower and upper bounds of the PML-free inner domain.
            lo = np.array([self.domain_inner[0], 
                           self.domain_inner[2], 
                           self.domain_inner[4]])
            hi = np.array([self.domain_inner[1], 
                           self.domain_inner[3], 
                           self.domain_inner[5]])
            
            # Check if any receiver is outside the PML-free inner domain.
            pts = np.asarray(self.rx_pos).T
            if np.any(pts < lo - 1e-9) or np.any(pts > hi + 1e-9):
                bad = np.where(np.any((pts < lo - 1e-9) | (pts > hi + 1e-9), axis=1))[0]
                raise ValueError(
                    f"EMSimulation._compute_receiver_geometry(): RX_MODE='DIRECT' needs every receiver inside the PML-free inner domain x {lo[0]*1e3:.2f}..{hi[0]*1e3:.2f}, y {lo[1]*1e3:.2f}..{hi[1]*1e3:.2f}, z {lo[2]*1e3:.2f}..{hi[2]*1e3:.2f} mm; {len(bad)} receiver(s) are outside (first: {(pts[bad[0]]*1e3).round(2)} mm).")

            self.rx_positions = pts.copy()
            self.rx_normals = np.zeros_like(pts)
            self.dS_huygens = np.ones(len(pts))

            print(f"_compute_receiver_geometry(): RX_MODE='DIRECT' - {len(pts)} real receivers (no Huygens surface).")

        # Estimate the memory required for storing receiver data in gprMax (6 components, float32)
        n_rx = len(self.rx_positions)
        est_gb = n_rx * 6 * self.n_time * 4 / (1000**3)
        print(f"_compute_receiver_geometry(): estimated receiver memory ~{est_gb:.2f} GB for {n_rx} RX points.")
        if est_gb > 4.0:
            print("_compute_receiver_geometry(): Warning - receiver memory estimate is still high. Consider a larger target_spacing (fewer samples per wavelength), a coarser wall sample density, or a shorter time window.")

    def _generate_simulation_input_file(self) -> None:
        """This function generates the gprMax input files based on the precomputed simulation parameters, including the source and receiver geometry, domain bounds, and other relevant settings.

        Args:
            None
        
        Returns:
            None

        Raises:
            ValueError: If any receiver is outside the PML-free inner domain.
        """

        # Create a folder based using current date and time. Ensure it is unique to avoid overwriting existing runs.
        path_inputfilename_list = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_root = Path(self.path_input_output_files)
        file_folder = file_root / timestamp
        n_dup = 1
        while file_folder.exists() and any(file_folder.iterdir()):
            n_dup += 1
            file_folder = file_root / f"{timestamp}_{n_dup}"
        file_folder.mkdir(parents=True, exist_ok=True)
        self.path_folder = file_folder

        # Compute the receiver geometry before determining the source array stride and other simulation parameters.
        self._compute_receiver_geometry()

        # Extract the number of grid points in the inner domain along the x and y axes.
        nx = int(self.n_x_domain_inner)
        ny = int(self.n_y_domain_inner)

        # Compute the source array stride based on the grating-lobe limitation. Then determine the number of source points along each axis.
        self.src_array_stride = self._compute_source_array_stride()
        n_src_x = len(range(0, nx, self.src_array_stride))
        n_src_y = len(range(0, ny, self.src_array_stride))
        print(f"_generate_simulation_input_file(): dipole source array {nx}x{ny} ({nx * ny} pts) -> {n_src_x}x{n_src_y} ({n_src_x * n_src_y} pts) after subsampling")

        # Get the z-position of the source plane within the inner domain.
        z_src = self.domain_inner[5]

        # Compute the belt geometry once, as it is identical for every TX position.
        p_0_belt = [0.0,
                    0.0,
                    self.l_z_grid / 2 - self.thickness_belt]
        p_1_belt = [self.l_x_grid,
                    self.l_y_grid,
                    self.l_z_grid / 2]
        p_belt = [p_0_belt, p_1_belt]

        # Compute the SUT geometry based on its shape.
        if self.shape_sut == "CYLINDER":

            radius_sut = self.sut['SIZE']['DIAMETER'] / 2
            thickness_sut = self.sut['SIZE']['THICKNESS']
            p_0_sut = [self.sut_cx,
                       self.sut_cy,
                       self.l_z_grid / 2]
            p_1_sut = [self.sut_cx,
                       self.sut_cy,
                       self.l_z_grid / 2 + thickness_sut]
            p_sut = [p_0_sut, 
                     p_1_sut, 
                     radius_sut]

            sut_geometry_line = (
                f"#cylinder: {p_0_sut[0]:.4f} {p_0_sut[1]:.4f} {p_0_sut[2]:.4f} {p_1_sut[0]:.4f} {p_1_sut[1]:.4f} {p_1_sut[2]:.4f} {radius_sut:.4f} sut\n\n")

            p_sut_bbox = [
                [self.sut_cx - radius_sut,
                 self.sut_cy - radius_sut,
                 self.l_z_grid / 2],
                [self.sut_cx + radius_sut,
                 self.sut_cy + radius_sut,
                 self.l_z_grid / 2 + thickness_sut]]

        elif self.shape_sut == "RECTANGULAR":

            length_x_sut, length_y_sut, thickness_sut = self.sut['SIZE']
            p_0_sut = [self.sut_cx - length_x_sut / 2,
                    self.sut_cy - length_y_sut / 2,
                    self.l_z_grid / 2]
            p_1_sut = [self.sut_cx + length_x_sut / 2,
                    self.sut_cy + length_y_sut / 2,
                    self.l_z_grid / 2 + thickness_sut]
            p_sut = [p_0_sut, 
                     p_1_sut]

            p_sut_bbox = [p_0_sut, 
                          p_1_sut]

            sut_geometry_line = (
                f"#box: {p_0_sut[0]:.4f} {p_0_sut[1]:.4f} {p_0_sut[2]:.4f} {p_1_sut[0]:.4f} {p_1_sut[1]:.4f} {p_1_sut[2]:.4f} sut\n")

        elif self.shape_sut == "FREEFORM":

            length_x_sut, length_y_sut, thickness_sut = self.sut['SIZE']
            p_0_sut = [self.sut_cx - length_x_sut / 2, 
                       self.sut_cy - length_y_sut / 2, 
                       self.l_z_grid / 2]
            p_1_sut = [self.sut_cx + length_x_sut / 2, 
                       self.sut_cy + length_y_sut / 2, 
                       self.l_z_grid / 2 + thickness_sut]
            p_sut = [p_0_sut, 
                     p_1_sut]
            p_sut_bbox = [p_0_sut, 
                          p_1_sut]

            sut_geometry_line = self._freeform_geometry_lines() if self.sut_mode == 'INGREDIENTS' else ""

        # Define geometry of the foreign object embedded in the SUT.
        fo_size = self.foreign_object['SIZE']
        fo_rel_pos = self.foreign_object['POSITION_RELATIVE']
        fo_center = [p_sut_bbox[0][axis] + fo_rel_pos[axis] *
                     (p_sut_bbox[1][axis] - p_sut_bbox[0][axis])
                     for axis in range(3)]
        p_0_fo = [fo_center[axis] - fo_size[axis] / 2
                  for axis in range(3)]
        p_1_fo = [fo_center[axis] + fo_size[axis] / 2
                  for axis in range(3)]

        # Determine the z-ranges for each SUT layer, used for defining geometry of layered SUT.
        layer_z_ranges = self._sut_layer_z_ranges()
        
        if self.sut_mode == 'INGREDIENTS' and not self.sut_is_layered:
            
            sut_material_lines = (
                f"#material: {self.debye_model_sut['EPS_INF']} {self.conductivity_sut} {self.permeability_relative_sut} {self.magnetic_loss_sut} sut\n\n"
                f"#add_dispersion_debye: 1 {self.debye_model_sut['DELTA_EPS']} {self.debye_model_sut['TAU']} sut\n\n")
            
            sut_geometry_block = sut_geometry_line

        elif self.sut_mode == 'INGREDIENTS':

            sut_material_lines = ""
            sut_geometry_block = ""
            for i, (z_lo, z_hi) in enumerate(layer_z_ranges):
                
                tag = f"sut_{i}"
                
                sut_material_lines += (
                    f"#material: {self.debye_model_sut_layers[i]['EPS_INF']} {self.conductivity_sut_layers[i]} {self.permeability_relative_sut_layers[i]} {self.magnetic_loss_sut_layers[i]} {tag}\n\n"
                    f"#add_dispersion_debye: 1 {self.debye_model_sut_layers[i]['DELTA_EPS']} {self.debye_model_sut_layers[i]['TAU']} {tag}\n\n")
                
                if self.shape_sut == "CYLINDER":
                    sut_geometry_block += (
                        f"#cylinder: {p_0_sut[0]:.4f} {p_0_sut[1]:.4f} {z_lo:.6f} "
                        f"{p_1_sut[0]:.4f} {p_1_sut[1]:.4f} {z_hi:.6f} {radius_sut:.4f} {tag}\n")
                else:
                    sut_geometry_block += (
                        f"#box: {p_0_sut[0]:.4f} {p_0_sut[1]:.4f} {z_lo:.6f} "
                        f"{p_1_sut[0]:.4f} {p_1_sut[1]:.4f} {z_hi:.6f} {tag}\n")
            
            sut_geometry_block += "\n"
        
        else:

            sut_material_lines = ""
            sut_geometry_block = ""

        # Generate discrete spherical inclusions for the SUT if applicable.
        if self.sut_inclusions is not None and self.sut_mode == 'INGREDIENTS':

            # Extract inclusion parameters.
            inc = self.sut_inclusions
            r_inc = inc['DIAMETER'] / 2
            cx, cy = self.sut_cx, self.sut_cy
            z_bottom = self.l_z_grid / 2

            # Check for overlaps between inclusions and the foreign-object box.
            for k, (dx_c, dy_c, z_c) in enumerate(inc['CENTERS']):
                c_abs = [cx + dx_c, cy + dy_c, z_bottom + z_c]
                gap = np.linalg.norm([max(p_0_fo[a] - c_abs[a], 0.0, c_abs[a] - p_1_fo[a])
                                      for a in range(3)])
                if gap < r_inc and not self.fo_may_overlap_inclusions:
                    raise ValueError(
                        f"EMSimulation._generate_simulation_input_file(): inclusion {k} overlaps the foreign-object box (gap {gap * 1e3:.3f} mm < radius {r_inc * 1e3:.3f} mm) - re-pack the inclusions with a larger foreign-object clearance.")
            
            sut_material_lines += (
                f"#material: {self.debye_model_inclusion['EPS_INF']} {self.conductivity_inclusion} {self.permeability_relative_inclusion} {self.magnetic_loss_inclusion} inclusion\n\n"
                f"#add_dispersion_debye: 1 {self.debye_model_inclusion['DELTA_EPS']} {self.debye_model_inclusion['TAU']} inclusion\n\n")

            # Add geometry for each inclusion center as a sphere.
            for dx_c, dy_c, z_c in inc['CENTERS']:
                sut_geometry_block += (
                    f"#sphere: {cx + dx_c:.6f} {cy + dy_c:.6f} {z_bottom + z_c:.6f} "
                    f"{r_inc:.6f} inclusion n\n")
            sut_geometry_block += "\n"

        # Add material lines for freeform SUT regions if in 'INGREDIENTS' mode.
        if self.sut_freeform is not None and self.sut_mode == 'INGREDIENTS':
            
            for r, mat in enumerate(self.region_materials):
                sut_material_lines += (
                    f"#material: {mat['debye']['EPS_INF']} {mat['sigma']} {mat['mu']} {mat['mu_loss']} region_{r}\n\n"
                    f"#add_dispersion_debye: 1 {mat['debye']['DELTA_EPS']} {mat['debye']['TAU']} region_{r}\n\n")
            
            for tag, mat in (('skin', self.skin_material), ('rim', self.rim_material), ('fo_shell', self.fo_shell_material)):
                if mat is None:
                    continue
                sut_material_lines += (
                    f"#material: {mat['debye']['EPS_INF']} {mat['sigma']} {mat['mu']} {mat['mu_loss']} {tag}\n\n"
                    f"#add_dispersion_debye: 1 {mat['debye']['DELTA_EPS']} {mat['debye']['TAU']} {tag}\n\n")

        # Add material lines for the foreign object.
        fo_material_lines = f"#material: {self.permittivity_relative_fo} {self.conductivity_fo} {self.permeability_relative_fo} {self.magnetic_loss_fo} fo\n\n"
        if self.fo_mode == 'SUT':
            fo_material_lines += (
                f"#add_dispersion_debye: 1 {self.debye_model_fo['DELTA_EPS']} {self.debye_model_fo['TAU']} fo\n\n")

        # FO geometry block: defines the spatial location of the foreign object. Normally one 'fo' box written after the SUT. In 'SUT' mode on a layered SUT the box is clipped per layer, where every piece gets that layer's material. The stack stays unperturbed.
        if self.fo_mode == 'SUT' and self.sut_is_layered:
            fo_geometry_block = ""
            for i, (z_lo, z_hi) in enumerate(layer_z_ranges):
                lo, hi = max(p_0_fo[2], z_lo), min(p_1_fo[2], z_hi)
                if hi - lo > 1e-9:
                    fo_geometry_block += (
                        f"#box: {p_0_fo[0]:.6f} {p_0_fo[1]:.6f} {lo:.6f} "
                        f"{p_1_fo[0]:.6f} {p_1_fo[1]:.6f} {hi:.6f} sut_{i}\n")
            fo_geometry_block += "\n"
        else:
            fo_geometry_block = (
                f"#box: {p_0_fo[0]:.4f} {p_0_fo[1]:.4f} {p_0_fo[2]:.4f} {p_1_fo[0]:.4f} {p_1_fo[1]:.4f} {p_1_fo[2]:.4f} fo\n\n")
            gap = float(self.foreign_object.get('AIR_GAP', 0.0))
            if gap > 0:
                fo_geometry_block = (
                    f"#box: {p_0_fo[0] - gap:.6f} {p_0_fo[1] - gap:.6f} {self.l_z_grid / 2:.6f} "
                    f"{p_1_fo[0] + gap:.6f} {p_1_fo[1] + gap:.6f} {p_1_fo[2]:.6f} free_space\n") + fo_geometry_block

        # Define the static header for the input file, including domain size, grid spacing, time window, PML cells, and material properties.
        static_header = (
            "#title: 3D Cookie Inspection on Conveyor Belt\n"
            f"#domain: {self.l_x_grid:.4f} {self.l_y_grid:.4f} {self.l_z_grid:.4f}\n"
            # Full float64 precision: gprMax derives dt and the iteration count from dx/dy/dz, and truncation made them drift
            # from self.dt/self.n_time. _sync_time_parameters_from_output() is a second safeguard.
            f"#dx_dy_dz: {self.dx:.15e} {self.dy:.15e} {self.dz:.15e}\n"
            f"#time_window: {self.time_window:.15e}\n\n"
            f"#pml_cells: {self.n_pml}\n\n"
            f"#material: {self.permittivity_relative_belt} {self.conductivity_belt} {self.permeability_relative_belt} {self.magnetic_loss_belt} belt\n"
            f"{sut_material_lines}"
            f"{fo_material_lines}"
            f"#box: {p_0_belt[0]:.4f} {p_0_belt[1]:.4f} {p_0_belt[2]:.4f} {p_1_belt[0]:.4f} {p_1_belt[1]:.4f} {p_1_belt[2]:.4f} belt\n"
            f"{sut_geometry_block}"
            f"{fo_geometry_block}"
            f"#waveform: gaussian {self.tx_amplitude} {self.f_center:.2e} impulse_source\n\n")

        # Each receiver (RX) is defined by its x, y, z coordinates. The format is "#rx: x y z\n".
        rx_lines = [f"#rx: {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in self.rx_positions]
        rx_block = "".join(rx_lines) + "\n"

        # Loop over each transmitter (TX) position to create an input file for each firing. The receiver block remains the same in all files.
        for i in range(self.n_pos):

            # Construct the full path for the input file corresponding to the current transmitter position.
            path_inputfilename = os.path.join(self.path_folder, 
                                              f'em_sim_{i}.in')
            path_inputfilename_list.append(path_inputfilename)

            # Define the plane-wave direction for the current transmitter position. This includes the rotation angles alpha and beta, and the time offset tau_off if applicable.
            alpha_line = f"alpha = math.radians({self.incident_wave_alpha[i]}) # Rotation around X-axis\n"
            beta_line = f"beta = math.radians({self.incident_wave_beta[i]})  # Rotation around Y-axis\n"
            tau_off_line = ""
            tau_line = "        tau = (kx*x + ky*y + kz*zsrc)/c\n"
            
            if self.plane_wave_angles == 'CORRECT':

                # Use the provided plane-wave angles directly 
                alpha_line = f"alpha = {float(self.incident_wave_alpha[i])!r} # polar angle theta from +z [rad], used as is (PLANE_WAVE_ANGLES='CORRECT')\n"
                beta_line = f"beta = {float(self.incident_wave_beta[i])!r}  # azimuth phi from +x in the xy-plane [rad], used as is\n"

                # Extract the wave vector components for the current transmitter position
                kx_i, ky_i, kz_i = (float(v) for v in self.k_tx[:, i])

                # Determine the last indices in the x and y directions for the source array based on the stride.
                i_last = max(range(0, int(self.n_x_domain_inner), 
                                   self.src_array_stride))
                j_last = max(range(0, int(self.n_y_domain_inner), 
                                   self.src_array_stride))

                # Determine the spatial extent of the source array in the x and y directions.
                xs = (self.domain_inner[0], 
                      self.domain_inner[0] + i_last * self.dx)
                ys = (self.domain_inner[2], 
                      self.domain_inner[2] + j_last * self.dy)

                # Compute the minimum time offset for the plane wave to reach the source array.
                tau_off_m = min(kx_i * x_ + ky_i * y_ for x_ in xs for y_ in ys) + kz_i * float(f"{z_src:.6f}")

                if self.plane_wave_tau_offset_m is None:
                    self.plane_wave_tau_offset_m = np.zeros(self.n_pos)
                self.plane_wave_tau_offset_m[i] = tau_off_m
                
                tau_off_line = f"tau_off = {tau_off_m!r}  # m: earliest dipole fires at t = 0\n"
                tau_line = "        tau = max(0.0, (kx*x + ky*y + kz*zsrc - tau_off)/c)\n"

            # Determine Huygens surface
            python_block = (
                "#python:\n"
                "import math\n"
                f"{alpha_line}"
                f"{beta_line}"
                "# Calculate the unit wave vector components based on angles\n"
                "kx = math.sin(alpha) * math.cos(beta)\n"
                "ky = math.sin(alpha) * math.sin(beta)\n"
                "kz = math.cos(alpha)\n"
                f"zsrc = {z_src:.6f}\n"
                f"c = {sp.constants.c:.6e}\n"
                f"{tau_off_line}"
                f"for i in range(0, {int(self.n_x_domain_inner)}, {self.src_array_stride}):\n"
                f"    for j in range(0, {int(self.n_y_domain_inner)}, {self.src_array_stride}):\n"
                f"        x = {self.domain_inner[0]} + i*{self.dx}\n"
                f"        y = {self.domain_inner[2]} + j*{self.dy}\n"
                f"{tau_line}"
                f"        pol = '{self.tx_polarization}'\n"
                f"        print(f'#hertzian_dipole: {{pol}} {{x:.4f}} {{y:.4f}} {{zsrc:.4f}} impulse_source {{tau:.3e}} {self.time_window:.3e}')\n"
                "#end_python:\n\n")

            # Write input file
            with open(path_inputfilename, "w") as f:
                f.write(static_header)
                f.write(python_block)
                f.write(rx_block)

        # Show the list of generated input files and a success message
        print(path_inputfilename_list)
        print(f"Successfully generated input file(s).")

        # Prepare the scene for visualization, including the belt, SUT, and FoV.
        p_sut_layers = None
        if self.sut_mode == 'INGREDIENTS' and self.sut_is_layered:
            p_sut_layers = []
            for z_lo, z_hi in layer_z_ranges:
                if self.shape_sut == "CYLINDER":
                    p_sut_layers.append([[p_0_sut[0], p_0_sut[1], z_lo],
                                         [p_1_sut[0], p_1_sut[1], z_hi], radius_sut])
                else:
                    p_sut_layers.append([[p_0_sut[0], p_0_sut[1], z_lo],
                                         [p_1_sut[0], p_1_sut[1], z_hi]])
        self._draw_scene(p_belt, p_sut if self.sut_mode == 'INGREDIENTS' else None, [p_0_fo, p_1_fo], aspect_ratio='-', p_sut_layers=p_sut_layers)
        self._draw_mimo_array_geometry()

        pass
