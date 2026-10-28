"""Figures of EMSimulation: scene and array drawings, response-axis styling (PlotsMixin) and the plot functions."""
import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.widgets import RadioButtons, Slider
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from EMSimulationHelpers import (_cookie_metric_short_name, 
                                 _cookie_short_name, 
                                 _fmt_mds, 
                                 _fmt_offset, 
                                 _freeform_columns, 
                                 _freeform_outline, 
                                 _freeform_region_outline, 
                                 _savefig)
from EMSimulationParameters import (_ANOVA_SIG_LEVEL, 
                                    _COMBINED_CONTRAST_COLOR_DARK, _COMBINED_CONTRAST_COLOR_LIGHT, 
                                    _COMBINED_RMSDPHASE_COLOR_DARK, 
                                    _COMBINED_RMSDPHASE_COLOR_LIGHT, 
                                    _COOKIE_COLORS, 
                                    _COOKIE_MARKERS, 
                                    _RESPONSE_LABELS, 
                                    _RESPONSE_SIGNED_METRICS, 
                                    _SUT_LAYER_COLORS, 
                                    ANOVA_CODE_TO_FACTOR)

# Constants (names, abbreviations, short forms, labels, colors, and plot settings)
_FF_STUDIES = ('Reflection', 'Transmission')
_FF_FACTOR_NAMES = {'diameter_mm': 'Diameter', 
                    'depth_mm': 'Depth', 
                    'position': 'Position', 
                    'material': 'Material',
                    'heterogeneity': 'Heterogeneity', 
                    'moisture': 'Moisture'}

_FF_FACTOR_ABBR = {'diameter_mm': 'Diam.', 
                   'depth_mm': 'Depth', 
                   'position': 'Pos.', 
                   'material': 'Mat.', 
                   'heterogeneity': 'Het.', 
                   'moisture': 'Moist.'}

_FF_FACTOR_SHORT = {'diameter_mm': 'diam', 
                    'depth_mm': 'depth', 
                    'position': 'pos', 
                    'material': 'mat',
                    'heterogeneity': 'het', 
                    'moisture': 'moist'}

_FF_LEVEL_LABELS = {'diameter_mm': {3.0: '3 mm', 7.0: '7 mm'}, 
                    'depth_mm': {0.5: '0.5 mm', 3.0: '3 mm'},
                    'position': {'Center': 'Center', 
                                 'Edge': 'Edge'}, 
                    'material': {'PVC': 'PVC', 'PP': 'PP'},
                    'heterogeneity': {'Homogeneous': 'Hom.', 
                                      'Porous/Layered': 'Por.'}, 
                    'moisture': {'Low': 'Low', 'High': 'High'}}

_FF_SHARE_LABELS = {**_FF_FACTOR_NAMES, 
                    'INT2': '2-way interactions', 
                    'INT3': '3-way interactions',
                    'RES': 'Residual (4-way and higher)'}

_FF_VARIANCE_RESPONSES = (('rms_dphase_max', 
                           'RMS $\\Delta$phase'), 
                           ('mean_dmag_max', 
                            'Mean $\\Delta|H|$'),
                          ('rms_dtau_max', 
                           'RMS $\\Delta$group delay'), 
                           ('frobenius_rel', 
                            'Relative Frobenius dev.'))

_FF_HALF_NORMAL_RESPONSES = (('rms_dphase_max', 
                              'RMS $\\Delta$phase (deg)'), 
                              ('mean_dmag_max', 
                               'Mean $\\Delta|H|$ (dB)'))

_FF_INK, _FF_SUB, _FF_GRID = '#0b0b0b', '#52514e', '#e3e2dd'

_FF_STUDY_COLORS = {'Reflection': '#2a78d6', 
                    'Transmission': '#eb6834'}

_FF_STUDY_MARKERS = {'Reflection': 'o', 
                     'Transmission': 's'}

_FF_RC = {'font.size': 7, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7, 
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5,
          'legend.fontsize': 6.5, 
          'axes.edgecolor': _FF_SUB, 
          'axes.labelcolor': _FF_INK, 
          'xtick.color': _FF_SUB,
          'ytick.color': _FF_SUB, 
          'font.family': 'DejaVu Sans', 
          'pdf.fonttype': 42, 
          'axes.linewidth': 0.6}


class PlotsMixin:
    """This mixing class provides plotting utilities. This includes scene and array drawings, response-axis styling, and other plot-related functions.
    """

    @staticmethod
    def _response_base_name(response: str) -> str:
        """This function returns the base name of a response metric by removing any trailing '_max' or '_mean' suffix.

        Args:
            response (str): 
                Response metric column name

        Returns:
            str: Base name of the response metric

        Raises:
            None
        """
        for suffix in ('_max', '_mean'):
            if response.endswith(suffix):
                return response[:-len(suffix)]
            
        return response

    @staticmethod
    def _response_label(response: str) -> str:
        """This function generates an axis label for a response metric column.

        Args:
            response (str): 
                Response metric column name

        Returns:
            str: Axis label for the response metric.

        Raises:
            None
        """
        base = PlotsMixin._response_base_name(response)
        label = _RESPONSE_LABELS.get(base, base)
        
        if response.endswith('_max'):
            return f"{label} (max across TX pos.)"
        if response.endswith('_mean'):
            return f"{label} (mean across TX pos.)"
        
        return label

    @staticmethod
    def _style_response_axis(ax, response: str, values) -> None:
        """This function applies a consistent y-axis styling for a response metric.

        Args:
            ax (matplotlib.axes.Axes): 
                Axis to style.
            response (str): 
                Response metric column name
            values (iterable of float): 
                All plotted values

        Returns:
            None
                
        Raises:
            None
        """
        base = PlotsMixin._response_base_name(response)
        values = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
            dtype=float)
        if values.size == 0:
            return
        if base in _RESPONSE_SIGNED_METRICS:
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            span = max(float(np.max(np.abs(values))) * 1.15, 1e-12)
            ax.set_ylim(-span, span)
        else:
            top = max(float(np.max(values)) * 1.15, 1e-12)
            ax.set_ylim(0, top)

    def _draw_scene(self, 
                    p_belt: list = None, 
                    p_sut: list = None, 
                    p_fo: list = None, 
                    aspect_ratio='cube', 
                    p_sut_layers: list = None) -> None:
            """This function draws the 3D scene and the 2D cross-section of domain, PML, belt, SUT, foreign object, TX/RX.

            Args:
                p_belt (list): 
                    First and second corner point of the belt (default: None).
                p_sut (list): 
                    SUT bounding box (default: None).
                p_fo (list): 
                    Foreign object's bounding box (default: None).
                aspect_ratio (str): 
                    Aspect ratio of the 3D plot (default: 'cube').
                p_sut_layers (list): 
                    For a layered SUT, one p_sut-style entry per layer.

            Returns:
                None

            Raises:
                None
            """

            def to_scalar(value):
                """This helper function returns floats.

                Args:
                    value: 
                        Scalar-like input.

                Returns:
                    float: The scalar value.

                Raises:
                    None
                """
                return float(np.asarray(value).reshape(-1)[0])

            def draw_box(ax, origin, size, color='blue', alpha=0.2):
                """This helper function draws a 3D box given its origin and size.

                Args:
                    ax (Matplotlib axes): 
                        Axes.
                    origin (tuple): 
                        Coordinates of the first corner point.
                    size (tuple): 
                        Side lengths.
                    color (str): 
                        Color (default: 'blue').
                    alpha (float): 
                        Alpha value (default: 0.2).

                Returns:
                    None

                Raises:
                    None
                """
                x, y, z = origin
                dx, dy, dz = size

                vertices = np.array([[x, y, z],
                                    [x+dx, y, z],
                                    [x+dx, y+dy, z],
                                    [x, y+dy, z],
                                    [x, y, z+dz],
                                    [x+dx, y, z+dz],
                                    [x+dx, y+dy, z+dz],
                                    [x, y+dy, z+dz]])

                # Order: bottom, top, front, back, right, left
                faces = [[vertices[j] for j in [0,1,2,3]],
                        [vertices[j] for j in [4,5,6,7]],
                        [vertices[j] for j in [0,1,5,4]],
                        [vertices[j] for j in [2,3,7,6]],
                        [vertices[j] for j in [1,2,6,5]],
                        [vertices[j] for j in [0,3,7,4]]]

                ax.add_collection3d(Poly3DCollection(faces,
                                                    facecolors=color,edgecolors='k',
                                                    alpha=alpha))

            def draw_cylinder(ax, p1, p2, radius, color='blue',alpha=0.2, resolution=50):
                """This helper function draws a cylinder between two points.

                Args:
                    ax: 
                        Matplotlib 3D axis.
                    p1: 
                        (x1, y1, z1) center of the first circular face.
                    p2: 
                        (x2, y2, z2) center of the second circular face.
                    radius: 
                        Cylinder radius.
                    color: 
                        Surface color.
                    alpha: 
                        Transparency.
                    resolution: 
                        Number of angular samples.

                Returns:
                    None

                Raises:
                    None
                """
                p1 = np.asarray(p1, dtype=float)
                p2 = np.asarray(p2, dtype=float)

                axis = p2 - p1
                length = np.linalg.norm(axis)
                if length == 0:
                    raise ValueError("p1 and p2 must be different.")
                axis = axis / length

                if np.allclose(axis, [0, 0, 1]):
                    not_axis = np.array([1, 0, 0])
                else:
                    not_axis = np.array([0, 0, 1])

                u = np.cross(axis, not_axis)
                u = u / np.linalg.norm(u)
                v = np.cross(axis, u)
                v = v / np.linalg.norm(v)

                theta = np.linspace(0, 2*np.pi, resolution)
                z = np.linspace(0, length, 2)
                Theta, Z = np.meshgrid(theta, z)

                X = (p1[0] + axis[0]*Z + 
                     radius*np.cos(Theta)*u[0] + radius*np.sin(Theta)*v[0])
                Y = (p1[1] + axis[1]*Z + 
                     radius*np.cos(Theta)*u[1] + radius*np.sin(Theta)*v[1])
                Zc = (p1[2] + axis[2]*Z + 
                      radius*np.cos(Theta)*u[2] + radius*np.sin(Theta)*v[2])

                ax.plot_surface(X, Y, Zc, color=color, alpha=alpha, linewidth=0)

                r = np.linspace(0, radius, resolution)
                Theta2, R = np.meshgrid(theta, r)
                for center in [p1, p2]:
                    Xcap = (center[0] + 
                            R*np.cos(Theta2)*u[0] + R*np.sin(Theta2)*v[0])
                    Ycap = (center[1] + 
                            R*np.cos(Theta2)*u[1] + R*np.sin(Theta2)*v[1])
                    Zcap = (center[2] + 
                            R*np.cos(Theta2)*u[2] + R*np.sin(Theta2)*v[2])
                    ax.plot_surface(Xcap, Ycap, Zcap, color=color, alpha=alpha, linewidth=0)

            def configure_axes_3d(ax, bounds_min, bounds_max, aspect_ratio, title):
                """This helper function sets the limits, aspect ratio, labels, title, and view angle for a 3D plot.

                Args:
                    ax (Matplotlib axes): 
                        3D axes to configure.
                    bounds_min (array-like): 
                        Lower (x, y, z) bounds.
                    bounds_max (array-like): 
                        Upper (x, y, z) bounds.
                    aspect_ratio (str): 
                        'cube' for equal box aspect, otherwise scaled to bounds.
                    title (str): 
                        Axes title.

                Returns:
                    None

                Raises:
                    None
                """
                ax.set_xlim(bounds_min[0], bounds_max[0])
                ax.set_ylim(bounds_min[1], bounds_max[1])
                ax.set_zlim(bounds_min[2], bounds_max[2])
                
                if aspect_ratio == 'cube':
                    ax.set_box_aspect((1, 1, 1))
                else:
                    ax.set_box_aspect(bounds_max - bounds_min)
                
                ax.set_xlabel('X / m')
                ax.set_ylabel('Y / m')
                ax.set_zlabel('Z / m')
                ax.set_title(title)
                ax.view_init(elev=25, azim=-130)

            def box_slice(p0, p1, cut_axis, disp_axes, cut_val):
                """This helper function determines the 2D cross-section of an axis-aligned box with a plane normal to the specified axis at the given coordinate.

                Args:
                    p0 (array-like): 
                        First corner point (x, y, z).
                    p1 (array-like): 
                        Opposite corner point (x, y, z).
                    cut_axis (int): 
                        Axis (0=X, 1=Y, 2=Z) the cutting plane is normal to.
                    disp_axes (tuple): 
                        Indices (d0, d1) of the axes shown on the 2D plot.
                    cut_val (float): 
                        Coordinate of the cutting plane along `cut_axis`.

                Returns:
                    tuple or None: (d0_min, d0_max, d1_min, d1_max) if the plane intersects the box, else None.

                Raises:
                    None
                """
                p0 = np.asarray(p0, dtype=float)
                p1 = np.asarray(p1, dtype=float)
                lo, hi = sorted((p0[cut_axis], p1[cut_axis]))
                
                if cut_val < lo or cut_val > hi:
                    return None
                
                d0, d1 = disp_axes
                d0_lo, d0_hi = sorted((p0[d0], p1[d0]))
                d1_lo, d1_hi = sorted((p0[d1], p1[d1]))
                return d0_lo, d0_hi, d1_lo, d1_hi

            def cylinder_slice(p1, p2, radius, cut_axis, disp_axes, cut_val, resolution=200):
                """This helper function computes the cross-section of a cylinder with a plane normal to cut axis at cut val.

                Args:
                    p1 (array-like): 
                        Center of the first circular face.
                    p2 (array-like): 
                        Center of the second circular face.
                    radius (float): 
                        Cylinder radius.
                    cut_axis (int): 
                        Axis (0=X, 1=Y, 2=Z) the cutting plane is normal to.
                    disp_axes (tuple): 
                        Indices (d0, d1) of the axes shown on the 2D plot.
                    cut_val (float): 
                        Coordinate of the cutting plane along cut axis.
                    resolution (int): 
                        Angular samples for the general-orientation case (default: 200).

                Returns:
                    list or None: (plot_x, plot_y) points outlining the slice, or None if the plane misses the cylinder.

                Raises:
                    None
                """
                p1 = np.asarray(p1, dtype=float)
                p2 = np.asarray(p2, dtype=float)
                axis = p2 - p1
                length = np.linalg.norm(axis)
                
                if length == 0:
                    return None
                axis = axis / length

                not_axis = np.array([1, 0, 0]) if np.allclose(axis, [0, 0, 1]) else np.array([0, 0, 1])
                u = np.cross(axis, not_axis); u = u / np.linalg.norm(u)
                v = np.cross(axis, u); v = v / np.linalg.norm(v)

                c = cut_axis
                d0, d1 = disp_axes

                if abs(axis[c]) < 1e-9:
                    
                    amp = radius * np.hypot(u[c], v[c])
                    k = cut_val - p1[c]
                    if amp < 1e-12 or abs(k) > amp:
                        return None
                    
                    phase = np.arctan2(v[c], u[c])
                    dtheta = np.arccos(np.clip(k / amp, -1.0, 1.0))
                    corners = []
                    
                    for s_end in (0.0, length):
                        for theta in (phase + dtheta, phase - dtheta):
                            pt = p1 + s_end * axis + radius * np.cos(theta) * u + radius * np.sin(theta) * v
                            corners.append((pt[d0], pt[d1]))
                    
                    xs = [pt[0] for pt in corners]
                    ys = [pt[1] for pt in corners]
                    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
                    
                    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

                # General orientation
                thetas = np.linspace(0.0, 2 * np.pi, resolution)
                s_vals = (cut_val - p1[c] - radius * np.cos(thetas) * u[c] - radius * np.sin(thetas) * v[c]) / axis[c]
                valid = (s_vals >= 0.0) & (s_vals <= length)
                
                if not np.any(valid):
                    return None
                
                pts = (p1[None, :] + s_vals[valid, None] * axis[None, :]
                    + radius * np.cos(thetas[valid])[:, None] * u[None, :]
                    + radius * np.sin(thetas[valid])[:, None] * v[None, :])
                
                return list(zip(pts[:, d0], pts[:, d1]))

            def draw_rect(target_ax, x_range, y_range, color, alpha):
                """This helper function draws a filled rectangle patch on a 2D axes.

                Args:
                    target_ax (Matplotlib axes): 
                        2D axes to draw on.
                    x_range (tuple): 
                        (x_min, x_max).
                    y_range (tuple): 
                        (y_min, y_max).
                    color (str): 
                        Face color.
                    alpha (float): 
                        Alpha value.

                Returns:
                    None

                Raises:
                    None
                """
                x_min, x_max = x_range
                y_min, y_max = y_range
                target_ax.add_patch(Rectangle((x_min, y_min), 
                                              x_max - x_min, y_max - y_min,facecolor=color, edgecolor='k', linewidth=0.5, alpha=alpha))

            # Get grid dimensions and domain center
            l_x_grid = to_scalar(self.l_x_grid)
            l_y_grid = to_scalar(self.l_y_grid)
            l_z_grid = to_scalar(self.l_z_grid)
            pml_x = to_scalar(self.pml_thickness[0])
            pml_y = to_scalar(self.pml_thickness[1])
            pml_z = to_scalar(self.pml_thickness[2])
            domain_center = [to_scalar(self.domain_center[0]),
                             to_scalar(self.domain_center[1]),
                             to_scalar(self.domain_center[2])]

            # Set plane configuration for cross section plotting
            axis_label = ('X', 'Y', 'Z')
            grid_len = (l_x_grid, l_y_grid, l_z_grid)
            plane_config = {
                'XY': {'cut_axis': 2, 'disp_axes': (0, 1)},
                'YZ': {'cut_axis': 0, 'disp_axes': (1, 2)},
                'XZ': {'cut_axis': 1, 'disp_axes': (0, 2)},}

            # Figure setup
            fig = plt.figure(figsize=(14, 6))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax_domain = fig.add_subplot(1, 2, 2)

            # Draw total domain and center
            draw_box(ax, 
                     origin=(0.0, 0.0, 0.0), 
                     size=(l_x_grid, l_y_grid, l_z_grid),
                     color='lightblue', alpha=0.1)
            ax.scatter(*domain_center, 
                       color='blue', 
                       marker='o', s=40,
                       depthshade=False, 
                       label='Domain Center')

            # Draw PML
            pml_slabs = [
                ((0.0, 0.0, 0.0), (l_x_grid, pml_y, l_z_grid)),
                ((0.0, l_y_grid - pml_y, 0.0), (l_x_grid, pml_y, l_z_grid)),
                ((0.0, 0.0, 0.0), (pml_x, l_y_grid, l_z_grid)),
                ((l_x_grid - pml_x, 0.0, 0.0), (pml_x, l_y_grid, l_z_grid)),
                ((0.0, 0.0, 0.0), (l_x_grid, l_y_grid, pml_z)),
                ((0.0, 0.0, l_z_grid - pml_z), (l_x_grid, l_y_grid, pml_z)),]
            
            for origin, size in pml_slabs:
                draw_box(ax, 
                         origin=origin, 
                         size=size, 
                         color='lightblue', 
                         alpha=0.2)

            # Draw belt
            p0_belt, p1_belt = np.asarray(p_belt[0]), np.asarray(p_belt[1])
            draw_box(ax, 
                     origin=tuple(p0_belt), 
                     size=tuple(p1_belt - p0_belt), 
                     color='blue', 
                     alpha=0.6)

            # Draw SUT
            if p_sut is not None:
                for k_layer, p_sut_k in enumerate(p_sut_layers if p_sut_layers else [p_sut]):
                    
                    layer_color = _SUT_LAYER_COLORS[k_layer % len(_SUT_LAYER_COLORS)] if p_sut_layers else None
                    
                    if self.shape_sut == "CYLINDER":
                        draw_cylinder(ax, p1=p_sut_k[0], p2=p_sut_k[1], radius=p_sut_k[2], color=layer_color or 'brown', alpha=0.8)
                    elif self.shape_sut in ("RECTANGULAR", "FREEFORM"):
                        p0_sut, p1_sut = np.asarray(p_sut_k[0]), np.asarray(p_sut_k[1])
                        draw_box(ax, origin=tuple(p0_sut), size=tuple(p1_sut - p0_sut), color=layer_color or 'blue', alpha=0.8)

            # Draw FO
            if p_fo is not None:
                p0_fo, p1_fo = np.asarray(p_fo[0]), np.asarray(p_fo[1])
                draw_box(ax, origin=tuple(p0_fo), size=tuple(p1_fo - p0_fo), color='red', alpha=0.9)

            # Plot TX and RX antenna positions (self.tx_pos/self.rx_pos are the
            # (3, n_pos) absolute-domain-coordinate arrays.
            have_tx = hasattr(self, 'tx_pos') and self.tx_pos is not None
            have_rx = hasattr(self, 'rx_pos') and self.rx_pos is not None
            if have_tx:
                ax.scatter(self.tx_pos[0], self.tx_pos[1], self.tx_pos[2],
                        color='red', marker='v', s=40, depthshade=False, label='TX')
            if have_rx:
                ax.scatter(self.rx_pos[0], self.rx_pos[1], self.rx_pos[2],
                        color='blue', marker='^', s=40, depthshade=False, label='RX')
            if have_tx or have_rx:
                ax.legend(loc='upper right')

            bounds_min = np.array([0.0, 0.0, 0.0])
            bounds_max = np.array([l_x_grid, l_y_grid, l_z_grid])
            
            for pos in (self.tx_pos if have_tx else None, self.rx_pos if have_rx else None):
                if pos is not None:
                    bounds_min = np.minimum(bounds_min, pos.min(axis=1))
                    bounds_max = np.maximum(bounds_max, pos.max(axis=1))
            
            margin = 0.05 * np.maximum(bounds_max - bounds_min, 1e-12)
            bounds_min -= margin
            bounds_max += margin

            configure_axes_3d(ax, bounds_min, bounds_max, aspect_ratio, 'Total simulation scene')

            state = {'plane': 'XY', 'slider': None}

            def update_cross_section(cut_val):
                """Define helper function for updating the 2D cross-section view based on the selected plane and cutting coordinate.

                Args:
                    cut_val (float): 
                        Coordinate of the cutting plane along the current normal axis.

                Returns:
                    None

                Raises:
                    None
                """
                cfg = plane_config[state['plane']]
                cut_axis, disp_axes = cfg['cut_axis'], cfg['disp_axes']
                d0, d1 = disp_axes
                lbl0, lbl1 = axis_label[d0], axis_label[d1]

                ax_domain.cla()

                # Draw domain outline
                draw_rect(ax_domain, 
                          (0.0, grid_len[d0]), 
                          (0.0, grid_len[d1]), 
                          color='lightblue', 
                          alpha=0.15)

                # Draw PML slabs
                for origin, size in pml_slabs:
                    p0 = origin
                    p1 = (origin[0] + size[0], 
                          origin[1] + size[1], 
                          origin[2] + size[2])
                    sl = box_slice(p0, p1, cut_axis, disp_axes, cut_val)
                    if sl is not None:
                        x_min, x_max, y_min, y_max = sl
                        draw_rect(ax_domain, (x_min, x_max), (y_min, y_max), color='lightblue', alpha=0.3)

                # Draw belt
                sl = box_slice(p_belt[0], 
                               p_belt[1], 
                               cut_axis, 
                               disp_axes, 
                               cut_val)
                if sl is not None:
                    x_min, x_max, y_min, y_max = sl
                    draw_rect(ax_domain, 
                              (x_min, x_max), 
                              (y_min, y_max), 
                              color='blue', 
                              alpha=0.6)

                # Draw SUT
                if p_sut is not None:
                    for k_layer, p_sut_k in enumerate(p_sut_layers if p_sut_layers else [p_sut]):
                        
                        layer_color = _SUT_LAYER_COLORS[k_layer % len(_SUT_LAYER_COLORS)] if p_sut_layers else None
                        
                        if self.shape_sut == "CYLINDER":
                            poly = cylinder_slice(p_sut_k[0], p_sut_k[1], p_sut_k[2], cut_axis, disp_axes, cut_val)
                            if poly is not None:
                                xs, ys = zip(*poly)
                                ax_domain.fill(xs, ys, 
                                               color=layer_color or 'brown', alpha=0.8, edgecolor='k', linewidth=0.5)
                        elif self.shape_sut in ("RECTANGULAR", "FREEFORM"):
                            sl = box_slice(p_sut_k[0], 
                                           p_sut_k[1], 
                                           cut_axis, 
                                           disp_axes, 
                                           cut_val)
                            if sl is not None:
                                x_min, x_max, y_min, y_max = sl
                                draw_rect(ax_domain, 
                                          (x_min, x_max), 
                                          (y_min, y_max), 
                                          color=layer_color or 'blue', 
                                          alpha=0.8)

                # Draw foreign object
                if p_fo is not None:
                    sl = box_slice(p_fo[0], p_fo[1], cut_axis, disp_axes, cut_val)
                    if sl is not None:
                        x_min, x_max, y_min, y_max = sl
                        draw_rect(ax_domain, (x_min, x_max), (y_min, y_max), color='red', alpha=0.9)

                ax_domain.set_xlim(0.0, grid_len[d0])
                ax_domain.set_ylim(0.0, grid_len[d1])
                ax_domain.set_aspect('equal', adjustable='box')
                ax_domain.set_xlabel(f'{lbl0} / m')
                ax_domain.set_ylabel(f'{lbl1} / m')
                ax_domain.set_title(f'{state["plane"]} cross-section at {axis_label[cut_axis]} = {cut_val:.3f} m')
                fig.canvas.draw_idle()

            def rebuild_slider(new_plane):
                """This helper function redraws the cross-section for the new cutting plane based on a slider.

                Args:
                    new_plane (str): 
                        One of 'XY', 'YZ', 'XZ'.

                Returns:
                    None

                Raises:
                    None
                """
                state['plane'] = new_plane
                cut_axis = plane_config[new_plane]['cut_axis']
                new_max = grid_len[cut_axis]
                new_val = min(max(domain_center[cut_axis], 0.0), new_max)

                slider_ax.cla()
                state['slider'] = Slider(slider_ax, f'{axis_label[cut_axis]}-slice / m', 0.0, new_max, valinit=new_val)
                state['slider'].on_changed(update_cross_section)

                update_cross_section(new_val)

            # Put space below the figure and place the slider
            fig.subplots_adjust(bottom=0.22)
            pos = ax_domain.get_position()
            slider_ax = fig.add_axes([pos.x0, 0.06, pos.width, 0.03])

            # Place the plane-selector radio buttons
            radio_ax = fig.add_axes([0.02, 0.03, 0.09, 0.15])
            radio = RadioButtons(radio_ax, ('XY', 'YZ', 'XZ'), active=2)
            radio.on_clicked(rebuild_slider)

            rebuild_slider('XZ')

            self._domain_slice_slider = state['slider']
            self._domain_slice_radio = radio

            plt.show(block=False)

    def _draw_mimo_array_geometry(self) -> None:
        """This function draws the MIMO antenna array geometry.

        Args:
            None
        
        Returns:
            None

        Raises:
            None
        """
        if self.n_rx > 32 or self.n_tx > 32:
            print(f"_draw_mimo_array_geometry(): {self.n_tx} TX / {self.n_rx} RX - this x-axis diagram is for small arrays only so skip.")
            return
        
        have_tx = hasattr(self, 'tx_pos') and self.tx_pos is not None
        have_rx = hasattr(self, 'rx_pos') and self.rx_pos is not None
        
        if not (have_tx and have_rx):
            print("_draw_mimo_array_geometry(): self.tx_pos/self.rx_pos not set yet, therefore jsut skip.")
            return

        center_x = float(np.asarray(self.domain_center).reshape(-1)[0])
        tx_x_mm = (self.tx_pos[0, :] - center_x) * 1e3
        rx_x_mm = (self.rx_pos[0, :] - center_x) * 1e3
        n_tx, n_rx = self.n_tx, self.n_rx

        # Get smallest wavelength and spacing
        lam_min = sp.constants.c / self.frequency_range[-1]
        d_half_lambda_mm = (lam_min / 2) * 1e3

        d_tx_mm = float(np.abs(np.diff(tx_x_mm)).mean()) if n_tx > 1 else float('nan')
        d_rx_mm = float(np.abs(np.diff(rx_x_mm)).mean()) if n_rx > 1 else float('nan')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Set real TX/RX antenna positions (left subplot)
        sample_half_width_mm = None
        if self.shape_sut == 'CYLINDER':
            sample_half_width_mm = self.sut['SIZE']['DIAMETER'] * 1e3 / 2
        elif self.shape_sut in ('RECTANGULAR', 'FREEFORM'):
            sample_half_width_mm = self.sut['SIZE'][0] * 1e3 / 2
        if sample_half_width_mm:
            ax1.axvspan(-sample_half_width_mm, 
                        sample_half_width_mm, 
                        facecolor='#d9c9a3',
                        edgecolor='#8a7248', 
                        linewidth=1, 
                        alpha=0.5, 
                        zorder=0,
                        label='Sample (SUT) x-extent')
        ax1.axvline(0, color='gray', linestyle=':', linewidth=1, zorder=0)
        ax1.text(0, -31, 
                 'center axis', 
                 ha='center', 
                 va='bottom', 
                 fontsize=9, 
                 color='gray')

        ax1.scatter(tx_x_mm, 
                    np.zeros(n_tx), 
                    s=140, marker='v', color='red',
                    edgecolor='k', zorder=3, label=f'TX ({n_tx})')

        # Set vertical offset to avoid labels overlapping each other.
        for i, x in enumerate(tx_x_mm):
            y_off = 14 + (i % 2) * 14
            ax1.annotate(f'TX{i}', (x, 0), xytext=(0, y_off), textcoords='offset points',
                         ha='center', fontsize=8, color='red')
        ax1.scatter(rx_x_mm, 
                    np.zeros(n_rx), 
                    s=140, 
                    marker='^', 
                    color='blue',
                    edgecolor='k', 
                    zorder=3, 
                    label=f'RX ({n_rx})')
        
        for i, x in enumerate(rx_x_mm):
            y_off = -20 - (i % 2) * 14
            ax1.annotate(f'RX{i}', 
                         (x, 0), 
                         xytext=(0, y_off), 
                         textcoords='offset points',
                         ha='center', 
                         fontsize=8, 
                         color='blue')

        # Annotate TX antennas
        if n_tx > 1:
            tx_sorted = np.sort(tx_x_mm)
            y_dim = 12
            for x0, x1 in zip(tx_sorted[:-1], tx_sorted[1:]):
                ax1.annotate('', xy=(x1, y_dim), xytext=(x0, y_dim),
                             arrowprops=dict(arrowstyle='<->', color='red', lw=1))
            ax1.text(float(tx_x_mm.mean()), y_dim + 1.5,
                     f"d_TX = {d_tx_mm:.4f} mm  (target N_RX x lambda/2 = "
                     f"{n_rx}x{d_half_lambda_mm:.4f} = {n_rx * d_half_lambda_mm:.4f} mm)",
                     ha='center', fontsize=8, color='red')

        # Annotate RX antennas
        if n_rx > 1:
            rx_sorted = np.sort(rx_x_mm)
            y_dim = -14
            for x0, x1 in zip(rx_sorted[:-1], rx_sorted[1:]):
                ax1.annotate('', xy=(x1, y_dim), xytext=(x0, y_dim),
                             arrowprops=dict(arrowstyle='<->', color='blue', lw=1))
            ax1.text(float(rx_x_mm.mean()), y_dim - 2.5,
                     f"d_RX = {d_rx_mm:.4f} mm  (target lambda/2 @ "
                     f"{self.frequency_range[-1] / 1e9:.0f} GHz = {d_half_lambda_mm:.4f} mm)",
                     ha='center', fontsize=8, color='blue')

        ax1.set_title('Real TX/RX antenna positions')
        ax1.set_xlabel('x offset from sample center / mm')
        ax1.set_yticks([])
        margin = 5.0
        ax1.set_xlim(float(tx_x_mm.min()) - 
                     margin, float(rx_x_mm.max()) + margin)
        ax1.set_ylim(-32, 32)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax1.grid(axis='x', alpha=0.25)

        # Construct and plot the virtual array of the MIMO (right subplot)
        cmap = plt.get_cmap('viridis')
        for i in range(n_tx):
            v_row = tx_x_mm[i] + rx_x_mm
            color = cmap(i / max(n_tx - 1, 1))
            ax2.plot(v_row, 
                     np.full(n_rx, n_tx - i), 
                     color=color, 
                     linewidth=1,
                     alpha=0.5, 
                     zorder=2)
            ax2.scatter(v_row, 
                        np.full(n_rx, n_tx - i), 
                        s=60, 
                        marker='o',
                        color=color, 
                        edgecolor='k', 
                        linewidth=0.5, 
                        zorder=3,
                        label=f'TX{i} + RX0..{n_rx - 1}')

        v_all = np.sort((tx_x_mm[:, None] + rx_x_mm[None, :]).ravel())
        y_virtual = -1.5
        ax2.scatter(v_all, 
                    np.full(v_all.size, y_virtual), 
                    s=70, 
                    marker='|',
                    color='black', 
                    zorder=3, 
                    label=f'Combined virtual array ({v_all.size} elements)')
        d_virtual_mm = np.diff(v_all)
        ax2.text(float(v_all.mean()), 
                 y_virtual - 1.0,
                 f"virtual spacing = {d_virtual_mm.mean():.4f} +/- {d_virtual_mm.std():.1e} mm (target lambda/2 = {d_half_lambda_mm:.4f} mm)",
                 ha='center', 
                 fontsize=8, 
                 color='black')

        ax2.set_title(f'MIMO virtual array ({n_tx}x{n_rx} = {n_tx * n_rx} elements)\noffset-sum construction TX_i + RX_j - design/spacing check, see docstring')
        ax2.set_xlabel('virtual element x offset from sample center / mm')
        ax2.set_yticks(list(range(1, n_tx + 1)) + [int(y_virtual)])
        ax2.set_yticklabels([f'TX{n_tx - 1 - k}' for k in range(n_tx)] + ['combined'])
        ax2.set_ylim(y_virtual - 2.5, n_tx + 1.6)
        ax2.legend(loc='upper left', 
                   bbox_to_anchor=(1.01, 1.0), 
                   fontsize=7, 
                   framealpha=0.9)
        ax2.grid(axis='x', 
                 alpha=0.25)

        fig.suptitle('MIMO antenna array geometry')
        fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
        
        plt.show(block=False)

# region FO size study

def _plot_cookie_mds_side_comparison(mds_all: 'pd.DataFrame', 
                                     response_metrics: list, 
                                     out_path, prefix: str = 'mds_mm_',
                                     ylabel: str = 'Smallest detectable size / mm') -> None:
    """This function generates side-by-side comparison plots for the smallest detectable size (MDS) of foreign objects per product for comparing reflection and transmission sides.

    Args:
        mds_all (pd.DataFrame): 
            Rows per (rx_side, cookie) with <prefix><m> for every response metric m and 'combined'.
        response_metrics (list): 
            Metric names (panels), 'combined' is appended.
        out_path (str or Path): 
            PNG destination.
        prefix (str): 
            Column prefix ('mds_mm_' for the size study, 'max_offset_mm_' for the position study).
        ylabel (str): 
            y-axis label.
    """
    # Prepare
    sides = [sd for sd in ('reflection', 'transmission') if sd in set(mds_all['rx_side'])]
    cookies = list(dict.fromkeys(mds_all['cookie']))
    keys = list(response_metrics) + ['combined']
    colors = {'reflection': '#3b6ea5', 'transmission': '#d9822b'}
    cap = lambda text: text[:1].upper() + text[1:]
    
    fig, axes = plt.subplots(1, len(keys), 
                             figsize=(4.2 * len(keys), 3.8), squeeze=False)

    # Plot each response metric side by side for each cookie and side
    width = 0.8 / max(len(sides), 1)
    for ax, key in zip(axes[0], keys):

        col = f'{prefix}{key}'
        top = np.nanmax(pd.to_numeric(mds_all[col], errors='coerce').to_numpy(dtype=float)) if mds_all[col].notna().any() else 1.0
        
        # Plot bars for each side and cookie combination
        for si, sd in enumerate(sides):
            for ci, name in enumerate(cookies):

                # Get the row
                row = mds_all[(mds_all['rx_side'] == sd) & (mds_all['cookie'] == name)]
                v = float(row[col].iloc[0]) if len(row) and pd.notna(row[col].iloc[0]) else np.nan
                
                x = ci + (si - (len(sides) - 1) / 2) * width
                
                # Draw the bar for the current side and cookie combination
                if np.isfinite(v):
                    ax.bar(x, v, 
                           width * 0.95, 
                           color=colors[sd], 
                           label=cap(sd) if ci == 0 else None)
                    ax.text(x, v, f'{v:g}', ha='center', va='bottom', fontsize=7)
                else:
                    ax.bar(x, 0.06 * top, 
                           width * 0.95, 
                           color='none', 
                           edgecolor=colors[sd], 
                           hatch='//', 
                           label=cap(sd) if ci == 0 else None)
                    ax.text(x, 0.06 * top, 'n.d.', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(range(len(cookies)))
        ax.set_xticklabels([c.split(' (')[0] for c in cookies], 
                           rotation=30, 
                           ha='right', 
                           fontsize=8)
        ax.set_ylabel(cap(ylabel))
        ax.set_title(cap(_cookie_metric_short_name(key) if key != 'combined' else 'combined'), fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    
    if handles:
        fig.legend(dict(zip(labels, handles)).values(), 
                   dict(zip(labels, handles)).keys(), 
                   loc='upper center', 
                   ncol=len(sides), 
                   frameon=False)

    # Plot and save the figure
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _savefig(fig, out_path, dpi=160)
    plt.close(fig)


def _plot_cookie_detectability(results_df: 'pd.DataFrame', 
                               mds_df: 'pd.DataFrame',
                               response_metrics: list, 
                               floors: dict, 
                               out_path,
                               detection_rule: str = None) -> None:
    """This function generates a plot showing the detectability of cookies based on foreign-object size and response metrics (one curve for every product, one panel for each response metric against log-log FO size).

    Args:
        results_df (pd.DataFrame): 
            Columns cookie, fo_size_mm, plus one column per response metric.
        mds_df (pd.DataFrame): 
            Columns cookie, thickness_mm and, per response metric m and for 'combined'
        response_metrics (list): 
            Metric columns to plot, one panel each.
        floors (dict): 
            Detection threshold (None: no line).
        out_path (str or Path): 
            PNG destination.
        detection_rule (str): 
            Rule behind the table's limits ('conservative' or 'envelope'). None omits the note.
    """
    ink, muted, grid, surface = '#1a1a19', '#6b6a63', '#e4e3dc', '#fcfcfb'
    
    n_panels = len(response_metrics)
    fig = plt.figure(figsize=(9.8, 4.6 * n_panels + 2.6), facecolor=surface)
    
    gs = fig.add_gridspec(n_panels + 1, 1,
                          height_ratios=[3.2] * n_panels + [0.5 + 0.34 * len(mds_df)],
                          hspace=0.32)
    axes = [fig.add_subplot(gs[i], facecolor=surface) for i in range(n_panels)]
    ax_tab = fig.add_subplot(gs[n_panels], facecolor=surface)
    ax_tab.axis('off')
    
    if detection_rule:
        ax_tab.set_title(f'smallest detectable size, {detection_rule} rule (floors as drawn)', fontsize=8, color=muted, loc='left')

    # Loop over each response metric and create a panel for it.
    cookies = list(dict.fromkeys(results_df['cookie']))
    ticks = [10, 8, 6, 5, 4, 3, 2, 1.5, 1]
    for p, (ax, metric) in enumerate(zip(axes, response_metrics)):
        
        floor = floors.get(metric)
        end_points = []
        
        for i, cookie in enumerate(cookies):

            color = _COOKIE_COLORS[i % len(_COOKIE_COLORS)]
            marker = _COOKIE_MARKERS[i % len(_COOKIE_MARKERS)]
            sub = results_df[results_df['cookie'] == cookie].sort_values(
                'fo_size_mm', ascending=False)
            
            ax.plot(sub['fo_size_mm'], 
                    sub[metric], 
                    color=color, 
                    lw=2,
                    marker=marker, 
                    ms=7, 
                    mec=surface, 
                    mew=1.5, 
                    label=cookie, 
                    zorder=3)
            
            end_points.append((cookie, 
                               sub['fo_size_mm'].iloc[-1], 
                               sub[metric].iloc[-1]))
            m = mds_df[mds_df['cookie'] == cookie].iloc[0]
            mds = m.get(f'mds_mm_{metric}', np.nan)
            
            if np.isfinite(mds) and floor is not None:
                ax.plot([mds], [floor], 
                        marker=marker, 
                        ms=11, 
                        mfc='none',
                        mec=color, 
                        mew=2, 
                        zorder=4)

        # Direct labels at the small-size end of each curve
        end_points.sort(key=lambda e: e[2])
        stack, last_y = 0, None
        for cookie, x_end, y_end in end_points:
            stack = stack + 1 if (last_y is not None and y_end > 0 and last_y > 0 and abs(np.log10(y_end / last_y)) < 0.07) else 0
            ax.annotate(cookie, (x_end, y_end), 
                        xytext=(8, 11 * stack),
                        textcoords='offset points', 
                        va='center', 
                        fontsize=8, 
                        color=ink)
            last_y = y_end

        if floor is not None:
            ax.axhline(floor, 
                       color=muted, 
                       ls='--', 
                       lw=1.2, 
                       zorder=2)
            ax.text(0.01, floor, f'detection floor = {floor:g}',
                    transform=ax.get_yaxis_transform(), ha='left', va='bottom',
                    fontsize=8, color=muted)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(11, 0.45)                   # 10 mm left -> 1 mm right (room for labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel(_response_label_safe(metric), color=ink, fontsize=9)
        ax.grid(True, which='major', color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        
        # Customize plot spines
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        
        if p == 0:
            ax.set_title('Smallest detectable foreign object by product (5 TX / 5 RX MIMO)', color=ink, fontsize=12, loc='left')
            ax.legend(loc='upper right', 
                      frameon=False, 
                      fontsize=8, 
                      labelcolor=ink)
        if p == n_panels - 1:
            ax.set_xlabel('Foreign-object edge length (mm), centered, cube', color=ink)

    # Prepare table data for the summary table underneath the plot (temporary)
    labels = ([_cookie_metric_short_name(m) for m in response_metrics] + ['combined'])
    keys = list(response_metrics) + ['combined']
    rows = []
    for _, m in mds_df.iterrows():
        rows.append([m['cookie'], f"{m['thickness_mm']:g}"]
                    + [_fmt_mds(m.get(f'mds_mm_{k}', np.nan),
                                m.get(f'status_{k}', 'no_floor_for_metric')) for k in keys])
    n_metric_cols = len(labels)
    col_widths = [0.40, 0.10] + [(1.0 - 0.50) / n_metric_cols] * n_metric_cols
    tab = ax_tab.table(cellText=rows,
                       colLabels=['Product (smallest detectable size ->)', 'mm thick']
                       + [f'{l}' for l in labels],
                       colWidths=col_widths,
                       loc='center', 
                       cellLoc='left', 
                       colLoc='left')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_facecolor(surface)
        cell.get_text().set_color(ink)

    # Save
    _savefig(fig, out_path, dpi=150, facecolor=surface, bbox_inches='tight')
    plt.close(fig)

# endregion

# region FO position study

def _plot_cookie_detectability_position(results_df: 'pd.DataFrame', 
                                        range_df: 'pd.DataFrame', response_metrics: list,
                                        floors: dict, 
                                        out_path, 
                                        detection_rule: str = None) -> None:
    """This function generates a plot showing the detectability of cookies as a function of the FO offset (from the SUT center).

    Args:
        results_df (pd.DataFrame): 
            Columns cookie, offset_mm, thickness_mm, fo_size_mm, plus one column per response metric.
        range_df (pd.DataFrame): 
            Output of _cookie_score_position_range(): cookie, thickness_mm, fo_size_mm and, per response metric m and for 'combined' etc.
        response_metrics (list): 
            Metric columns to plot, one panel each.
        floors (dict): 
            Metric -> detection threshold (None: no line).
        out_path (str or Path): 
            PNG destination.
        detection_rule (str): 
            Rule behind the table's limits ('conservative' or 'envelope'), printed above the table.
    """
    # Setup
    ink, muted, grid, surface = '#1a1a19', '#6b6a63', '#e4e3dc', '#fcfcfb'
    n_panels = len(response_metrics)
    fig = plt.figure(figsize=(9.8, 4.6 * n_panels + 2.6), facecolor=surface)
    gs = fig.add_gridspec(n_panels + 1, 1,
                          height_ratios=[3.2] * n_panels + [0.5 + 0.34 * len(range_df)],
                          hspace=0.32)
    axes = [fig.add_subplot(gs[i], facecolor=surface) for i in range(n_panels)]
    ax_tab = fig.add_subplot(gs[n_panels], facecolor=surface)
    ax_tab.axis('off')
    if detection_rule:
        ax_tab.set_title(f'largest detectable offset from center, {detection_rule} rule (floors as drawn)', 
        fontsize=8, color=muted, loc='left')

    # Get cookie data and parameters
    cookies = list(dict.fromkeys(results_df['cookie']))
    max_offset = max(float(results_df['offset_mm'].max()), 1.0)
    
    # Loop over each response metric and create diagram.
    for p, (ax, metric) in enumerate(zip(axes, response_metrics)):
        
        floor = floors.get(metric)
        end_points = []
        
        for i, cookie in enumerate(cookies):
            color = _COOKIE_COLORS[i % len(_COOKIE_COLORS)]
            marker = _COOKIE_MARKERS[i % len(_COOKIE_MARKERS)]
            sub = results_df[results_df['cookie'] == cookie].sort_values('offset_mm')
            ax.plot(sub['offset_mm'], 
                    sub[metric], 
                    color=color, 
                    lw=2,
                    marker=marker, 
                    ms=7, 
                    mec=surface, 
                    mew=1.5, 
                    label=cookie, 
                    zorder=3)
            end_points.append((cookie, sub['offset_mm'].iloc[-1], sub[metric].iloc[-1]))
            r = range_df[range_df['cookie'] == cookie].iloc[0]
            d = r.get(f'max_offset_mm_{metric}', np.nan)

            # Plot floor at end
            if np.isfinite(d) and floor is not None:
                ax.plot([d], [floor], marker=marker, ms=11, mfc='none',
                        mec=color, mew=2, zorder=4)

        # Annotations for end points
        end_points.sort(key=lambda e: e[2])
        stack, last_y = 0, None
        for cookie, x_end, y_end in end_points:
            stack = stack + 1 if (last_y is not None and y_end > 0 and last_y > 0 and abs(np.log10(y_end / last_y)) < 0.07) else 0
            ax.annotate(cookie, (x_end, y_end), xytext=(8, 11 * stack),
                        textcoords='offset points', va='center', fontsize=8, color=ink)
            last_y = y_end

        # Plot detection floor line and label
        if floor is not None:
            ax.axhline(floor, color=muted, ls='--', lw=1.2, zorder=2)
            ax.text(0.01, floor, f'detection floor = {floor:g}',
                    transform=ax.get_yaxis_transform(), ha='left', va='bottom',
                    fontsize=8, color=muted)
        
        ax.set_yscale('log')
        ax.set_xlim(-0.03 * max_offset, 1.18 * max_offset)
        ax.set_ylabel(_response_label_safe(metric), color=ink, fontsize=9)
        ax.grid(True, which='major', color=grid, lw=0.8)
        ax.tick_params(colors=muted)

        # Define spines
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        if p == 0:
            ax.set_title('Detectability of a fixed-size foreign object vs. its position (5 TX / 5 RX MIMO)', color=ink, fontsize=12, loc='left')
            ax.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=ink)
        if p == n_panels - 1:
            ax.set_xlabel('Foreign-object offset from SUT center (mm)', color=ink)

    labels = ([_cookie_metric_short_name(m) for m in response_metrics] + ['combined'])
    keys = list(response_metrics) + ['combined']
    rows = []
    for _, r in range_df.iterrows():
        rows.append([r['cookie'], f"{r['thickness_mm']:g}", f"{r['fo_size_mm']:g}"] + [_fmt_offset(r.get(f'max_offset_mm_{k}', np.nan),
        r.get(f'status_{k}', 'no_floor_for_metric')) for k in keys])
    
    n_metric_cols = len(labels)
    col_widths = [0.34, 0.09, 0.08] + [(1.0 - 0.51) / n_metric_cols] * n_metric_cols

    # Create table with general information of the FO position study results
    tab = ax_tab.table(cellText=rows,
                       colLabels=['Product (largest detectable offset ->)', 'mm thick', 'FO mm']
                       + [f'{l}' for l in labels],
                       colWidths=col_widths,
                       loc='center', cellLoc='left', colLoc='left')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_facecolor(surface)
        cell.get_text().set_color(ink)
    
    # Save figure
    _savefig(fig, out_path, dpi=150, facecolor=surface, bbox_inches='tight')
    plt.close(fig)


def _plot_cookie_detectability_comparison_journal(mds_all: 'pd.DataFrame', 
                                                  sens_df: 'pd.DataFrame', out_path) -> None:
    """This function generates a comparison plot for cookie detectability, showing the smallest detectable size per product (MDS) and the mean over detected products versus assumed SNR.

    Args:
        mds_all (pd.DataFrame): 
            Results of MDS (per side and product)
        sens_df (pd.DataFrame): 
            Same for SNR
        out_path (str or Path): 
            Destination of PNG file.
    
    Returns:
        None

    Raises:
        None
    """
    # Format specification
    colors = {'reflection': _COOKIE_COLORS[0], 
              'transmission': _COOKIE_COLORS[1]}
    hatch = {'reflection': None, 
             'transmission': '////'}
    sides = ['reflection', 
             'transmission']

    # Get sorted cookie data
    cookies = sorted(dict.fromkeys(mds_all['cookie']), key=_cookie_short_name)

    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7,
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5, 
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42, 
          'hatch.linewidth': 0.6}

    # Define helper function for drawing bars with labels or 'n.d.'
    def _bar(draw, pos, value, status, stub, **kw):
        if np.isfinite(value):
            return draw(pos, value, **kw), ('<' if status == 'below_range' else '') + f'{value:.2f}'
        
        return draw(pos, stub, **kw), 'n.d.'

    # Create the plot within the specified rc context
    with plt.rc_context(rc):

        two = len(sens_df) > 0
        fig, axs = plt.subplots(1, 
                                2 if two else 1, 
                                figsize=(523.92 / 72, 3.0), constrained_layout=True, 
                                squeeze=False,
                                gridspec_kw={'width_ratios': [1.25, 1]} if two else None)
        ax = axs[0][0]

        # Loop over each side and cookie to draw bars with appropriate labels or 'n.d.'.
        top = np.nanmax(mds_all['mds_mm_combined'].to_numpy(dtype=float))
        for si, sd in enumerate(sides):
            
            sub = mds_all[mds_all['rx_side'] == sd].set_index('cookie')

            # Loop over each cookie for the current side and draw the a bar.
            for ci, cookie in enumerate(cookies):
                v, st = float(sub.loc[cookie, 'mds_mm_combined']), sub.loc[cookie, 'status_combined']
                y = ci + (si - 0.5) * 0.38
                bar, text = _bar(lambda p, h, **kw: ax.barh(p, h, 0.36, color=colors[sd], hatch=hatch[sd], edgecolor='white', lw=0.4, **kw), y, v, st, 0.04 * top)
                ax.text(bar[0].get_width() + 0.01 * top, y, text, va='center', fontsize=6)

        ax.set_yticks(range(len(cookies)))
        ax.set_yticklabels([c for c in cookies])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15 * top)
        ax.set_xlabel('Smallest detectable edge length / mm')
        ax.set_title('(a) Detectability by product')
        ax.grid(True, axis='x', color='#e4e3dc', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        # If two panels are requested, create the second panel for detectability vs. assumed SNR.
        panels = [ax]
        if two:

            ax = axs[0][1]
            snrs = sorted(sens_df['snr_db'].unique())
            tops = []
            for si, sd in enumerate(sides):
                for k, snr in enumerate(snrs):
                    sub = sens_df[(sens_df['rx_side'] == sd) & (sens_df['snr_db'] == snr)]
                    vals = sub['mds_mm_combined'].to_numpy(dtype=float)
                    tops.append(np.nanmax(vals) if np.isfinite(vals).any() else 0.0)

            # Draw side-specific bar
            top2 = max(tops) or 1.0
            for si, sd in enumerate(sides):
                for k, snr in enumerate(snrs):
                    sub = sens_df[(sens_df['rx_side'] == sd) & (sens_df['snr_db'] == snr)]
                    vals = sub['mds_mm_combined'].to_numpy(dtype=float)
                    n_det = int(np.isfinite(vals).sum())
                    mean = float(np.nanmean(vals)) if n_det else np.nan
                    below = bool((sub['status_combined'] == 'below_range').any())
                    x = k + (si - 0.5) * 0.38
                    bar, text = _bar(lambda p, h, 
                                     **kw: ax.bar(p, h, 0.36, color=colors[sd], hatch=hatch[sd], edgecolor='white', lw=0.4, **kw), 
                                     x, mean, 
                                     'below_range' if below else '', 0.04 * top2)
                    if 0 < n_det < len(vals):
                        text += f'\n({n_det}/{len(vals)})'
                    ax.text(x, bar[0].get_height() + 0.01 * top2, text, ha='center', va='bottom', fontsize=6)

            ax.set_xticks(range(len(snrs)))
            ax.set_xticklabels([f'{s:g} dB' for s in snrs])
            ax.set_ylim(0, 1.18 * top2)
            ax.set_xlabel('Assumed per-bin SNR')
            ax.set_ylabel('Mean MDS over detected products / mm')
            ax.set_title('(b) Detectability vs. assumed SNR')
            ax.grid(True, axis='y', color='#e4e3dc', lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            panels.append(ax)
        
        for a in panels:
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
        
        fig.legend(handles=[Patch(facecolor=colors[sd], hatch=hatch[sd], 
                                  edgecolor='white', label=sd.capitalize()) for sd in sides],
                   loc='outside upper center', 
                   ncol=2, 
                   frameon=False)

        # Just save
        _savefig(fig, out_path, dpi=300)
        plt.close(fig)


def _plot_cookie_detectability_journal(results_df: 'pd.DataFrame', 
                                       mds_df: 'pd.DataFrame',
                                       response_metrics: list, 
                                       floors: dict,
                                       out_path) -> None:
    """This function creates a figure showing the detectability of cookies with regard to MDS for different response metrics.

    Args:
        results_df (pd.DataFrame): 
            Columns cookie, fo_size_mm, plus one column per response metric.
            Cookie parameters and response metrics.
        mds_df (pd.DataFrame): 
            Cookie-specific MDS and response metrics
        response_metrics (list): 
            Metric columns to plot, one panel each.
        floors (dict): 
            Metric -> detection threshold (None: no line).
        out_path (str or Path): 
            Destination; written as .png.

    Returns:
        None

    Raises:
        None
    """
    # Format specification
    titles = {'rms_dphase_max': 
              ('RMS \u0394phase / deg', r'$\varphi_\mathrm{floor}$', '\u00b0'),
              'rms_dmag_max': 
              ('RMS \u0394$|H|$ / dB', r'$A_\mathrm{floor}$', ' dB')}
    ink, muted, grid = '#1a1a19', '#6b6a63', '#e4e3dc'
    ticks = [10, 8, 6, 5, 4, 3, 2, 1.5, 1]

    # Cookie data
    cookies = list(dict.fromkeys(results_df['cookie']))

    # Define helper function for cookie-related labels
    def short(cookie):
        thick = mds_df.loc[mds_df['cookie'] == cookie, 'thickness_mm']
        base = cookie.split(' (')[0].replace('-', ' ')
        return f'{base} ({thick.iloc[0]:g} mm)' if len(thick) else base

    rc = {'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.titlesize': 7.5,
          'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
          'legend.fontsize': 6.5, 'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 'pdf.fonttype': 42}

    # Create figure and axes for each response metric
    with plt.rc_context(rc):

        n = len(response_metrics)
        fig, axs = plt.subplots(1, n, 
                                figsize=(523.92 / 72, 3.0),
                                constrained_layout=True, 
                                squeeze=False)

        # Plot each response metric on its respective axis
        for j, (ax, metric) in enumerate(zip(axs[0], response_metrics)):
            
            floor = floors.get(metric)
            title, symbol, unit = titles.get(metric, (_response_label_safe(metric), 'Floor', ''))
            
            if floor is not None:
                ax.axhline(floor, color='black', 
                           lw=0.8, ls=(0, (4, 3)), 
                           zorder=1)
                ax.text(0.99, floor * 1.06, f'{symbol} = {floor:g}{unit}',
                        transform=ax.get_yaxis_transform(), ha='right',
                        va='bottom', fontsize=6, color=ink)

            # Plot lines for each cookie (size of FO)
            for k, cookie in enumerate(cookies):
                col = _COOKIE_COLORS[k % len(_COOKIE_COLORS)]
                mk = _COOKIE_MARKERS[k % len(_COOKIE_MARKERS)]
                sub = results_df[results_df['cookie'] == cookie].sort_values(
                    'fo_size_mm', ascending=False)
                ax.plot(sub['fo_size_mm'], 
                        sub[metric], 
                        color=col, 
                        lw=1.0,
                        marker=mk, 
                        ms=3.6, 
                        mec='white', 
                        mew=0.5, 
                        zorder=3)
                d = mds_df.loc[mds_df['cookie'] == cookie].iloc[0].get(
                    f'mds_mm_{metric}', np.nan)
                if np.isfinite(d) and floor is not None:
                    ax.plot([d], 
                            [floor], 
                            marker='o', 
                            ms=7 + 2.5 * k, 
                            mfc='none', 
                            mec=col, 
                            mew=1.1, 
                            zorder=4)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(11.5, 0.85)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(f'({chr(97 + j)}) {title}')
            ax.set_xlabel('Foreign-object edge length / mm')
            ax.set_ylabel('Max over channels (log scale)')
            ax.grid(True, which='major', color=grid, lw=0.5, zorder=0)
            
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

        # Set handles for legend
        handles = [Line2D([], [], 
                          color=_COOKIE_COLORS[k % len(_COOKIE_COLORS)], lw=1.0,
                          marker=_COOKIE_MARKERS[k % len(_COOKIE_MARKERS)], ms=3.6,
                          mec='white', 
                          mew=0.5, 
                          label=short(c))
                   for k, c in enumerate(cookies)]
        
        handles.append(Line2D([], [], 
                              color='none', 
                              marker='o', 
                              ms=8, 
                              mfc='none', 
                              mec=muted, 
                              mew=1.1, 
                              label='Smallest detectable size'))

        # Set legend and save
        fig.legend(handles=handles, 
                   loc='outside upper center', 
                   ncol=len(handles), 
                   frameon=False, 
                   columnspacing=1.0, 
                   handletextpad=0.4)
        _savefig(fig, out_path, dpi=300)
        plt.close(fig)


def _plot_cookie_detectability_position_journal(results_df: 'pd.DataFrame',
                                                range_df: 'pd.DataFrame', response_metrics: list,
                                                floors: dict, 
                                                out_path) -> None:
    """This function generates plots of cookie detectability as a function of position (FO position study)

    Args:
        results_df (pd.DataFrame): 
            Columns cookie, offset_mm, fo_size_mm, plus one column for every response metric.
        range_df (pd.DataFrame): 
            Output of _cookie_score_position_range(): cookie, thickness_mm, fo_size_mm and max_offset_mm_<m> per metric m.
        response_metrics (list): 
            Metric columns to plot, one panel each.
        floors (dict): 
            Detection threshold (None means no line).
        out_path (str or Path): 
            Destination (PNG file)

    Returns:
        None

    Raises:
        None
    """
    titles = {'rms_dphase_max': 
              ('RMS Δphase / deg', r'$\varphi_\mathrm{floor}$', '°'), 'rms_dmag_max': 
              ('RMS Δ$|H|$ / dB', r'$A_\mathrm{floor}$', ' dB')}
    ink, muted, grid = '#1a1a19', '#6b6a63', '#e4e3dc'
    
    # Get cookie data and maximum offset
    cookies = list(dict.fromkeys(results_df['cookie']))
    max_offset = max(float(results_df['offset_mm'].max()), 1.0)

    # Define this helper function to create a cookie-specific label
    def short(cookie):
        r = range_df.loc[range_df['cookie'] == cookie]
        base = cookie.split(' (')[0].replace('-', ' ')
        if len(r):
            return f"{base} ({r['thickness_mm'].iloc[0]:g} mm)"
        
        return base

    rc = {'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.titlesize': 7.5,
          'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
          'legend.fontsize': 6.5, 'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 'pdf.fonttype': 42}

    # Create the plots within the specified rc context
    with plt.rc_context(rc):

        n = len(response_metrics)
        fig, axs = plt.subplots(1, n, 
                                figsize=(523.92 / 72, 3.0),
                                constrained_layout=True, 
                                squeeze=False)
        
        for j, (ax, metric) in enumerate(zip(axs[0], response_metrics)):
            
            floor = floors.get(metric)
            title, symbol, unit = titles.get(metric, (_response_label_safe(metric), 'Floor', ''))

            # If a floor value is specified, draw a horizontal line and label it
            if floor is not None:
                ax.axhline(floor, color='black', lw=0.8, ls=(0, (4, 3)), zorder=1)
                ax.text(0.99, floor * 1.06, f'{symbol} = {floor:g}{unit}',
                        transform=ax.get_yaxis_transform(), ha='right',
                        va='bottom', fontsize=6, color=ink)
            
            # Plot the data for each cookie
            for k, cookie in enumerate(cookies):
                
                col = _COOKIE_COLORS[k % len(_COOKIE_COLORS)]
                mk = _COOKIE_MARKERS[k % len(_COOKIE_MARKERS)]
                sub = results_df[results_df['cookie'] == cookie].sort_values('offset_mm')
                
                ax.plot(sub['offset_mm'], 
                        sub[metric], 
                        color=col, 
                        lw=1.0,
                        marker=mk, 
                        ms=3.6, 
                        mec='white', 
                        mew=0.5, 
                        zorder=3)
                d = range_df.loc[range_df['cookie'] == cookie].iloc[0].get(
                    f'max_offset_mm_{metric}', np.nan)

                # Plot the largest detectable offset for this cookie
                if np.isfinite(d) and floor is not None:
                    ax.plot([d], [floor], marker='o', ms=7 + 2.5 * k, mfc='none',mec=col, mew=1.1, zorder=4)

            ax.set_yscale('log')
            ax.set_xlim(-0.03 * max_offset, 1.03 * max_offset)
            ax.set_title(f'({chr(97 + j)}) {title}')
            ax.set_xlabel('Foreign-object offset from SUT center / mm')
            ax.set_ylabel('Max over channels (log scale)')
            ax.grid(True, which='major', color=grid, lw=0.5, zorder=0)
            
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

        handles = [Line2D([], [], 
                          color=_COOKIE_COLORS[k % len(_COOKIE_COLORS)], lw=1.0,
                          marker=_COOKIE_MARKERS[k % len(_COOKIE_MARKERS)], ms=3.6,
                          mec='white', mew=0.5, 
                          label=short(c))
                   for k, c in enumerate(cookies)]
        handles.append(Line2D([], [], 
                              color='none', 
                              marker='o', 
                              ms=8, 
                              mfc='none',
                              mec=muted, 
                              mew=1.1, 
                              label='Largest detectable offset'))

        # Set legend and save the figure
        fig.legend(handles=handles, 
                   loc='outside upper center', 
                   ncol=len(handles), 
                   frameon=False, 
                   columnspacing=1.0, 
                   handletextpad=0.4)
        _savefig(fig, out_path)
        plt.close(fig)


def _plot_cookie_position_comparison_journal(
            res: 'pd.DataFrame', 
            cov: 'pd.DataFrame', 
            share: 'pd.DataFrame', 
            thickness: 'pd.Series',
            snrs: list, 
            out_path, 
            x_col: str = 'offset_mm',
            x_label: str = 'Foreign-object offset from SUT center / mm',
            title_a: str = '(a) Response vs. offset', 
            what: str = 'positions',
            x_ticks: list = None, 
            legend_loc: str = 'upper right', 
            y_headroom: float = 1.0) -> None:
    """This function creates a figure comparing cookie detectability across different positions for various products based on reflection and transmission responses (combined response / threshold; N of positions above threshold; share of positions above threshold with assumed SNR).

    Args:
        res (pd.DataFrame): 
            Columns rx_side, product, offset_mm, ratio (combined response / threshold, nominal thresholds).
        cov (pd.DataFrame): 
            Index product; n_above_<side>_<snr>dB and n_positions.
        share (pd.DataFrame): 
            rx_side, snr_db, share_percent.
        thickness (pd.Series): 
            Product thickness in mm by short product name.
        snrs (list): 
            Assumed SNRs in dB
        out_path (str or Path): 
            Destination of the PNG file
        x_col (str): 
            Column of 'res' on the x axis of panel (offset or frquency).
        x_label (str): 
            x-label of first panel
        title_a (str): 
            Title of first panel
        what (str): 
            Noun for the counted items in panels (b) and (c): 'positions' or 'bands'.
        x_ticks (list): 
            Fixed x ticks of first panel
        legend_loc (str): 
            Location of the product legend of first panel
        y_headroom (float): 
            Factor on the upper y limit of first panel for the legend.
    """
    sides = ['reflection', 
             'transmission']
    colors = {'reflection': _COOKIE_COLORS[0], 
              'transmission': _COOKIE_COLORS[1]}
    hatch = {'reflection': None, 
             'transmission': '////'}
    
    # Get products and SNR for plotting
    prods = list(cov.index)
    snr_mid = 30 if 30 in snrs else snrs[len(snrs) // 2]
    n_pos = int(cov['n_positions'].max())
    
    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7, 
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5, 
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left', 
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42, 
          'hatch.linewidth': 0.6}

    # Apply the custom rc settings for the plot context
    with plt.rc_context(rc):

        fig, axs = plt.subplots(1, 3, 
                                figsize=(523.92 / 72, 3.1), constrained_layout=True,
                                gridspec_kw={'width_ratios': [1.45, 1, 1]})
        
        # Begin plotting the first panel (response ratio)
        ax = axs[0]
        ax.axhline(1.0, color='black', lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(0.99, 1.0 / 1.08, 'threshold = 1', transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=6)

        # Plot the response ratio for each product and side
        for sd in sides:
            for k, p in enumerate(prods):
                g = res[(res['rx_side'] == sd) & (res['product'] == p)].sort_values(x_col)
                ax.plot(g[x_col], g['ratio'], 
                        color=colors[sd], 
                        lw=1.0, 
                        ls='-' if sd == 'reflection' else (0, (4, 2)),marker=_COOKIE_MARKERS[k % 4], 
                        ms=3.6, 
                        mec='white', 
                        mew=0.5, 
                        zorder=3)
                
        ax.set_yscale('log')
        
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{t:g}' for t in x_ticks])
        if y_headroom != 1.0:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * y_headroom)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Response / threshold (combined, log scale)')
        ax.set_title(title_a)
        
        ax.legend(handles=[Line2D([], [], 
                                  color='black', 
                                  lw=0, 
                                  marker=_COOKIE_MARKERS[k % 4], 
                                  ms=3.6,
                                  label=f"{p} ({thickness[p]:g} mm)") for k, p in enumerate(prods)],
                  loc=legend_loc, frameon=False, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)
        ax.grid(True, which='major', color='#e4e3dc', lw=0.5, zorder=0)

        # Begin plotting the second panel (coverage by product)
        ax = axs[1]
        for si, sd in enumerate(sides):
            for ci, p in enumerate(prods):
                v = int(cov.loc[p, f'n_above_{sd}_{snr_mid:g}dB'])
                y = ci + (si - 0.5) * 0.38
                ax.barh(y, v, 0.36, color=colors[sd], hatch=hatch[sd], edgecolor='white', lw=0.4)
                ax.text(v + 0.15, y, f'{v}', va='center', fontsize=6)
        ax.set_yticks(range(len(prods)))
        ax.set_yticklabels([p.replace(' ', '\n') for p in prods])
        ax.invert_yaxis()
        ax.set_xlim(0, n_pos + 1.5)
        ax.set_xticks(range(0, n_pos + 1, 1 if n_pos <= 6 else 2))
        ax.set_xlabel(f'{what.capitalize()} above threshold (of {n_pos})')
        ax.set_title(f'(b) Coverage by product ({snr_mid:g} dB)')
        ax.grid(True, axis='x', color='#e4e3dc', lw=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Begin plotting the third panel (coverage vs. SNR)
        ax = axs[2]
        for si, sd in enumerate(sides):
            for k, snr in enumerate(snrs):
                v = float(share[(share['rx_side'] == sd) & (share['snr_db'] == snr)]['share_percent'].iloc[0])
                x = k + (si - 0.5) * 0.38
                ax.bar(x, v, 0.36, color=colors[sd], hatch=hatch[sd], edgecolor='white', lw=0.4)
                ax.text(x, v + 1.0, f'{v:.0f}', ha='center', va='bottom', fontsize=6)
        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([f'{s:g} dB' for s in snrs])
        ax.set_ylim(0, 112)
        ax.set_xlabel('Assumed per-bin SNR')
        ax.set_ylabel(f'{what.capitalize()} above threshold / %\n(mean across products)')
        ax.set_title('(c) Coverage vs. SNR')
        ax.grid(True, axis='y', color='#e4e3dc', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        for a in axs:
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)

        fig.legend(handles=[Patch(facecolor=colors[sd], 
                                  hatch=hatch[sd], 
                                  edgecolor='white', 
                                  label=sd.capitalize()) for sd in sides],
                   loc='outside upper center', 
                   ncol=2, 
                   frameon=False)
        
        # Save
        _savefig(fig, out_path, dpi=300)
        plt.close(fig)

# endregion

# region FO frequency study

def _plot_cookie_detectability_frequency(results_df: 'pd.DataFrame', 
                                         response_metrics: list, 
                                         floors: dict, 
                                         out_path) -> None:
    """This function generates a plot showing the detectability of a fixed foreign object across different frequency bands. Response of the fixed foreign object vs. band-center frequency.

    Args:
        results_df (pd.DataFrame): 
            Columns cookie, f_center_ghz, fo_size_mm, plus one column per response metric.
        response_metrics (list): 
            Metric columns to plot
        floors (dict): 
            Detection threshold (None: no line).
        out_path (str or Path): 
            PNG file output path.

    Returns:
        None

    Raises:
        None
    """
    titles = {'rms_dphase_max': 
              ('RMS Δphase / deg', r'$\varphi_\mathrm{floor}$', '°'),
              'rms_dmag_max': 
              ('RMS Δ$|H|$ / dB', r'$A_\mathrm{floor}$', ' dB')}
    ink, muted, grid = '#1a1a19', '#6b6a63', '#e4e3dc'
    
    # Get cookie data and frequency parameters
    cookies = list(dict.fromkeys(results_df['cookie']))
    f_centers = sorted(results_df['f_center_ghz'].unique())
    fo_mm = float(results_df['fo_size_mm'].iloc[0])
    c0 = sp.constants.c

    # Define a helper function to shorten cookie names
    def short(cookie):
        base = cookie.split(' (')[0].replace('-', ' ')
        t = results_df.loc[results_df['cookie'] == cookie, 'thickness_mm']
        
        return f"{base} ({t.iloc[0]:g} mm)" if len(t) else base

    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5,
          'axes.labelsize': 7, 
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5,
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42}

    # Apply the custom rc settings for the plot context
    with plt.rc_context(rc):

        n = len(response_metrics)

        fig, axs = plt.subplots(1, n, 
                                figsize=(523.92 / 72, 3.0),
                                constrained_layout=True, 
                                squeeze=False)
        
        for j, (ax, metric) in enumerate(zip(axs[0], response_metrics)):

            floor = floors.get(metric)
            title, symbol, unit = titles.get(metric, (_response_label_safe(metric), 'Floor', ''))
            
            if floor is not None:
                ax.axhline(floor, 
                           color='black', 
                           lw=0.8, 
                           ls=(0, (4, 3)), 
                           zorder=1)
                last = results_df.loc[results_df['f_center_ghz'] == max(f_centers), metric].to_numpy(dtype=float)
                
                above = not (((last >= floor) & (last < 1.6 * floor)).any() and not ((last < floor) & (last > floor / 1.6)).any())
                
                ax.text(0.99, 
                        floor * 1.06 if above else floor / 1.06, f'{symbol} = {floor:g}{unit}',
                        transform=ax.get_yaxis_transform(), 
                        ha='right',
                        va='bottom' if above else 'top', 
                        fontsize=6, 
                        color=ink)

            # Plot the data for each cookie
            for k, cookie in enumerate(cookies):
                col = _COOKIE_COLORS[k % len(_COOKIE_COLORS)]
                mk = _COOKIE_MARKERS[k % len(_COOKIE_MARKERS)]
                sub = results_df[results_df['cookie'] == cookie].sort_values('f_center_ghz')
                ax.plot(sub['f_center_ghz'], sub[metric], 
                        color=col, 
                        lw=1.0, 
                        marker=mk,
                        ms=3.6, 
                        mec='white', 
                        mew=0.5, 
                        zorder=3)
                
                if floor is not None:
                    low = sub[sub[metric] < floor]
                    ax.plot(low['f_center_ghz'], low[metric], 
                            ls='none', 
                            marker='o', 
                            ms=7 + 2.5 * k, 
                            mfc='none', 
                            mec=col, 
                            mew=1.1, 
                            zorder=4)
                    
            ax.set_yscale('log')
            span = max(f_centers) - min(f_centers) if len(f_centers) > 1 else 10.0
            ax.set_xlim(min(f_centers) - 0.06 * span, 
                        max(f_centers) + 0.06 * span)
            ax.set_xticks(f_centers)
            ax.set_xticklabels([f'{f:g}\n({f * 1e9 * fo_mm * 1e-3 / c0:.2f}' + r'$\lambda_0$)' for f in f_centers])
            ax.set_title(f'({chr(97 + j)}) {title}')
            ax.set_xlabel(r'Band-center frequency / GHz (FO edge / $\lambda_0$)')
            ax.set_ylabel('Max over channels (log scale)')
            ax.grid(True, which='major', color=grid, lw=0.5, zorder=0)
            
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

        handles = [Line2D([], [], 
                          color=_COOKIE_COLORS[k % len(_COOKIE_COLORS)], 
                          lw=1.0,
                          marker=_COOKIE_MARKERS[k % len(_COOKIE_MARKERS)], ms=3.6,
                          mec='white', 
                          mew=0.5, 
                          label=short(c))
                   for k, c in enumerate(cookies)]
        handles.append(Line2D([], [], 
                              color='none', 
                              marker='o', 
                              ms=8, 
                              mfc='none',
                              mec=muted, 
                              mew=1.1, 
                              label='Below threshold'))
        fig.legend(handles=handles, 
                   loc='outside upper center', 
                   ncol=len(handles),
                   frameon=False, 
                   columnspacing=1.0, 
                   handletextpad=0.4)

        # Save figure
        _savefig(fig, out_path)
        plt.close(fig)

# endregion

# region Miscellaneous auxiliary plot functions

def _response_label_safe(response_metric: str) -> str:
    """This function attempts to return a axis label for the given response metric.

    Args:
        response_metric (str): 
            The response metric for which to generate an axis label.

    Returns:
        str: the axis label

    Raises:
        Exception: If the response label cannot be generated, the raw metric name is returned.
    """
    try:
        return PlotsMixin._response_label(response_metric)
    except Exception:
        return response_metric


def _combined_sort_key(v):
    """This function returns a  key for sorting that treats numeric values as numbers and non-numeric values as strings.
    
    Args:
        v: The value to generate a sort key for.

    Returns:
        tuple: A tuple where the first element is 0 for numeric values and 1 for non-numeric values. The second element is the numeric value or  a string.

    Raises:
        TypeError: If the value cannot be converted to a float and is not a string. 
    """
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))

#-->
def plot_combined_contrast_vs_rmsdphase(scored_df: pd.DataFrame,
                                         anova_summary_df: pd.DataFrame,
                                         output_dir,
                                         response_a: str = 'contrast_real',
                                         response_b: str = 'rms_dphase_max',
                                         trace_factor: str = None,
                                         sig_level: float = _ANOVA_SIG_LEVEL,
                                         factor_columns: list = None,
                                         filename_tag: str = 'contrast_vs_rmsdphase') -> None:
    """Write the two combined overlay figures: 'combined_main_effects_<filename_tag>.png' (every factor) and
    'combined_interactions_<filename_tag>.png' (only 2-way terms significant for `response_a` or `response_b`).

    Args:
        scored_df (pandas.DataFrame): analyze_doe_batch()'s scored rows (status 'OK' only); must carry `response_a`/`response_b`
            and every column of `factor_columns`.
        anova_summary_df (pandas.DataFrame): fit_anova_all_responses()'s return value (needs 'response'/'term'/'p_value');
            only used to pick the interaction subplots. If no 2-way term is significant, the interactions PNG is skipped with a note.
        output_dir (str or Path): Directory for both PNGs.
        response_a, response_b (str): The two responses to overlay, `response_a` on the left y-axis, `response_b` on the right.
        trace_factor (str): Factor whose levels become the two line styles in the interaction subplots; None = 'moisture' if present, else the last factor.
        sig_level (float): p-value threshold for a 2-way term to get a subplot (default _ANOVA_SIG_LEVEL = 0.05).
        factor_columns (list of str): DOE factor columns to consider; None = ANOVA_CODE_TO_FACTOR's readable names present in `scored_df`.
        filename_tag (str): Suffix of both output filenames (default 'contrast_vs_rmsdphase').

    Raises:
        ValueError: If `response_a`/`response_b` aren't both columns of `scored_df`, or fewer than 2 factor columns are found.
    """
    if factor_columns is None:
        factor_columns = [c for c in ANOVA_CODE_TO_FACTOR.values() if c in scored_df.columns]
    if len(factor_columns) < 2:
        raise ValueError(
            f"plot_combined_contrast_vs_rmsdphase(): fewer than 2 factor "
            f"columns found ({factor_columns}) - nothing to plot.")
    missing = [r for r in (response_a, response_b) if r not in scored_df.columns]
    if missing:
        raise ValueError(
            f"plot_combined_contrast_vs_rmsdphase(): scored_df is missing "
            f"response column(s) {missing}.")
    if trace_factor is None:
        trace_factor = 'moisture' if 'moisture' in factor_columns else factor_columns[-1]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_combined_main_effects(scored_df, factor_columns, out_dir,
                                 response_a, response_b, filename_tag)
    _plot_combined_interactions(scored_df, anova_summary_df, factor_columns, out_dir,
                                 response_a, response_b, trace_factor, sig_level, filename_tag)


def _plot_combined_main_effects(scored_df, factor_columns, out_dir,
                                 response_a, response_b, filename_tag) -> None:
    """'combined_main_effects_<filename_tag>.png': one subplot per factor; `response_a` mean +/- 1 SEM (left axis, solid) and
    `response_b` mean +/- 1 SEM (right twin axis, dashed) vs. the factor's levels."""
    ncols = 3
    nrows = int(np.ceil(len(factor_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, factor in zip(axes_flat, factor_columns):
        grouped_a = scored_df.groupby(factor)[response_a].agg(['mean', 'std', 'count'])
        grouped_a = grouped_a.reindex(sorted(grouped_a.index, key=_combined_sort_key))
        sem_a = (grouped_a['std'] / np.sqrt(grouped_a['count'])).fillna(0.0)
        grouped_b = scored_df.groupby(factor)[response_b].agg(['mean', 'std', 'count'])
        grouped_b = grouped_b.reindex(sorted(grouped_b.index, key=_combined_sort_key))
        sem_b = (grouped_b['std'] / np.sqrt(grouped_b['count'])).fillna(0.0)
        x = np.arange(len(grouped_a))

        ax.errorbar(x, grouped_a['mean'], yerr=sem_a, marker='o', capsize=4,
                     linestyle='-', color=_COMBINED_CONTRAST_COLOR_DARK)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in grouped_a.index], rotation=20, ha='right')
        ax.set_xlabel(factor)
        ax.set_ylabel(response_a, color=_COMBINED_CONTRAST_COLOR_DARK)
        ax.tick_params(axis='y', labelcolor=_COMBINED_CONTRAST_COLOR_DARK)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.errorbar(x, grouped_b['mean'], yerr=sem_b, marker='s', capsize=4,
                      linestyle='--', color=_COMBINED_RMSDPHASE_COLOR_DARK)
        ax2.set_ylabel(response_b, color=_COMBINED_RMSDPHASE_COLOR_DARK)
        ax2.tick_params(axis='y', labelcolor=_COMBINED_RMSDPHASE_COLOR_DARK)
        ax.set_title(factor)

    for ax in axes_flat[len(factor_columns):]:
        ax.axis('off')

    fig.suptitle(f"Main effects — {response_a} (blue, left axis) vs "
                 f"{response_b} (red, right axis)")
    fig.tight_layout()
    png_path = out_dir / f"combined_main_effects_{filename_tag}.png"
    _savefig(fig, png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"plot_combined_contrast_vs_rmsdphase(): wrote '{png_path}'.")


def _plot_combined_interactions(scored_df, anova_summary_df, factor_columns, out_dir,
                                 response_a, response_b, trace_factor, sig_level,
                                 filename_tag) -> None:
    """'combined_interactions_<filename_tag>.png': one subplot per 2-way term significant (p < sig_level) for `response_a` or `response_b`,
    both responses overlaid (twin axis) at every level combination, one line style per level of the trace factor.
    Writes nothing (prints a note) if no term qualifies."""
    sig_mask = (anova_summary_df['response'].isin([response_a, response_b])
                & anova_summary_df['term'].str.contains(':', regex=False)
                & (anova_summary_df['p_value'] < sig_level))
    sig_terms = set(anova_summary_df.loc[sig_mask, 'term'])

    # itertools.combinations enumerates each unordered pair once, in factor_columns' order (same as the formula's interaction terms).
    ordered_pairs = []
    for fx, ftrace in itertools.combinations(factor_columns, 2):
        if f"{fx}:{ftrace}" in sig_terms or f"{ftrace}:{fx}" in sig_terms:
            # Orient so trace_factor is the 'trace' (line) side of the pair; otherwise keep as is.
            if fx == trace_factor and ftrace != trace_factor:
                ordered_pairs.append((ftrace, fx))
            else:
                ordered_pairs.append((fx, ftrace))

    if not ordered_pairs:
        print(f"plot_combined_contrast_vs_rmsdphase(): no 2-way interaction "
              f"term is significant (p < {sig_level}) for {response_a!r} or "
              f"{response_b!r} - skipping 'combined_interactions_"
              f"{filename_tag}.png'.")
        return

    ncols = min(3, len(ordered_pairs))
    nrows = int(np.ceil(len(ordered_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (fx, ftrace) in zip(axes_flat, ordered_pairs):
        ax2 = ax.twinx()
        trace_vals = sorted(scored_df[ftrace].unique(), key=_combined_sort_key)
        x = None
        for i, trace_val in enumerate(trace_vals):
            sub = scored_df[scored_df[ftrace] == trace_val]
            style = '-' if i == 0 else '--'
            marker_a = 'o' if i == 0 else '^'
            marker_b = 's' if i == 0 else 'D'
            color_a = _COMBINED_CONTRAST_COLOR_DARK if i == 0 else _COMBINED_CONTRAST_COLOR_LIGHT
            color_b = _COMBINED_RMSDPHASE_COLOR_DARK if i == 0 else _COMBINED_RMSDPHASE_COLOR_LIGHT

            grouped_a = sub.groupby(fx)[response_a].mean()
            grouped_a = grouped_a.reindex(sorted(grouped_a.index, key=_combined_sort_key))
            grouped_b = sub.groupby(fx)[response_b].mean()
            grouped_b = grouped_b.reindex(sorted(grouped_b.index, key=_combined_sort_key))
            x = np.arange(len(grouped_a))

            ax.plot(x, grouped_a.values, marker=marker_a, linestyle=style, color=color_a,
                     label=f"{response_a}, {ftrace}={trace_val}")
            ax2.plot(x, grouped_b.values, marker=marker_b, linestyle=style, color=color_b,
                      label=f"{response_b}, {ftrace}={trace_val}")

        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in grouped_a.index], rotation=20, ha='right')
        ax.set_xlabel(fx)
        ax.set_ylabel(response_a, color=_COMBINED_CONTRAST_COLOR_DARK)
        ax.tick_params(axis='y', labelcolor=_COMBINED_CONTRAST_COLOR_DARK)
        ax2.set_ylabel(response_b, color=_COMBINED_RMSDPHASE_COLOR_DARK)
        ax2.tick_params(axis='y', labelcolor=_COMBINED_RMSDPHASE_COLOR_DARK)
        ax.set_title(f"{fx} x {ftrace}")
        ax.grid(True, alpha=0.3)
        lines_a, labels_a = ax.get_legend_handles_labels()
        lines_b, labels_b = ax2.get_legend_handles_labels()
        ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc='best')

    for ax in axes_flat[len(ordered_pairs):]:
        ax.axis('off')

    fig.suptitle(f"Significant 2-way interactions — {response_a} (blue, left "
                 f"axis) vs {response_b} (red, right axis)")
    fig.tight_layout()
    png_path = out_dir / f"combined_interactions_{filename_tag}.png"
    _savefig(fig, png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"plot_combined_contrast_vs_rmsdphase(): wrote '{png_path}'.")

# endregion

# region ANOVA

def _anova_plot_pvalue_heatmap(summary_df: pd.DataFrame, 
                               out_dir: Path) -> None:
    """This function generates a heatmap of -log10(p) values for each term and response metric in the ANOVA summary.

    Args:
        summary_df (pd.DataFrame): 
            The ANOVA summary dataframe
        out_dir (Path): 
            The output directory

    Returns:
        None

    Raises:
        None
    """
    # Get terms excluding 'Intercept' and 'Residual'
    terms_df = summary_df[~summary_df['term'].isin(['Intercept', 'Residual'])]
    pivot = terms_df.pivot(index='term', columns='response', values='p_value')
    
    # Sort data (main effects first, then interactions, each alphabetical)
    main = sorted(t for t in pivot.index if ':' not in t)
    interactions = sorted(t for t in pivot.index if ':' in t)
    pivot = pivot.reindex(main + interactions)

    # Compute -log10(p) values. Clip to avoid very small p-values)
    neglogp = -np.log10(pivot.clip(lower=1e-300))


    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * pivot.shape[1] + 2.0),
                                    max(4.0, 0.35 * pivot.shape[0] + 2.0)))

    # Plot the heatmap
    im = ax.imshow(neglogp.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Highlight significant cells with a white outline (above threshold)
    sig_thresh = -np.log10(_ANOVA_SIG_LEVEL)
    values = neglogp.values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = values[i, j]
            if np.isfinite(v) and v >= sig_thresh:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 
                                       1, 1, 
                                       fill=False,
                                       edgecolor='white', 
                                       linewidth=1.5))

    fig.colorbar(im, ax=ax, 
                 label=f"-log10(p)  (white outline: p < {_ANOVA_SIG_LEVEL})")
    ax.set_title("ANOVA significance across every response metric")

    # Save the figure
    fig.tight_layout()
    _savefig(fig, out_dir / "anova_pvalue_heatmap.png", 
             dpi=300, 
             bbox_inches="tight")
    plt.close(fig)


def _anova_plot_r_squared_by_response(summary_df: pd.DataFrame, 
                                      out_dir: Path) -> None:
    """This function generates a bar plot of the ANOVA showing each response metric's model R2 (descending order).
    
    Args:
        summary_df (pd.DataFrame): 
            The ANOVA summary dataframe
        out_dir (Path): 
            The output directory.

    Returns:
        None

    Raises:
        None
    """
    per_response = (summary_df.drop_duplicates('response')[['response', 'r_squared']].sort_values('r_squared', ascending=False))
    
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(per_response) + 2.0), 
                                    5.0))
    
    ax.bar(per_response['response'], 
           per_response['r_squared'], 
           color='steelblue')
    ax.set_xticks(range(len(per_response)))
    ax.set_xticklabels(per_response['response'], 
                       rotation=60, ha='right', fontsize=7)
    ax.set_ylabel("R2 (fraction of variance explained by the 6 DOE factors)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("DOE model fit (R2) by response metric")
    
    # Save
    fig.tight_layout()
    _savefig(fig, out_dir / "anova_r_squared_by_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# endregion

# region Auxiliary

def _ff_style_axis(ax) -> None:
    """Light grid behind the data, no top/right spines.
    This function sets the style of an axis (light grid, removes spines).
    
    Args:
        ax (matplotlib.axes.Axes): The axis to style.

    Returns:
        None

    Raises:
        None
    """
    ax.grid(True, color=_FF_GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _ff_save(fig, out_dir: Path, 
             name: str) -> None:
    """This function saves the figure as a PNG file (then closes it).
    
    Args:
        fig (matplotlib.figure.Figure): 
            The figure to save.
        out_dir (Path): 
            The output directory.
        name (str): 
            The name of the output file (without extension).

    Returns:
        None

    Raises:
        None
    """
    path = _savefig(fig, out_dir / f'{name}.png', 
                    bbox_inches='tight', 
                    facecolor='white')
    plt.close(fig)
    
    print(f"export_full_factorial_paper_assets(): wrote '{path}'.")


def _plot_ff_variance_shares(shares: dict, 
                             out_dir: Path) -> None:
    """This function creates a figure showing the share of the total sum of squares per source for each response and study.

    Args:
        shares (dict): 
            {(study, response): {source: (share in %, p) or value}} from _ff_variance_shares().
        out_dir (Path): 
            Output folder.

    Returns:
        None

    Raises:
        None
    """
    # Define the sources of variance to be plotted.
    sources = list(_FF_FACTOR_NAMES) + ['INT2', 'INT3', 'RES']
    
    # Create the figure and axes for the variance shares plot.
    with plt.rc_context(_FF_RC):

        fig, axs = plt.subplots(2, 2, 
                                figsize=(7.16, 5.0), 
                                constrained_layout=True)
        y = np.arange(len(sources))[::-1]
        
        # Loop over each response and create a subplot for it.
        for i, (response, title) in enumerate(_FF_VARIANCE_RESPONSES):
            
            ax = axs[i // 2, i % 2]

            # Loop over each mode (reflection or transmission) and plot the variance shares as horizontal bars.
            for j, study in enumerate(_FF_STUDIES):
                values = np.array([shares[(study, response)][c][0] for c in sources])
                yy = y + (0.19 if j == 0 else -0.19)
                ax.barh(yy, values, 0.36, 
                        color=_FF_STUDY_COLORS[study], 
                        hatch='' if j == 0 else '////',
                        edgecolor='white', 
                        linewidth=0.5)
                for a, b in zip(yy, values):
                    if b >= 4:
                        ax.text(b + 1, a, 
                                f'{b:.0f}', 
                                va='center', 
                                fontsize=6, 
                                color=_FF_INK)
            
            ax.set_yticks(y)
            ax.set_yticklabels([_FF_SHARE_LABELS[c] for c in sources] if i % 2 == 0 else [], fontsize=6.5)
            ax.set_xlim(0, 100)
            ax.set_title(f'({"abcd"[i]}) ' + title, loc='left')
            ax.xaxis.grid(True, color=_FF_GRID, lw=0.5)
            ax.set_axisbelow(True)

            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', length=0)
            
            if i // 2 == 1:
                ax.set_xlabel('Share of total sum of squares (%)')

        fig.legend(handles=[Patch(facecolor=_FF_STUDY_COLORS['Reflection'], 
                                  label='Reflection'),
                            Patch(facecolor=_FF_STUDY_COLORS['Transmission'], 
                                  hatch='////', 
                                  edgecolor='white',
                                  label='Transmission')],
                   loc='lower center', 
                   ncol=2, 
                   frameon=False, 
                   bbox_to_anchor=(0.5, -0.06))

        # Save the figure
        _ff_save(fig, out_dir, 'fig_anova_variance_shares')


def _ff_level_stats(df: pd.DataFrame, 
                    factor: str, 
                    response: str, 
                    levels: list) -> tuple:
    """This function calculates statistics (mean and standard error of the response variable).
    
    Args:
        df (pd.DataFrame): 
            Input data frame containing the factor and response columns.
        factor (str): 
            Name of the factor column.
        response (str): 
            Name of the response column.
        levels (list): 
            List of factor levels in the desired order.

    Returns:
        tuple: A tuple containing two numpy arrays (Means of the response variable for each factor level; standard errors of the response variable for each factor level).
    """
    g = df.groupby(factor)[response].agg(['mean', 'std', 'count']).reindex(levels)
    return g['mean'].values, (g['std'] / np.sqrt(g['count'])).fillna(0).values


def _plot_ff_main_effects(data: dict, 
                          level_order: dict, 
                          out_dir: Path) -> None:
    """
    This function plots the main effects (ANOVA) for the specified response variables across different factor levels.

    Args:
        data (dict): 
            study: scored DOE rows.
        level_order (dict): 
            factor: levels in coded order (-1, +1).
        out_dir (Path): Output folder.
    """
    blue, orange = _FF_STUDY_COLORS['Reflection'], _FF_STUDY_COLORS['Transmission']

    # Plot main effects for each factor and study
    with plt.rc_context(_FF_RC):
        
        fig, axs = plt.subplots(2, 6, 
                                figsize=(7.16, 3.5), 
                                sharey='row', 
                                constrained_layout=True)
        x = np.arange(2)
        
        # Loop over each factor and plot the main effects for each study
        for k, factor in enumerate(_FF_FACTOR_NAMES):
            for j, study in enumerate(_FF_STUDIES):

                m, e = _ff_level_stats(data[study], factor, 'rms_dphase_max', level_order[factor])

                # Plot the main effect for the current factor and study on the top row of subplots
                axs[0, k].errorbar(x + (-0.04 if j == 0 else 0.04), 
                                   m, 
                                   yerr=e, 
                                   color=_FF_STUDY_COLORS[study],
                                   marker=_FF_STUDY_MARKERS[study], 
                                   ms=3.5, 
                                   lw=1.1, 
                                   capsize=2, 
                                   ls='-' if j == 0 else '--')
            
            m, e = _ff_level_stats(data[_FF_STUDIES[0]], 
                                   factor, 
                                   'contrast_real', 
                                   level_order[factor])
            
            axs[1, k].errorbar(x, m, 
                               yerr=e, 
                               color=_FF_INK, 
                               marker='D', 
                               ms=3, 
                               lw=1.0, 
                               capsize=2)
            axs[1, k].axhline(0, color=_FF_SUB, lw=0.6, ls=':')
            axs[0, k].set_title(_FF_FACTOR_NAMES[factor])

            # Set style and set limits for each subplot in the current column
            for row in (0, 1):
                _ff_style_axis(axs[row, k])
                axs[row, k].set_xlim(-0.45, 1.45)
                axs[row, k].set_xticks(x)
            axs[0, k].set_xticklabels([])
            axs[1, k].set_xticklabels([_FF_LEVEL_LABELS[factor][v] for v in level_order[factor]], rotation=25, ha='right')
        
        axs[0, 0].set_ylabel('RMS $\\Delta$phase (deg)')
        axs[1, 0].set_ylabel('contrast_real')
        fig.legend(handles=[Line2D([], [], 
                                   color=blue, 
                                   marker='o', 
                                   ms=3.5, 
                                   lw=1.1, 
                                   label='Reflection'),
                            Line2D([], [], 
                                   color=orange, 
                                   marker='s', 
                                   ms=3.5, 
                                   lw=1.1, 
                                   ls='--', 
                                   label='Transmission'),
                            Line2D([], [], 
                                   color=_FF_INK, 
                                   marker='D', 
                                   ms=3, 
                                   lw=1.0,
                                   label='contrast_real (materials only, both studies)')],
                   loc='lower center', 
                   ncol=3, 
                   frameon=False, 
                   bbox_to_anchor=(0.5, -0.07))

        # Save the figure for main effects shared across studies
        _ff_save(fig, out_dir, 'fig_main_effects_shared')


def _plot_ff_interactions(data: dict, 
                          pairs: list, 
                          level_order: dict, 
                          out_dir: Path) -> None:
    """This function plots the significant 2-way interactions.

    Args:
        data (dict): 
            study: scored DOE rows
        pairs (list of (str, str, str)): 
            (x factor, trace factor, studies with p < 0.05 as initials).
        level_order (dict): 
            factor: levels in coded order (-1, +1)
        out_dir (Path): 
            Output folder.

    Returns:
        None

    Raises:
        None
    """
    if not pairs:
        print("export_full_factorial_paper_assets(): no significant 2-way interaction. Fig_interactions_shared skipped.")
        return
    
    n_cols = 4
    n_rows = int(np.ceil(len(pairs) / n_cols))
    
    # Create subplots for the interaction plots in the figure
    with plt.rc_context(_FF_RC):

        fig, axs = plt.subplots(n_rows, n_cols, 
                                figsize=(7.16, 2.05 * n_rows + 0.35), sharey=True,
                                constrained_layout=True, 
                                squeeze=False)
        
        # Loop through each interaction pair and plot the data for each study and trace level
        x = np.arange(2)
        for ax, (fx, ft, who) in zip(axs.ravel(), pairs):
            for study in _FF_STUDIES:
                
                color = _FF_STUDY_COLORS[study]
                
                # Loop through each level of the trace factor and plot the corresponding data
                for i, tv in enumerate(level_order[ft]):
                    
                    sub = data[study][data[study][ft] == tv]
                    m = sub.groupby(fx)['rms_dphase_max'].mean().reindex(level_order[fx]).values
                    
                    ax.plot(x, m, 
                            color=color, 
                            marker=_FF_STUDY_MARKERS[study] if i == 0 else '^', 
                            ms=3.8, 
                            lw=1.1,
                            ls='-' if i == 0 else '--', mfc=color if i == 0 else 'white')
                    
            ax.set_xticks(x)
            ax.set_xticklabels([_FF_LEVEL_LABELS[fx][v] for v in level_order[fx]], fontsize=6.5)
            ax.set_xlim(-0.35, 1.35)

            _ff_style_axis(ax)

            ax.set_title(f'{_FF_FACTOR_ABBR[fx]} $\\times$ {_FF_FACTOR_ABBR[ft]} ({"/".join(who)})', fontsize=7)
            ax.set_xlabel(_FF_FACTOR_NAMES[fx], 
                          fontsize=6.5)
            
            ax.legend(handles=[Line2D([], [], 
                                      color=_FF_SUB, 
                                      ls='-' if i == 0 else '--', 
                                      marker='o' if i == 0 else '^', 
                                      ms=3,
                                      mfc=_FF_SUB if i == 0 else 'white',
                                      label=f'{_FF_FACTOR_ABBR[ft]}: {_FF_LEVEL_LABELS[ft][tv]}')
                               for i, tv in enumerate(level_order[ft])],
                      fontsize=5.5, 
                      frameon=False, 
                      loc='upper left', 
                      handlelength=1.6, 
                      borderpad=0.2, 
                      labelspacing=0.2)
            
        for ax in axs.ravel()[len(pairs):]:
            ax.axis('off')
        for row in range(n_rows):
            axs[row, 0].set_ylabel('RMS $\\Delta$phase (deg)')
        fig.legend(handles=[Line2D([], [], 
                                   color=_FF_STUDY_COLORS[s], marker=_FF_STUDY_MARKERS[s], 
                                   ms=3.5, 
                                   lw=1.1, 
                                   label=s)
                            for s in _FF_STUDIES],
                   loc='lower center', 
                   ncol=2, 
                   frameon=False, 
                   bbox_to_anchor=(0.5, -0.06))

        # Save the figure
        _ff_save(fig, out_dir, 'fig_interactions_shared')


def _plot_ff_half_normal(effects: dict, 
                         out_dir: Path) -> None:
    """This function generates half-normal plots for factorial effects, highlighting the most significant effects beyond Lenth's margin of error.

    Args:
        effects (dict): 
            {(response, study): (effects table sorted by |effect| with 'q', 'abs', 'sig', 'term', margin of error)}
            from _ff_half_normal_effects().
        out_dir (Path): 
            Output folder.
    """
    # Generate half-normal plots for each response and study combination.
    with plt.rc_context(_FF_RC):
        
        fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
        
        # Loop over each response and study (reflection and transmission)
        for i, (response, title) in enumerate(_FF_HALF_NORMAL_RESPONSES):
            for j, study in enumerate(_FF_STUDIES):
                
                ax = axs[i, j]
                E, me = effects[(response, study)]
                color = _FF_STUDY_COLORS[study]

                _ff_style_axis(ax)

                ax.axhline(me, color=_FF_INK, lw=0.7, ls='--')
                ax.text(0.03, me * 1.06, f'ME = {me:.3g}', transform=ax.get_yaxis_transform(), fontsize=6, va='bottom')
                pos = E[E['abs'] > 0]

                # Draw the non-significant effects first and significant effects on top
                ax.scatter(pos.q[~pos.sig], pos['abs'][~pos.sig], 
                           s=9, 
                           color=color, 
                           alpha=0.45, 
                           edgecolor='none')

                ax.scatter(pos.q[pos.sig], 
                           pos['abs'][pos.sig], 
                           s=16, color=color, 
                           edgecolor='white', 
                           linewidth=0.4, 
                           zorder=3)

                # Annotate the top 5 significant effects
                for k, (_, t) in enumerate(E[E.sig].tail(5).iloc[::-1].iterrows()):
                    up = (k % 2 == 0)
                    ax.annotate(t.term, (t.q, t['abs']), 
                                xytext=(0, 5 if up else -5), 
                                textcoords='offset points', 
                                ha='center',
                                va='bottom' if up else 'top', 
                                fontsize=5.8, 
                                color=_FF_INK)
                    
                ax.set_yscale('log')
                ax.set_ylim(pos['abs'].min() * 0.7, E['abs'].max() * 2.5)
                ax.set_xlim(0, E.q.max() * 1.06)
                ax.set_title(f'({"abcd"[i * 2 + j]}) {study}: {title}', loc='left')
                ax.set_xlabel('Half-normal quantile' if i == 1 else '')
                ax.set_ylabel('|effect| (log scale)' if j == 0 else '')
        
        # Save
        _ff_save(fig, out_dir, 'fig_half_normal')


def _plot_mimo_channel_sensitivity(results_df, 
                                   response_metric, 
                                   n_scored, 
                                   out_dir) -> None:
    """This function generates a 3-panel plot for MIMO channel sensitivity analysis (response metric averaged over all DOE runs; draw-to-draw std; and split by material, if present).
    
    Args:
        results_df (pd.DataFrame): 
            DataFrame containing the results of the MIMO channel sensitivity analysis.
        response_metric (str): 
            The response metric to plot.
        n_scored (int): 
            Number of DOE runs scored.
        out_dir (str): 
            Directory to save the generated plot.
    
    Returns:
        None

    Raises:
        None
    """
    
    metric_col = f'{response_metric}_mean'
    std_col = f'{response_metric}_std'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Draw first Panel (expected detectability vs. channel count)
    ax = axes[0]
    grouped = results_df.groupby('n_channels')[metric_col].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_index()
    sem = grouped['std'] / np.sqrt(grouped['count'])
    
    ax.errorbar(grouped.index, grouped['mean'], 
                yerr=sem, 
                marker='o', 
                capsize=4,
                color='#1f77b4')
    
    ax.set_xlabel('Number of MIMO channels combined (of 25)')
    ax.set_ylabel(f'{response_metric} (mean over {n_scored} DOE runs)')
    ax.set_title('(a) Expected detectability vs. channel count')
    ax.grid(True, alpha=0.3)

    # Draw second panel (channel-selection risk vs. channel count)
    ax_std = axes[1]
    grouped_std = results_df.groupby('n_channels')[std_col].mean().sort_index()
    
    ax_std.plot(grouped_std.index, 
                grouped_std.values, 
                marker='o', 
                color='#d62728')
    
    ax_std.set_xlabel('Number of MIMO channels combined (of 25)')
    ax_std.set_ylabel(f'std. dev. of {response_metric} across random channel subsets')
    ax_std.set_title('(b) Channel-selection risk vs. channel count')
    ax_std.grid(True, alpha=0.3)

    # Draw third panel (expected detectability vs. channel count, by material)
    ax2 = axes[2]
    if 'material' in results_df.columns:
        for material, sub in results_df.groupby('material'):
            g = sub.groupby('n_channels')[metric_col].mean().sort_index()
            ax2.plot(g.index, 
                     g.values, 
                     marker='o', 
                     label=str(material))
        ax2.legend(title='material')
        ax2.set_title('(c) Expected detectability vs. channel count, by material')
    else:
        ax2.axis('off')

    ax2.set_xlabel('Number of MIMO channels combined (of 25)')
    ax2.set_ylabel(f'{response_metric} (mean)')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('MIMO channel-count sensitivity (coherent combining of the existing 5x5 virtual array)')
    fig.tight_layout()

    # Save
    out_png_path = out_dir / 'mimo_channel_sensitivity.png'
    _savefig(fig, out_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"analyze_mimo_channel_sensitivity(): wrote '{out_png_path}'.")


def _plot_mimo_channel_draw_surface(results_df, 
                                    response_metric, 
                                    n_scored, 
                                    out_dir) -> None:
    """This function generates a surface plot showing the sensitivity of the response metric to the number of MIMO channels combined and the draw index.

    Args:
        results_df: 
            DataFrame containing the simulation results.
        response_metric: 
            The response metric to plot.
        n_scored: 
            Number of scored DOE runs.
        out_dir: 
            Output directory
    
    Returns:
        None.

    Raises:
        None.
    """
    
    metric_col = f'{response_metric}_mean'
    std_col = f'{response_metric}_std'

    # Determine which channel counts have more than one draw
    draws_per_n = results_df.groupby('n_channels')['draw_index'].nunique()
    n_list = sorted(draws_per_n[draws_per_n > 1].index)
    if not n_list:
        print("_plot_mimo_channel_draw_surface(): every channel count has only one possible draw (nothing random to plot) - skipping.")
        return
    plot_df = results_df[results_df['n_channels'].isin(n_list)]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # Define helper functions for pivoting and plotting the envelope panel
    def _pivot(col):
        return plot_df.pivot(index='draw_index', 
                             columns='n_channels', 
                             values=col).reindex(columns=n_list)

    # Define helper function for plotting the envelope panel (one line for each draw index + min-max envelope + mean)
    def _envelope_panel(ax, piv, title, ylabel):

        x = np.arange(len(n_list))

        # Plot one line per draw index
        for _, row in piv.iterrows():
            ax.plot(x, row.values, 
                    color='#1f77b4', 
                    alpha=0.08,
                     linewidth=0.8, 
                     zorder=1)

        # The stats
        lo = piv.min(axis=0).values
        hi = piv.max(axis=0).values
        mean = piv.mean(axis=0).values
        
        ax.fill_between(x, lo, hi, color='#1f77b4', alpha=0.15, zorder=2,
                         label='min-max envelope across draws')
        ax.plot(x, mean, color='#d62728', linewidth=2, marker='o', zorder=3,
                label='mean across draws')
        ax.set_xticks(x)
        ax.set_xticklabels(n_list)
        ax.set_xlabel('Number of MIMO channels combined (of 25)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        ax.legend(fontsize=8, loc='best')
        
        ax.grid(True, alpha=0.3)

    # Pivot the metric and std columns and plot the corresponding envelope panels
    piv_mean = _pivot(metric_col)
    _envelope_panel(axes[0], piv_mean,
                    f'(a) {response_metric}, mean over {n_scored} DOE runs\n({piv_mean.shape[0]} random draws per N)',
f'{response_metric} (mean over runs)')

    piv_std = _pivot(std_col)
    _envelope_panel(
        axes[1], piv_std,
        f'(b) {response_metric}, std across DOE runs\n({piv_std.shape[0]} random draws per N)',
        f'{response_metric} (std across runs)')

    # Plot the draw spread (std across random draws) for the mean metric
    ax2 = axes[2]
    draw_spread = piv_mean.std(axis=0)
    ax2.plot(n_list, draw_spread.values, marker='o', color='#2ca02c')
    ax2.set_xlabel('Number of MIMO channels combined (of 25)')
    ax2.set_ylabel(f'std., across draws, of the {n_scored}-run-averaged '
                    f'{response_metric}')
    ax2.set_title("(c) How much would a different random draw\nhave changed the answer?")
    ax2.grid(True, alpha=0.3)

    fig.suptitle('MIMO channel-subset draw-stability surface '
                  '(same random channel subsets scored across every DOE run)')
    fig.tight_layout()
    
    # Save figure
    out_png_path = out_dir / 'mimo_channel_draw_surface.png'
    _savefig(fig, out_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"_plot_mimo_channel_draw_surface(): wrote '{out_png_path}'.")


def plot_gaussian_pulse(frequency_range=(77e9, 81e9), 
                        f_center=None, 
                        amplitude: float = 1.0, 
                        dt: float = None,
                        time_window: float = 2e-9, 
                        out_path=None, 
                        show: bool = True) -> dict:
    """This function generates a figure showing the Gaussian excitation pulse in both the time and frequency domains.

    Args:
        frequency_range (tuple): 
            (f_low, f_high) in Hz of the band
        f_center (float): 
            Center frequency f_c in Hz
        amplitude (float): 
            Pulse amplitude.
        dt (float): 
            Time step in s
        time_window (float): 
            Time window of the sampled signal in s
        out_path (str or Path): 
            Output path
        show (bool): 
            Flag for showing the figure.

    Returns:
        dict: f_center_hz, t0_s, sigma_s, band_rolloff_db (|P| at f_high relative to f_low), band_drop_db (|P| at f_high
        relative to the DC peak), max_fft_vs_analytic_db (largest FFT-analytic deviation inside the band) and out_path.

    Raises:
        None
    """
    # Get frequency and time parameters
    f_low, f_high = float(frequency_range[0]), float(frequency_range[1])
    f_c = (f_low + f_high) / 2 if f_center is None else float(f_center)
    dt = 1 / (100 * f_high) if dt is None else float(dt)

    # Pulse parameters
    t0 = 1 / f_c
    sigma = 1 / (2 * np.pi * f_c)
    n_time = int(round(time_window / dt)) + 1
    t = np.arange(n_time) * dt
    p = amplitude * np.exp(-2 * np.pi**2 * f_c**2 * (t - t0)**2)

    # Compute spectrum via FFT
    freqs = np.fft.rfftfreq(n_time, d=dt)
    P_fft = np.abs(np.fft.rfft(p)) * dt
    P_ana = amplitude * sigma * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi**2 * sigma**2 * freqs**2)
    P_peak = amplitude * sigma * np.sqrt(2 * np.pi)
    tiny = np.finfo(float).tiny
    fft_db = 20 * np.log10(np.maximum(P_fft, tiny) / P_peak)
    ana_db = 20 * np.log10(np.maximum(P_ana, tiny) / P_peak)

    # Define helper function to compute the spectrum in dB at a given frequency
    def ana_db_at(f):
        return 20 * np.log10(np.exp(-2 * np.pi**2 * sigma**2 * f**2))

    # Extract frequency band of interest and maximum deviation from the analytic spectrum
    band = (freqs >= f_low) & (freqs <= f_high)
    max_dev_db = float(np.max(np.abs(fft_db[band] - ana_db[band]))) if band.any() else float('nan')
    rolloff_db = ana_db_at(f_high) - ana_db_at(f_low)

    # Plot the time and frequency domain representations of the Gaussian pulse. Set style parameters for journal figures
    ink, grid = '#1a1a19', '#e4e3dc'
    c_main = _COOKIE_COLORS[0]
    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7,
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5, 
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42}
    
    # Apply the plotting style parameters within this context
    with plt.rc_context(rc):

        fig, (ax_t, ax_f) = plt.subplots(1, 2, 
                                         figsize=(252 / 72, 2.0), constrained_layout=True)

        # Plot time representation of the Gaussian pulse
        ax_t.plot(t * 1e12, p / amplitude, 
                  color=c_main, 
                  lw=1.0, 
                  zorder=3)
        ax_t.axvline(t0 * 1e12, 
                     color=ink, 
                     lw=0.5, 
                     ls=':', 
                     zorder=2)
        half = np.exp(-0.5)
        ax_t.annotate('', 
                      xy=((t0 - sigma) * 1e12, half), 
                      xytext=((t0 + sigma) * 1e12, half),
                      arrowprops=dict(arrowstyle='<->', 
                                      lw=0.6, 
                                      color=ink, 
                                      shrinkA=0, 
                                      shrinkB=0))
        ax_t.text(t0 * 1e12, half - 0.06, 
                  r'$2\sigma$', 
                  ha='center', 
                  va='top', 
                  fontsize=6, 
                  color=ink)
        ax_t.text(t0 * 1e12 + 0.6, 1.04, 
                  r'$t_0$', 
                  ha='left', 
                  va='bottom', 
                  fontsize=6, 
                  color=ink)
        ax_t.set_xlim(0, 2 * t0 * 1e12)
        ax_t.set_ylim(0, 1.18)
        ax_t.set_xlabel('Time / ps')
        ax_t.set_ylabel(r'$p(t)\,/\,a$')
        ax_t.set_title('(a) Time domain')

        # Plot frequency representation of the Gaussian pulse in dB relative to the DC maximum
        f_max_plot = 3 * f_high
        sel = freqs <= f_max_plot
        ax_f.plot(freqs[sel] / 1e9, 
                  fft_db[sel], 
                  color=c_main, 
                  lw=1.0, 
                  zorder=3)
        ax_f.plot(freqs[sel] / 1e9, ana_db[sel], 
                  color=ink, 
                  lw=0.7, 
                  ls=(0, (3, 2)), 
                  zorder=4)
        ax_f.set_xlim(0, f_max_plot / 1e9)
        ax_f.set_ylim(max(-60.0, float(ana_db[sel].min()) - 3), 3)
        ax_f.set_xlabel('Frequency / GHz')
        ax_f.set_ylabel(r'$|P(f)|\,/\,|P(0)|$ / dB')
        ax_f.set_title('(b) Frequency domain')

        # Apply grid and style settings to both axes
        for ax in (ax_t, ax_f):
            ax.grid(True, axis='y', color=grid, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

        handles = [Line2D([], [], 
                          color=c_main, 
                          lw=1.0, 
                          label='Sampled pulse (FFT)'),
                   Line2D([], [], 
                          color=ink, 
                          lw=0.7, 
                          ls=(0, (3, 2)), 
                          label='Analytic')]
        
        fig.legend(handles=handles, 
                   loc='outside upper center', 
                   ncol=2, 
                   frameon=False,
                   columnspacing=1.0, 
                   handlelength=1.6, 
                   handletextpad=0.4)

        # Save and show
        written = None
        if out_path is not None:
            written = _savefig(fig, out_path, dpi=300)
            print(f"plot_gaussian_pulse(): wrote '{written}'.")
        
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)

    result = {'f_center_hz': f_c, 
              't0_s': t0, 
              'sigma_s': sigma,
              'band_rolloff_db': rolloff_db, 
              'band_drop_db': ana_db_at(f_high),
              'max_fft_vs_analytic_db': max_dev_db, 
              'out_path': written}

    # Print summary
    print(f"plot_gaussian_pulse(): f_c = {f_c/1e9:.1f} GHz, t0 = {t0*1e12:.2f} ps, sigma = {sigma*1e12:.3f} ps, |P| over {f_low/1e9:g}-{f_high/1e9:g} GHz: {ana_db_at(f_low):.2f} to {ana_db_at(f_high):.2f} dB rel. DC (roll-off {rolloff_db:.2f} dB), FFT vs analytic in band: {max_dev_db:.3f} dB.")
    
    return result


def plot_mimo_tx_rx_geometry(parms, 
                             out_path=None, 
                             show: bool = True) -> dict:
    """ This function creates a figure showing the MIMO TX/RX array geometry.

    Args:
        parms (dict or str or Path): 
            Parameter dict or path of a parameter .json file.
        out_path (str or Path): 
            Output path.
        show (bool): 
            Show the figure.

    Returns:
        dict: tx_x_mm, rx_x_mm, d_tx_mm, d_rx_mm, half_lambda_mm, virtual_spacing_mm (mean, std), z_tx_mm,
        z_rx_reflection_mm, z_rx_transmission_mm, out_path.

    Raises:
        None
    """
    # Load parameters from JSON file if a path is provided.
    if not isinstance(parms, dict):
        with open(parms, encoding='utf-8') as f:
            parms = json.load(f)

    # Helper functions to generate x and y coordinates from the specification.
    def _x(spec, n):
        return np.linspace(spec[0][0], spec[0][-1], n) * 1e3

    def _y(spec, n):
        return (np.full(n, spec[1]) if np.ndim(spec[1]) == 0 else np.linspace(spec[1][0], spec[1][-1], n)) * 1e3

    # Extract number of TX and RX elements, their coordinates, positions along z-axis, and spacing.
    n_tx, n_rx = int(parms['N_TX']), int(parms['N_RX'])
    tx_x, rx_x = _x(parms['TX_POS_RANGE'], n_tx), _x(parms['RX_POS_RANGE'], n_rx)
    tx_y, rx_y = _y(parms['TX_POS_RANGE'], n_tx), _y(parms['RX_POS_RANGE'], n_rx)
    z_tx = float(parms['TX_POS_RANGE'][2]) * 1e3
    z_rx = float(parms['RX_POS_RANGE'][2]) * 1e3
    z_refl, z_trans = (z_rx, -z_rx) if np.sign(z_rx) == np.sign(z_tx) else (-z_rx, z_rx)

    # Extract highest frequency and calculate half wavelength in mm.
    f_high = float(parms['FREQUENCY_RANGE'][-1])
    half_lambda_mm = sp.constants.c / f_high / 2 * 1e3
    d_tx = float(np.abs(np.diff(tx_x)).mean()) if n_tx > 1 else float('nan')
    d_rx = float(np.abs(np.diff(rx_x)).mean()) if n_rx > 1 else float('nan')
    v = np.sort((tx_x[:, None] + rx_x[None, :]).ravel())
    dv = np.diff(v) if v.size > 1 else np.array([np.nan])

    # Extract SUT parameters and belt thickness.
    sut = parms['SUT']
    off = [o * 1e3 for o in parms.get('SUT_OFFSET', [0.0, 0.0])]
    if sut['SHAPE'] == 'CYLINDER':
        sut_half = (sut['SIZE']['DIAMETER'] * 1e3 / 2,) * 2
        sut_thick = sut['SIZE']['THICKNESS'] * 1e3
    else:
        sut_half = (sut['SIZE'][0] * 1e3 / 2, sut['SIZE'][1] * 1e3 / 2)
        sut_thick = sut['SIZE'][2] * 1e3
    belt_thick = float(parms.get('BELT', {}).get('THICKNESS', 0.0)) * 1e3

    ink, muted, grid = '#1a1a19', '#6b6a63', '#e4e3dc'
    c_tx, c_rx, c_sut, e_sut, c_belt = '#d63a2f', _COOKIE_COLORS[0], '#d9c9a3', '#8a7248', '#9a9a94'
    mk = dict(ls='none', ms=4.5, mew=0.4, zorder=4)
    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7,
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5, 
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 'pdf.fonttype': 42}
    
    # Determine x-axis limits for the plot.
    x_lo = min(float(tx_x.min()), off[0] - sut_half[0]) - 4
    x_hi = max(float(rx_x.max()), off[0] + sut_half[0]) + 4
    rx_left, rx_right = int(np.argmin(rx_x)), int(np.argmax(rx_x))

    # Plot the scene with the specified parameters.
    with plt.rc_context(rc):

        fig, (ax, ax_s) = plt.subplots(2, 1, 
                                       figsize=(252 / 72, 3.9), constrained_layout=True, 
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [1.25, 1.0]})

        # Plot the top view of the scene. Start with SUT.
        if sut['SHAPE'] == 'CYLINDER':
            ax.add_patch(Circle((off[0], off[1]), 
                                sut_half[0], 
                                facecolor=c_sut, 
                                edgecolor=e_sut, 
                                lw=0.6, 
                                alpha=0.6,
                                zorder=1))
        else:
            ax.add_patch(Rectangle((off[0] - sut_half[0], 
                                    off[1] - sut_half[1]), 
                                    2 * sut_half[0], 2 * sut_half[1],
                                    facecolor=c_sut, 
                                    edgecolor=e_sut, 
                                    lw=0.6, 
                                    alpha=0.6, 
                                    zorder=1))
        ax.axvline(off[0], color=ink, lw=0.5, ls=':', zorder=2)

        # Plot the TX and RX positions on the scene.
        ax.plot(tx_x, tx_y, 
                marker='^', 
                color=c_tx, 
                mec='white', **mk)
        ax.plot(rx_x, rx_y, 
                marker='v', 
                color=c_rx, 
                mec='white', **mk)

        # Annotate the TX and RX positions
        for i, x in enumerate(tx_x):
            ax.text(x, tx_y[i] + 2.6, f'TX{i}', ha='center', va='bottom', fontsize=5.5, color=c_tx)

        y_lab = float(rx_y.min()) - 2.6
        ax.text(rx_x[rx_left] + 0.3, y_lab, 
                f'RX{rx_left}', 
                ha='right', 
                va='top', 
                fontsize=5.5, 
                color=c_rx)
        ax.text(rx_x[rx_right] - 0.3, y_lab, 
                f'RX{rx_right}', 
                ha='left', 
                va='top', 
                fontsize=5.5, 
                color=c_rx)

        y_top = float(max(tx_y.max(), rx_y.max())) + 9.0

        # Annotate the TX array spacing
        if n_tx > 1:
            ts = np.sort(tx_x)
            for x0, x1 in zip(ts[:-1], ts[1:]):
                ax.annotate('', xy=(x1, y_top), 
                            xytext=(x0, y_top),
                            arrowprops=dict(arrowstyle='<->', 
                                            lw=0.5, 
                                            color=c_tx, 
                                            shrinkA=0, 
                                            shrinkB=0))
            ax.text(float(ts.mean()), 
                    y_top + 1.5, rf'$d_\mathrm{{TX}} = N_\mathrm{{RX}}\,\lambda/2 = {d_tx:.2f}$ mm',
                    ha='center', 
                    va='bottom', 
                    fontsize=6, 
                    color=c_tx)
        
        y_bot = float(min(tx_y.min(), rx_y.min())) - 10.0

        # Annotate the RX array spacing
        if n_rx > 1:
            rs = np.sort(rx_x)
            for x0, x1 in zip(rs[:-1], rs[1:]):
                ax.annotate('', xy=(x1, y_bot), xytext=(x0, y_bot),
                            arrowprops=dict(arrowstyle='|-|, widthA=0.25, widthB=0.25', lw=0.5, color=c_rx,
                                            shrinkA=0, shrinkB=0))
            ax.text(float(rs.max()) + 1.5, y_bot, rf'$d_\mathrm{{RX}} = \lambda/2 = {d_rx:.2f}$ mm', ha='left',
                    va='center', fontsize=6, color=c_rx)
            
        y_ext = max(sut_half[1], abs(y_top) + 4, abs(y_bot) + 3) + 1
        
        ax.set_xlim(x_lo, x_hi)
        ax.update_datalim([(x_lo, -y_ext), (x_hi, y_ext)])
        ax.autoscale_view(scalex=False)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_ylabel('$y$ / mm')
        ax.set_title('(a) Top view')

        # Plot the side view of the setup
        ax_s.axhline(-max(belt_thick, 1.0) / 2, color=c_belt, lw=1.6, zorder=1)
        ax_s.add_patch(Rectangle((off[0] - sut_half[0], 0.0), 
                                2 * sut_half[0], 
                                sut_thick, 
                                facecolor=c_sut,
                                edgecolor=e_sut, 
                                lw=0.6, 
                                alpha=0.9, 
                                zorder=2))
        ax_s.axvline(off[0], color=ink, lw=0.5, ls=':', zorder=1)
        
        # Draw rays from TX array to sample center to RX array centers
        tx_c, rx_c = float(tx_x.mean()), float(rx_x.mean())
        for zr in (z_refl, z_trans):
            ax_s.plot([tx_c, off[0], rx_c], [z_tx, sut_thick / 2, zr], 
                      color=muted, 
                      lw=0.5, 
                      ls=(0, (1, 1.5)), 
                      zorder=1)
        
        tx_mark = 'v' if z_tx > 0 else '^'
        ax_s.plot(tx_x, np.full(n_tx, z_tx), 
                  marker=tx_mark, 
                  color=c_tx, 
                  mec='white', **mk)
        ax_s.plot(rx_x, np.full(n_rx, z_refl), 
                  marker='v' if z_refl > 0 else '^', 
                  color=c_rx, 
                  mec='white', **mk)
        ax_s.plot(rx_x, np.full(n_rx, z_trans), 
                  marker='v' if z_trans > 0 else '^', 
                  color='white', 
                  mec=c_rx, **mk)
        
        dz = 0.17 * abs(z_tx)
        ax_s.text(tx_c, z_tx + np.sign(z_tx) * dz, 
                  'TX array', 
                  ha='center', 
                  va='center', 
                  fontsize=5.5, 
                  color=c_tx)
        ax_s.text(rx_c, z_refl + np.sign(z_refl) * dz, 
                  'RX, reflection', 
                  ha='center', 
                  va='center', 
                  fontsize=5.5,
                  color=c_rx)
        ax_s.text(rx_c, z_trans + np.sign(z_trans) * dz, 
                  'RX, transmission', 
                  ha='center', 
                  va='center', 
                  fontsize=5.5,
                  color=c_rx)

        # Annotate distances for reflection and transmission paths
        ax_s.annotate('', xy=(x_lo + 3, z_refl), 
                      xytext=(x_lo + 3, 0),
                      arrowprops=dict(arrowstyle='<->', 
                                      lw=0.5, 
                                      color=ink, 
                                      shrinkA=0, 
                                      shrinkB=0))
        ax_s.annotate('', xy=(x_lo + 3, z_trans), 
                      xytext=(x_lo + 3, 0),
                      arrowprops=dict(arrowstyle='<->', 
                                      lw=0.5, 
                                      color=ink, 
                                      shrinkA=0, 
                                      shrinkB=0))
        
        ax_s.text(x_lo + 4, z_refl / 2, f'{abs(z_refl):g} mm', 
                  ha='left', 
                  va='center', 
                  fontsize=5.5, 
                  color=ink)
        ax_s.text(x_lo + 4, z_trans / 2, f'{abs(z_trans):g} mm', 
                  ha='left', 
                  va='center', 
                  fontsize=5.5, 
                  color=ink)
        z_ext = 1.3 * max(abs(z_tx), abs(z_refl))
        ax_s.set_ylim(-z_ext, z_ext)
        ax_s.set_xlim(x_lo, x_hi)
        ax_s.set_ylabel('$z$ / mm')
        ax_s.set_xlabel('$x$ offset from sample center / mm')
        ax_s.set_title('(b) Side view ($z$ not to scale)')

        # Apply grid and spine settings to both main and side view axes
        for a in (ax, ax_s):
            a.grid(True, axis='x', color=grid, lw=0.5, zorder=0)
            a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)

        # Define legend
        handles = [Line2D([], [], 
                          marker='^', 
                          color=c_tx, 
                          mec='white', 
                          label=f'TX ({n_tx})', 
                          **mk),
                   Rectangle((0, 0), 1, 1, 
                            facecolor=c_sut, 
                            edgecolor=e_sut, 
                            lw=0.6, 
                            alpha=0.6, 
                            label='Sample'),
                   Line2D([], [], 
                          marker='v', 
                          color=c_rx, 
                          mec='white', 
                          label=f'RX reflection ({n_rx})', **mk),
                   Line2D([], [], 
                          color=c_belt, 
                          lw=1.6, 
                          label='Belt'),
                   Line2D([], [], 
                          marker='^', 
                          color='white', 
                          mec=c_rx, 
                          label=f'RX transmission ({n_rx})', **mk)]
        
        fig.legend(handles=handles, 
                   loc='outside upper center', 
                   ncol=3, 
                   frameon=False, 
                   columnspacing=1.0,
                   handletextpad=0.4)

        # Save and plot
        written = None
        if out_path is not None:
            written = _savefig(fig, out_path, dpi=300)
            print(f"plot_mimo_tx_rx_geometry(): wrote '{written}'.")
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)

    result = {'tx_x_mm': tx_x, 
              'rx_x_mm': rx_x, 
              'd_tx_mm': d_tx, 
              'd_rx_mm': d_rx, 
              'half_lambda_mm': half_lambda_mm,
              'virtual_spacing_mm': (float(np.mean(dv)), float(np.std(dv))), 'z_tx_mm': z_tx,
              'z_rx_reflection_mm': z_refl, 
              'z_rx_transmission_mm': z_trans, 
              'out_path': written}
    
    print(f"plot_mimo_tx_rx_geometry(): {n_tx} TX, {n_rx} RX, d_TX = {d_tx:.4f} mm (target {n_rx * half_lambda_mm:.4f}), d_RX = {d_rx:.4f} mm (target lambda/2 = {half_lambda_mm:.4f} at {f_high / 1e9:g} GHz), virtual spacing = {np.mean(dv):.4f} +/- {np.std(dv):.1e} mm, z: TX {z_tx:g}, RX refl. {z_refl:g}, RX trans. {z_trans:g} mm.")
    return result


def plot_camera_measured_vs_simulated(study_dir, 
                                      out_path=None, 
                                      sheets=(0, 10, 20), 
                                      full_scale: float = 1.0,
                                      display: str = 'fixed', 
                                      cmap: str = 'jet', 
                                      marker_shift_x: float = 0.07,
                                      show: bool = True) -> dict:
    """This function generates a figure comparing measured and simulated THz line-camera images for different attenuation levels.

    Args:
        study_dir (str or Path):
            Study folder
        out_path (str or Path):
            Output path of folder
        sheets (tuple):
            Numbers of paper sheets (attenuation).
        full_scale (float):
            Color scale 0 .. full_scale of the simulated images for display == 'fixed'.
        display (str):
            'fixed' or 'auto'
        cmap (str):
            Colormap of the simulated images, cut to the red (belt) and darkest blue of the first reference image.
        marker_shift_x (float):
            Shift of the FO marker on the measured images, fraction of the image width (positive = right).
        show (bool):
            Flag for showing the figure

    Returns:
        dict: sheets, measured (file per column), px_per_mm, out_path.

    Raises:
        FileNotFoundError: If the expected reference images or simulation results are not found.
    """
    # Set paths and load data
    study_dir = Path(study_dir)
    ref_dir, res_dir = study_dir / 'references', study_dir / 'results_analysis'
    npz_path = res_dir / 'moving_sample_camera_images.npz'

    # Check if simulation results exist and load them
    if not npz_path.exists():
        raise FileNotFoundError(f"plot_camera_measured_vs_simulated(): '{npz_path}' missing - run analyze_moving_sample_camera() first.")
    sim = np.load(npz_path)
    with open(res_dir / 'scan_config.json') as fh:
        cfg = json.load(fh)
    
    # Load measured reference images for each sheet number
    measured = {}
    for n in sheets:
        hits = sorted(ref_dir.glob(f'*_{int(n)}p.png'))
        if len(hits) != 1:
            raise FileNotFoundError(f"plot_camera_measured_vs_simulated(): expected one '*_{int(n)}p.png' in '{ref_dir}', found {len(hits)}.")
        measured[int(n)] = hits[0]

    # Define helper function to compute edges from coordinates
    def edges(v):
        mid = (v[1:] + v[:-1]) / 2
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    
    # Get the edges, FO size etc.
    xe, ye = edges(np.asarray(sim['x_mm'], float)), edges(np.asarray(sim['y_mm'], float))
    fx, fy, _ = cfg['fo_center_cookie_frame_mm']
    fo_size = float(cfg['fo_size_mm'])

    # Read the first reference image and create a mask for non-red regions that serves as a bounding box
    a = plt.imread(measured[int(sheets[0])])[..., :3]
    m = ~((a[..., 0] > 0.8) & (a[..., 1] < 0.35) & (a[..., 2] < 0.35))
    m = sp.ndimage.binary_fill_holes(sp.ndimage.binary_closing(sp.ndimage.binary_opening(m, iterations=2), iterations=6))
    lab, n_lab = sp.ndimage.label(m)
    
    py, px = np.where(lab == int(np.argmax(sp.ndimage.sum(m, lab, range(1, n_lab + 1)))) + 1)
    ol = np.asarray(cfg['sut_outline_mm'], float)
    px_per_mm = ((px.max() - px.min()) / np.ptp(ol[0]) + 
                 (py.max() - py.min()) / np.ptp(ol[1])) / 2
    cx_px, cy_px = (px.min() + px.max()) / 2, (py.min() + py.max()) / 2
    
    print(f"plot_camera_measured_vs_simulated(): reference scale {px_per_mm:.2f} px/mm, cookie center at pixel ({cx_px:.0f}, {cy_px:.0f}).")

    # Compute the reference red and blue colors for color mapping
    rgb = a.reshape(-1, 3)
    c_red = np.median(rgb[(rgb[:, 0] > 0.8) & (rgb[:, 1] < 0.35) & (rgb[:, 2] < 0.35)], axis=0)
    blue = rgb[(rgb[:, 2] > rgb[:, 0] + 0.2) & (rgb[:, 2] > rgb[:, 1] + 0.2)]
    c_blue = np.median(blue[blue.sum(axis=1) <= np.percentile(blue.sum(axis=1), 2)], axis=0)
    base = plt.get_cmap(cmap)
    t = np.linspace(0, 1, 2001)
    lut = base(t)[:, :3]
    t_lo, t_hi = (float(t[np.argmin(np.linalg.norm(lut - c, axis=1))]) for c in (c_blue, c_red))
    cmap_sim = LinearSegmentedColormap.from_list(f'{base.name}_ref', 
                                                 base(np.linspace(t_lo, t_hi, 256)))
    
    print(f"plot_camera_measured_vs_simulated(): {base.name} cut to {t_lo:.3f} .. {t_hi:.3f} (reference blue {np.round(c_blue, 2)}, red {np.round(c_red, 2)}).")

    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7, 
          'axes.labelsize': 7,
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42}

    # Apply the custom rc settings for the plot context
    with plt.rc_context(rc):

        fig, axs = plt.subplots(2, len(sheets), 
                                figsize=(252 / 72, 2.3), constrained_layout=True, 
                                squeeze=False)

        # Loop over each sheet and plot the measured and simulated images side by side.
        for j, n in enumerate(sheets):
            
            ax_m, ax_s = axs[0, j], axs[1, j]
            img_m = plt.imread(measured[int(n)])
            ax_m.imshow(img_m, aspect='equal', interpolation='nearest')

            # Draw a circle indicating the FO footprint on the measured image.
            ax_m.add_patch(plt.Circle((cx_px + fx * px_per_mm + marker_shift_x * img_m.shape[1], cy_px - fy * px_per_mm), 
                                      (fo_size / 2 + 2.0) * px_per_mm, 
                                       fill=False, 
                                       ec='k', 
                                       lw=0.8))
            img = sim[f'fo_{int(n)}p']
            vmin, vmax = (0.0, full_scale) if display == 'fixed' else (float(img.min()), float(img.max()))
            ax_s.pcolormesh(xe, ye, img, cmap=cmap_sim, vmin=vmin, vmax=max(vmax, vmin + 1e-12), shading='flat')

            
            # Draw a circle indicating the FO footprint on the simulated image.
            ax_s.add_patch(plt.Circle((fx, fy), 
                                      fo_size / 2 + 2.0, 
                                      fill=False, 
                                      ec='k', 
                                      lw=0.8))
            ax_s.set_aspect('equal')
            
            # Make the simulated image be equally large as the measured image.
            h_px, w_px = img_m.shape[:2]
            ax_s.set_xlim((-0.5 - cx_px) / px_per_mm, 
                          (w_px - 0.5 - cx_px) / px_per_mm)
            ax_s.set_ylim((cy_px + 0.5 - h_px) / px_per_mm, 
                          (cy_px + 0.5) / px_per_mm)
            ax_m.set_title(f'{int(n)}p')
            
            for ax in (ax_m, ax_s):
                ax.set_xticks([]); ax.set_yticks([])
        
        axs[0, 0].set_ylabel('Measured')
        axs[1, 0].set_ylabel('Simulated')

        # Save and plot
        written = None
        if out_path is not None:
            written = _savefig(fig, out_path, dpi=300)
            print(f"plot_camera_measured_vs_simulated(): wrote '{written}'.")
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
    
    return {'sheets': list(map(int, sheets)), 
            'measured': {k: str(v) for k, v in measured.items()}, 
            'px_per_mm': float(px_per_mm),
            'out_path': written}


def plot_freeform_cookie_geometry(sut: dict, 
                                  dx: float, 
                                  out_path=None, 
                                  show: bool = True, 
                                  fo: dict = None,
                                  cmap: str = 'viridis') -> dict:
    """This function visualizes the geometry of a freeform cookie, highlighting the top surface, regions, and any foreign objects.

    Args:
        sut (dict): 
            SUT dict with SHAPE 'FREEFORM'.
        dx (float): 
            Grid cell in m.
        out_path (str or Path): 
            Output path
        show (bool): 
            Show the figure without blocking; False = close it after saving.
        fo (dict): 
            Foreign object {'center_mm': [x, y, z] (cookie frame, z above the belt), 'size_mm', 'material', optional 'air_gap_mm' (hole from below: belt to FO top, this much wider per side)}; None = not drawn.
        cmap (str): 
            Colormap of the height map.

    Returns:
        dict: out_path (written file or None), fig_number (None if closed), n_columns, top_mm (min, max), region_bottom_mm (min per region).

    Raises:
        None
    """
    # Extract freeform geometry and compute
    ff = sut.get('FREEFORM', {})
    tile = max(1, int(round(float(ff.get('TILE', 0.0005)) / dx))) * dx
    col = _freeform_columns(sut, tile, dx)
    x, y = col['x'] * 1e3, col['y'] * 1e3
    inside = col['inside']
    top = np.where(inside, col['n_hi'] * dx * 1e3, np.nan)
    bot = np.where(inside, col['n_lo'] * dx * 1e3, np.nan)
    regions = ff.get('REGIONS', [])
    cx, cy = (np.asarray(regions[0].get('CENTER', (0.0, 0.0))) * 1e3) if regions else (0.0, 0.0)
    ix, iy = int(np.argmin(np.abs(x - cx))), int(np.argmin(np.abs(y - cy)))

    ink, grid = '#1a1a19', '#e4e3dc'
    c_sut, e_sut, c_belt, c_fo = '#d9c9a3', '#8a7248', '#9a9a94', '#3d3d3a'
    c_reg = ['#b3202a', '#7b3294', '#e08214']
    c_skin = '#a8864f'
    c_rim = '#6e5230'
    c_shell = '#4a6c8c'
    gap_ = float(fo.get('air_gap_mm', 0.0)) if fo is not None else 0.0
    rc = {'font.family': 'DejaVu Sans', 
          'font.size': 6.5, 
          'axes.titlesize': 7.5, 
          'axes.labelsize': 7,
          'xtick.labelsize': 6.5, 
          'ytick.labelsize': 6.5, 
          'legend.fontsize': 6.5, 
          'axes.titlelocation': 'left',
          'axes.linewidth': 0.6, 
          'pdf.fonttype': 42}

    # Compute layout parameters for the figure based on the geometry and number of legend entries.
    xr = float(np.ceil(np.max(np.abs(x)) + 1.0))
    yr = float(np.ceil(np.max(np.abs(y)) + 1.0))
    z_lo, z_hi = -1.0, float(np.ceil(np.nanmax(top) + 1.0))
    W, w = 252 / 72, 2.45
    h_a, h_c = w * (2 * yr) / (2 * xr), w * (z_hi - z_lo) / (2 * xr)
    n_leg = 2 + len(regions) + bool(col['skin']) + bool(col['rim']) + bool(col['shell']) + (fo is not None) + (bool(col['gaps']) or gap_ > 0)
    ncol_leg = n_leg if n_leg <= 6 else -(-n_leg // 2)
    gap_lab, gap_title, leg = 0.33, 0.22, (0.3 if n_leg <= 6 else 0.45)
    y_c = leg + gap_lab
    y_b = y_c + h_c + gap_title + gap_lab
    y_a = y_b + h_c + gap_title + gap_lab
    H = y_a + h_a + gap_title + 0.05
    left = (W - w) / 2

    # Create figure and axes
    with plt.rc_context(rc):

        fig = plt.figure(figsize=(W, H))

        ax = fig.add_axes([left / W, y_a / H, w / W, h_a / H])
        ax_x = fig.add_axes([left / W, y_b / H, w / W, h_c / H])
        ax_y = fig.add_axes([left / W, y_c / H, w / W, h_c / H])
        cax = fig.add_axes([(left + w + 0.07) / W, y_a / H, 0.07 / W, h_a / H])

        # Plot the top view of the surface
        im = ax.pcolormesh(x, y, top, 
                           cmap=cmap, 
                           shading='nearest', 
                           rasterized=True)
        ax.plot(*[v * 1e3 for v in _freeform_outline(sut)], 
                color=ink, 
                lw=0.6)
        for reg in regions:
            ax.plot(*[v * 1e3 for v in _freeform_region_outline(reg)], 
                    color='white', 
                    lw=0.6, 
                    ls=(0, (3, 1.5)))
        ax.axhline(y[iy], color='white', lw=0.4, ls=':')
        ax.axvline(x[ix], color='white', lw=0.4, ls=':')
        
        if fo is not None:
            fx, fy, _ = fo['center_mm']
            s_ = float(fo['size_mm'])
            ax.add_patch(Rectangle((fx - s_ / 2, fy - s_ / 2), s_, s_, 
                                    fill=False, 
                                    edgecolor='white', 
                                    lw=0.6, 
                                    ls=(0, (3, 1.5)),
                                    zorder=5))
            if gap_ > 0:
                ax.add_patch(Rectangle((fx - s_ / 2 - gap_, 
                                        fy - s_ / 2 - gap_), 
                                          s_ + 2 * gap_, s_ + 2 * gap_, 
                                        fill=False,
                                        edgecolor='white', 
                                        lw=0.5, 
                                        ls=':', 
                                        zorder=5))
        
        ax.set_xlim(-xr, xr)
        ax.set_ylim(-yr, yr)
        ax.set_xlabel('$x$ / mm')
        ax.set_ylabel('$y$ / mm')
        ax.set_title('(a) Top surface height')
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('$z$ above belt / mm')
        cb.outline.set_linewidth(0.6)

        # Define helper function for plotting cuts (side views)
        def cut(a, s, t_, b_, sel, label, xlabel, fo_rect):
            
            a.axhspan(z_lo, 0.0, color=c_belt, lw=0, zorder=1)
            a.fill_between(s, b_, t_, color=c_sut, step='mid', lw=0, zorder=2)
            
            for m, n_lo, n_hi in col['skin']:
                mm = sel(m)
                a.bar(s[mm], 
                      (sel(n_hi) - sel(n_lo))[mm] * dx * 1e3, 
                      bottom=sel(n_lo)[mm] * dx * 1e3, 
                      width=tile * 1e3,
                      color=c_skin, 
                      lw=0, 
                      zorder=2.5)
                
            for m, n_lo, n_hi in col['rim']:
                mm = sel(m)
                a.bar(s[mm], (sel(n_hi) - sel(n_lo))[mm] * dx * 1e3, 
                      bottom=sel(n_lo)[mm] * dx * 1e3, 
                      width=tile * 1e3,
                      color=c_rim, 
                      lw=0, 
                      zorder=2.55)
                
            for m, n_lo, n_hi in col['shell']:
                mm = sel(m)
                a.bar(s[mm], (sel(n_hi) - sel(n_lo))[mm] * dx * 1e3, 
                      bottom=sel(n_lo)[mm] * dx * 1e3, 
                      width=tile * 1e3,
                      color=c_shell, 
                      lw=0, 
                      zorder=2.65)
                
            for m, n_lo, n_hi in col['gaps']:
                mm = sel(m)
                a.bar(s[mm], (sel(n_hi) - sel(n_lo))[mm] * dx * 1e3, 
                      bottom=sel(n_lo)[mm] * dx * 1e3, 
                      width=tile * 1e3,
                      color='white', 
                      lw=0, 
                      zorder=2.6)
                
            a.step(s, t_, where='mid', color=e_sut, lw=0.5, zorder=3)
            
            for r, (m, n_lo, n_hi) in enumerate(col['regions']):
                mm = sel(m)
                a.bar(s[mm], (sel(n_hi) - sel(n_lo))[mm] * dx * 1e3, 
                      bottom=sel(n_lo)[mm] * dx * 1e3, 
                      width=tile * 1e3,
                      color=c_reg[r % len(c_reg)], 
                      lw=0, 
                      zorder=4)
                
            if fo_rect is not None:
                if gap_ > 0:
                    (u0, v0), su, sv = fo_rect
                    a.add_patch(Rectangle((u0 - gap_, 0.0), 
                                            su + 2 * gap_, v0 + sv, facecolor='white', 
                                            lw=0, 
                                            zorder=4.5))
                a.add_patch(Rectangle(fo_rect[0], 
                                        fo_rect[1], 
                                        fo_rect[2], 
                                        facecolor=c_fo, 
                                        edgecolor='white', 
                                        lw=0.5, 
                                        zorder=5))

            a.set_xlim(-xr, xr)
            a.set_ylim(z_lo, z_hi)
            a.set_ylabel('$z$ / mm')
            a.set_xlabel(xlabel)
            a.set_title(label)
            a.grid(True, axis='y', color=grid, lw=0.5, zorder=0)
            a.set_axisbelow(True)

        fo_xz = fo_yz = None
        if fo is not None:
            fx, fy, fz = fo['center_mm']
            s_ = float(fo['size_mm'])
            if abs(y[iy] - fy) <= s_ / 2:
                fo_xz = ((fx - s_ / 2, fz - s_ / 2), s_, s_)
            if abs(x[ix] - fx) <= s_ / 2:
                fo_yz = ((fy - s_ / 2, fz - s_ / 2), s_, s_)

        # Generate XZ and YZ cuts of the geometry
        cut(ax_x, x, 
            top[iy], bot[iy], 
            lambda A: A[iy], 
            '(b) XZ cut', 
            '$x$ / mm', 
            fo_xz)
        
        cut(ax_y, y, 
            top[:, ix], bot[:, ix], 
            lambda A: A[:, ix], 
            '(c) YZ cut', 
            '$y$ / mm', 
            fo_yz)

        # Hide the spines
        for a in (ax, ax_x, ax_y):
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)

        # Set handles for the legend
        handles = [Patch(facecolor=c_sut, 
                         edgecolor=e_sut, 
                         lw=0.5, 
                         label='Core')]
        if col['skin']:
            handles += [Patch(facecolor=c_skin, 
                              label='Skin')]
        if col['rim']:
            handles += [Patch(facecolor=c_rim, 
                              label='Rim')]
        if col['shell']:
            handles += [Patch(facecolor=c_shell, 
                              label='Moist shell')]
        
        handles += [Patch(facecolor=c_reg[r % len(c_reg)], 
                          label=str(reg.get('NAME', f'region {r}')).capitalize()) for r, reg in enumerate(regions)]
        
        if fo is not None:
            handles += [Patch(facecolor=c_fo, 
                              label=f"FO ({fo.get('material', '')} {float(fo['size_mm']):g} mm)".replace('( ', '('))]
            if gap_ > 0 and not col['gaps']:
                handles += [Patch(facecolor='white', edgecolor=ink, lw=0.4, label='Air')]
        
        if col['gaps']:
            handles += [Patch(facecolor='white', 
                              edgecolor=ink, 
                              lw=0.4, 
                              label='Air')]
        
        handles += [Patch(facecolor=c_belt, 
                          label='Belt')]
        fig.legend(handles=handles, 
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02 / H), 
                   ncol=ncol_leg, 
                   frameon=False,
                   columnspacing=1.0, 
                   handletextpad=0.4)

        # Save figure and plot
        written = None
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            written = _savefig(fig, out_path, dpi=300)
            print(f"plot_freeform_cookie_geometry(): wrote '{written}'.")
        fig_number = fig.number
        if show:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close(fig)
            fig_number = None

    reg_bottom = [float(np.min(n_lo[m]) * dx * 1e3) if m.any() else float('nan') for m, n_lo, _ in col['regions']]
    
    return {'out_path': written, 
            'fig_number': fig_number, 
            'n_columns': int(inside.sum()),
            'top_mm': (float(np.nanmin(top)), 
                       float(np.nanmax(top))), 
                       'region_bottom_mm': reg_bottom}

# endregion