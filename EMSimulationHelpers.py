"""Small helper functions shared by the EMSimulation modules (no imports from the other split modules)."""
from pathlib import Path


def _savefig(fig, path, **kwargs):
    """Save figure as PNG with 300 dpi and no metadata (to avoid embedding the user's username in the file).

    Args:
        fig (matplotlib.figure.Figure): 
            Figure to save.
        path (str or Path): 
            Destination file.
        **kwargs: 
            Passed to Figure.savefig(); dpi defaults to 300.

    Returns:
        Path: The written file.
    """
    path = Path(path).with_suffix('.png')
    kwargs.setdefault('dpi', 300)
    fig.savefig(path, format='png', metadata={'Software': None}, **kwargs)
    return path


def _get_sut_thickness_m(sut: dict) -> float:
    """Return the SUT's thickness in meters
    
    Args:
        sut (dict): The SUT dict from the manifest.

    Returns:
        float: The SUT's thickness in meters.
    """
    size = sut['SIZE']
    if isinstance(size, dict):
        return size['THICKNESS']
    
    return size[2]


def _rescale_fractions(fractions: dict, key: str, target_fraction: float) -> dict:
    """Return rescaled fractions dict (fractions[key] set to target_fraction and all other entries rescaled so they sum to 1).
    
    Args:
        fractions (dict): 
            Original fractions dict.
        key (str): 
            The key to set to target_fraction.
        target_fraction (float): The target fraction for the specified key.

    Returns:
        dict: New fractions dict with the specified key set to target_fraction and all other entries rescaled to sum to 1 - target_fraction.
    """
    others = {k: v for k, v in fractions.items() if k != key}
    others_sum = sum(others.values())
    remaining = max(1.0 - target_fraction, 0.0)
    if others_sum > 0:
        others = {k: v / others_sum * remaining for k, v in others.items()}
    out = dict(others)
    out[key] = target_fraction

    return out


def _cookie_metric_short_name(response_metric: str) -> str:
    """Retrieve the short name for a cookie response metric.
    
    Args:
        response_metric (str): The full response metric name.
    
    Returns:
        str: The short name for the response metric.
    """
    if 'dphase' in response_metric:
        return 'phase'
    if 'dmag' in response_metric:
        return 'amplitude'
    
    return response_metric


def _fmt_mds(mds: float, status: str) -> str:
    """Get cell text.
    
    Args:
        mds (float): 
            The minimum detectable size in mm.
        status (str): 
            The status of the detection ('not_detected', 'below_range', 'no_floor_for_metric', or other).

    Returns:
        str: The formatted cell text.
    """
    if status == 'not_detected':
        return 'not detected'
    if status == 'below_range':
        return f"<= {mds:g} mm"
    if status == 'no_floor_for_metric':
        return 'n/a'
    
    return f"{mds:.2f} mm"


def _fmt_offset(offset: float, status: str) -> str:
    """Get cell text for offset values.
    
    Args:
        offset (float): 
            The offset in mm.
        status (str): 
            The status of the detection ('not_detected', 'beyond_range', 'no_floor_for_metric', or other).

    Returns:
        str: The formatted cell text.
    """
    if status == 'not_detected':
        return 'not detected at center'
    if status == 'beyond_range':
        return f">= {offset:g} mm"
    if status == 'no_floor_for_metric':
        return 'n/a'
    
    return f"{offset:.2f} mm"


def _resolve_cookie_run_folder(study_dir, stored) -> 'Path':
    """Resolve a cookie-study run folder from a manifest's 'results_folder'.

    Args:
        study_dir (str or Path): 
            Study folder (the parent of results_analysis/ and input_output_files/).
        stored (str): 
            The manifest's results_folder entry.

    Returns:
        pathlib.Path: The resolved (or best-guess) run folder.
    """
    study_dir = Path(study_dir)
    p = Path(str(stored).replace('\\', '/'))
    candidates = [p] if p.is_absolute() else [study_dir / p, p]
    candidates.append(study_dir / 'input_output_files' / p.name)
    for c in candidates:
        if c.is_dir():
            return c
        
    return candidates[0]


def _cookie_short_name(cookie: str) -> str:
    """Get cookie's short name.
    
    Args:
        cookie (str): The cookie name from the manifest.

    Returns:
        str: The cookie's short name (label used in the paper's tables and figures).
    """
    return cookie.split(' (')[0].replace('-', ' ')


def _format_duration(seconds: float) -> str:
    """Get duration that is easy to read.
    
    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration string.
    """
    total_minutes = int(round(seconds / 60.0))
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    
    return " ".join(parts)


def _freeform_pinch(ff: dict, a: float, b: float, t):
    """Relative outline radius from SIDE_PINCH [m]: the four sides (+-x, +-y) pushed inwards, localized by SIDE_PINCH_SHARPNESS."""
    import numpy as np
    t = np.asarray(t, float)
    pinch = float(ff.get('SIDE_PINCH', 0.0))
    if pinch <= 0:
        return np.ones_like(t)
    return 1.0 - pinch / np.hypot(a * np.cos(t), b * np.sin(t)) * np.abs(np.cos(2 * t)) ** float(ff.get('SIDE_PINCH_SHARPNESS', 4))


def _freeform_rho(ff: dict, a: float, b: float, t):
    """Relative outline radius of a 'FREEFORM' SUT at ellipse parameter t: side pinch (_freeform_pinch()) times
    FREEFORM['SCALLOPS'] scallops of depth SCALLOP_DEPTH [m]."""
    import numpy as np
    n_s, depth = int(ff.get('SCALLOPS', 0)), float(ff.get('SCALLOP_DEPTH', 0.0))
    scallop = (1.0 - depth / ((a + b) / 2) * (1.0 - np.cos(n_s * np.asarray(t, float))) / 2
               if n_s > 0 and depth > 0 else 1.0)
    return _freeform_pinch(ff, a, b, t) * scallop


def _freeform_region_rho(region: dict, t):
    """Relative radius of a REGIONS ellipse at angle t: 1 + IRREGULARITY x a smooth random wobble (harmonics 2-5, seed SEED)."""
    import numpy as np
    t = np.asarray(t, float)
    irr = float(region.get('IRREGULARITY', 0.0))
    if irr <= 0:
        return np.ones_like(t)
    rng = np.random.default_rng(int(region.get('SEED', 1)))
    coef = [(k, rng.standard_normal(), rng.uniform(0, 2 * np.pi)) for k in range(2, 6)]
    tt = np.linspace(0.0, 2 * np.pi, 721)
    norm = max(np.max(np.abs(sum(c / k * np.cos(k * tt + ph) for k, c, ph in coef))), 1e-12)
    return 1.0 + irr * sum(c / k * np.cos(k * t + ph) for k, c, ph in coef) / norm


def _freeform_outline(sut: dict, n: int = 720):
    """Closed outline of a 'FREEFORM' SUT in the cookie frame: ellipse SIZE[0] x SIZE[1] with the scallops of _freeform_rho().

    Args:
        sut (dict): SUT dict with SHAPE 'FREEFORM'.
        n (int): Number of outline points.

    Returns:
        tuple: (x, y) arrays [m] (first point repeated at the end).
    """
    import numpy as np
    a, b = sut['SIZE'][0] / 2, sut['SIZE'][1] / 2
    t = np.linspace(0.0, 2 * np.pi, n + 1)
    rho = _freeform_rho(sut.get('FREEFORM', {}), a, b, t)
    return a * rho * np.cos(t), b * rho * np.sin(t)


def _freeform_region_outline(region: dict, n: int = 360):
    """Closed outline (x, y) [m] of one FREEFORM['REGIONS'] ellipse (with its IRREGULARITY) in the cookie frame."""
    import numpy as np
    t = np.linspace(0.0, 2 * np.pi, n + 1)
    cx, cy = region.get('CENTER', (0.0, 0.0))
    r = _freeform_region_rho(region, t)
    return cx + region['SIZE'][0] / 2 * r * np.cos(t), cy + region['SIZE'][1] / 2 * r * np.sin(t)


def _freeform_columns(sut: dict, tile: float, dz: float) -> dict:
    """Column model of a 'FREEFORM' SUT on a tile grid in the cookie frame (heights above the belt in whole dz cells).

    Outline from _freeform_outline(); top/bottom edges rounded with radius EDGE_RADIUS_TOP / EDGE_RADIUS_BOTTOM [m] (over the inward distance
    from the outline); top surface plus a smooth random field of peak ROUGHNESS_AMPLITUDE [m], correlation length ROUGHNESS_CORRELATION [m]
    and seed ROUGHNESS_SEED, tapered to zero at the rim; ROUGHNESS_MODE 'LOWER' = only below the nominal top (0..-amplitude), 'RAISE' = only above. REGIONS (ellipses CENTER [dx, dy], SIZE [Lx, Ly] [m]) sit in a well (outline wobbled by IRREGULARITY): top
    TOP_OFFSET [m] below the unroughened dough top at the region edge, sagging parabolically by TOP_SAG [m] towards the center, where
    the region is DEPTH [m] thick (TO_BELT = True: the region reaches down to the belt, DEPTH ignored); the well rim and the region's bottom edge are rounded with EDGE_RADIUS [m]. GROOVES: ring grooves
    concentric to the (pinched) ellipse at SCALE x its semi-axes (cosine profile); width and depth vary smoothly around the ring between
    WIDTH_MIN..WIDTH_MAX and DEPTH_MIN..DEPTH_MAX [m] (seed SEED; WIDTH alone = constant width). SKIN: moister outer layer THICKNESS [m] on
    FACES ('TOP' = below the dough top outside the regions, 'SIDES' = columns within THICKNESS of the outline, 'BOTTOM' = above the belt).
    RIM: band of THICKNESS [m], full height (punched edge: compressed, bent-down layers), its own material, written after the skin; FACES
    'OUTER' = inside the outline (default), 'INNER' = around the regions (well walls). FO_SHELL: moist layer of THICKNESS [m] around the
    FO's drilled hole, BOX = [x0, x1, y0, y1, z_top] [m] of the hole (set by MODE 12), from the belt up to z_top + THICKNESS. LAMINATION: the core between the skins is N_SHEETS sheets of SHEET_THICKNESS [m] (the SUT material) with air gaps between and around
    them, spread evenly over each column's core height (so the sheets follow the surfaces); thin columns get fewer gaps, none if g < 1 cell.

    Args:
        sut (dict): SUT dict with SHAPE 'FREEFORM', SIZE [Lx, Ly, thickness] [m] and an optional 'FREEFORM' block.
        tile (float): Lateral tile size [m].
        dz (float): Height quantum [m] (grid cell).

    Returns:
        dict: x (Nx,), y (Ny,) tile centers [m]; inside (Ny, Nx) bool; n_lo, n_hi (Ny, Nx) column bottom/top in cells above the belt;
            regions: list of (mask, n_lo, n_hi); skin: list of (mask, n_lo, n_hi) (empty without SKIN); rim: same (empty without RIM);
            gaps: air gaps of LAMINATION, same form; shell: FO_SHELL, same form.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    ff = sut.get('FREEFORM', {})
    lx, ly, thick = sut['SIZE']
    a, b = lx / 2, ly / 2
    nx, ny = int(np.ceil(lx / tile)), int(np.ceil(ly / tile))
    x = (np.arange(nx) - (nx - 1) / 2) * tile
    y = (np.arange(ny) - (ny - 1) / 2) * tile
    X, Y = np.meshgrid(x, y)
    U, V = X / a, Y / b
    inside = np.hypot(U, V) <= _freeform_rho(ff, a, b, np.arctan2(V, U))
    d_in = np.clip(distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1] * tile - tile / 2, 0.0, None)

    def drop(r):
        s = np.clip(r - d_in, 0.0, None)
        return r - np.sqrt(np.clip(r ** 2 - s ** 2, 0.0, None)) if r > 0 else np.zeros_like(d_in)
    r_top, r_bot = float(ff.get('EDGE_RADIUS_TOP', 0.0)), float(ff.get('EDGE_RADIUS_BOTTOM', 0.0))
    z_lo = drop(r_bot)
    z_top = thick - drop(r_top)
    z_hi = z_top.copy()
    amp = float(ff.get('ROUGHNESS_AMPLITUDE', 0.0))
    if amp > 0 and inside.any():
        rng = np.random.default_rng(int(ff.get('ROUGHNESS_SEED', 1)))
        field = gaussian_filter(rng.standard_normal(inside.shape),
                                sigma=max(float(ff.get('ROUGHNESS_CORRELATION', 0.003)) / tile, 0.5), mode='wrap')
        field /= np.max(np.abs(field[inside]))
        mode = str(ff.get('ROUGHNESS_MODE', 'SYMMETRIC')).upper()
        if mode in ('LOWER', 'RAISE'):
            f_in = field[inside]
            field = ((field - f_in.max()) if mode == 'LOWER' else (field - f_in.min())) / max(float(np.ptp(f_in)), 1e-12)
        taper = np.clip(d_in / r_top, 0.0, 1.0) if r_top > 0 else 1.0
        z_hi = z_hi + amp * field * taper
    for grv in ff.get('GROOVES', []):
        t = np.arctan2(V, U)
        dist = (np.hypot(U, V) / _freeform_pinch(ff, a, b, t) - float(grv['SCALE'])) * np.hypot(a * np.cos(t), b * np.sin(t))
        rng = np.random.default_rng(int(grv.get('SEED', 1)))
        def wave():
            v = sum(rng.standard_normal() * np.cos(k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 5))
            return (v - v.min()) / max(np.ptp(v), 1e-12)
        depth = float(grv['DEPTH_MIN']) + (float(grv['DEPTH_MAX']) - float(grv['DEPTH_MIN'])) * wave()
        w_lo, w_hi = float(grv.get('WIDTH_MIN', grv.get('WIDTH'))), float(grv.get('WIDTH_MAX', grv.get('WIDTH')))
        w = w_lo + (w_hi - w_lo) * wave()
        z_hi = z_hi - np.where(np.abs(dist) < w / 2, depth * np.cos(np.pi * dist / w) ** 2, 0.0)
    masks = []
    for reg in ff.get('REGIONS', []):
        cx, cy = reg.get('CENTER', (0.0, 0.0))
        Ur, Vr = (X - cx) / (reg['SIZE'][0] / 2), (Y - cy) / (reg['SIZE'][1] / 2)
        rho2 = (np.hypot(Ur, Vr) / _freeform_region_rho(reg, np.arctan2(Vr, Ur))) ** 2       # 1 on the region outline
        m = inside & (rho2 <= 1.0)
        r_e = float(reg.get('EDGE_RADIUS', 0.0))
        rho2 = np.clip(rho2, 0.0, 1.0)
        level = z_top - float(reg.get('TOP_OFFSET', 0.0))                       # region top at its edge
        sag = float(reg.get('TOP_SAG', 0.0))
        top = (level if reg.get('SMOOTH_TOP', True) else z_hi - float(reg.get('TOP_OFFSET', 0.0))) - sag * (1.0 - rho2)
        floor_ = level - sag - float(reg['DEPTH'])                              # region bottom (center thickness = DEPTH)
        bottom = np.minimum(floor_, top)
        if r_e > 0:
            # Well rim: dough top rounds down to the region top within r_e outside; bottom edge of the region rounds up within r_e inside.
            d_out = np.clip(distance_transform_edt(~m) * tile - tile / 2, 0.0, None)
            d_reg = np.clip(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1] * tile - tile / 2, 0.0, None)
            q_out = np.sqrt(np.clip(1.0 - (np.clip(r_e - d_out, 0.0, None) / r_e) ** 2, 0.0, None))
            q_in = np.sqrt(np.clip(1.0 - (np.clip(r_e - d_reg, 0.0, None) / r_e) ** 2, 0.0, None))
            rim = inside & ~m & (d_out < r_e)
            z_hi = np.where(rim, np.maximum(level, level + (z_hi - level) * q_out), z_hi)
            bottom = top - (top - floor_) * q_in
        z_hi = np.where(m, top, z_hi)
        masks.append((m, bottom))
    n_lo = np.rint(z_lo / dz).astype(int)
    n_hi = np.rint(z_hi / dz).astype(int)
    inside &= n_hi > n_lo
    regions = []
    for (m, bottom), reg in zip(masks, ff.get('REGIONS', [])):
        n_b = n_lo.copy() if reg.get('TO_BELT', False) else np.maximum(np.rint(bottom / dz).astype(int), n_lo)
        m = m & inside & (n_hi > n_b)
        regions.append((m, n_b, n_hi.copy()))
    skin = []
    sk = ff.get('SKIN')
    if sk:
        t_sk = float(sk['THICKNESS'])
        k = max(1, int(np.rint(t_sk / dz)))
        faces = {f.upper() for f in sk.get('FACES', ('TOP', 'SIDES'))}
        no_reg = ~np.any([m for m, _, _ in regions], axis=0) if regions else np.ones_like(inside)
        if 'TOP' in faces:
            skin.append((inside & no_reg, np.maximum(n_hi - k, n_lo), n_hi.copy()))
        if 'SIDES' in faces:
            skin.append((inside & (d_in < t_sk), n_lo.copy(), n_hi.copy()))
        if 'BOTTOM' in faces:
            skin.append((inside, n_lo.copy(), np.minimum(n_lo + k, n_hi)))
    rim = []
    rb = ff.get('RIM')
    if rb:
        t_rim = float(rb['THICKNESS'])
        r_faces = {f.upper() for f in rb.get('FACES', ('OUTER',))}
        m_rim = (inside & (d_in < t_rim)) if 'OUTER' in r_faces else np.zeros_like(inside)
        if 'INNER' in r_faces and regions:
            reg_any = np.any([m for m, _, _ in regions], axis=0)
            d_reg = np.clip(distance_transform_edt(~reg_any) * tile - tile / 2, 0.0, None)
            m_rim = m_rim | (inside & ~reg_any & (d_reg < t_rim))
        rim.append((m_rim, n_lo.copy(), n_hi.copy()))
    gaps = []
    lam = ff.get('LAMINATION')
    if lam:
        n_sh = int(lam['N_SHEETS'])
        t_sh = max(1, int(np.rint(float(lam['SHEET_THICKNESS']) / dz)))
        no_reg = ~np.any([m for m, _, _ in regions], axis=0) if regions else np.ones_like(inside)
        k_sk = max(1, int(np.rint(float(sk['THICKNESS']) / dz))) if sk else 0
        faces = {f.upper() for f in sk.get('FACES', ('TOP', 'SIDES'))} if sk else set()
        c_lo = n_lo + (k_sk if 'BOTTOM' in faces else 0)
        c_hi = n_hi - (k_sk if 'TOP' in faces else 0)
        side = (d_in < float(sk['THICKNESS'])) if 'SIDES' in faces else np.zeros_like(inside)
        if rb:
            side = side | rim[0][0]
        g_tot = c_hi - c_lo - n_sh * t_sh                                   # air cells per column
        core = inside & no_reg & ~side & (g_tot >= 1)
        for j in range(n_sh + 1):                                            # gap j below sheet j (j = n_sh: above the last sheet)
            g0 = c_lo + (j * g_tot) // (n_sh + 1) + j * t_sh
            g1 = c_lo + ((j + 1) * g_tot) // (n_sh + 1) + j * t_sh
            gaps.append((core & (g1 > g0), g0, g1))
    shell = []
    fs = ff.get('FO_SHELL')
    if fs and fs.get('BOX'):
        x0, x1, y0, y1, z_t = (float(v) for v in fs['BOX'])
        t_fs = float(fs['THICKNESS'])
        m_fs = inside & (X >= x0 - t_fs) & (X <= x1 + t_fs) & (Y >= y0 - t_fs) & (Y <= y1 + t_fs)
        hi_fs = np.minimum(n_hi, int(np.rint((z_t + t_fs) / dz)))
        shell.append((m_fs & (hi_fs > n_lo), n_lo.copy(), hi_fs))
    return dict(x=x, y=y, inside=inside, n_lo=n_lo, n_hi=n_hi, regions=regions, skin=skin, rim=rim, gaps=gaps, shell=shell)


def _freeform_fo_position_fraction(sut: dict, fo_size_mm: float) -> list:
    """FO POSITION_RELATIVE [x, y, z] for 'RING_LEFT' in a 'FREEFORM' SUT: on the x axis (y = 0) left of the center, halfway between the
    outer edge and the first region's edge (the center without regions), resting on the belt (bottom at z = 0).

    Args:
        sut (dict): SUT dict with SHAPE 'FREEFORM'.
        fo_size_mm (float): FO cube edge [mm].

    Returns:
        list: [x, y, z] fractions of the SUT bounding box SIZE.
    """
    import numpy as np
    ff = sut.get('FREEFORM', {})
    lx, ly, thick = sut['SIZE']
    a, b = lx / 2, ly / 2
    x_out = -a * float(_freeform_rho(ff, a, b, np.pi))
    regions = ff.get('REGIONS', [])
    if regions:
        reg = regions[0]
        x_in = reg.get('CENTER', (0.0, 0.0))[0] - reg['SIZE'][0] / 2 * float(_freeform_region_rho(reg, np.pi))
    else:
        x_in = 0.0
    x_c = (x_out + x_in) / 2
    return [(x_c + a) / lx, 0.5, fo_size_mm / 2000.0 / thick]
