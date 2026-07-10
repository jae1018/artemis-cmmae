"""
Classification-showcase plotting for the ARTEMIS plasma classifier.

This module renders a multi-panel figure that summarises a classified pass:
an ion energy spectrogram, the GSE magnetic field, ion density + temperature,
and two region strips -- "raw" (per-sample) and a persistence strip (regions
held continuously for at least ``min_duration``, default '1h') -- optionally
beside a GSE X-Y trajectory panel with ported Shue magnetopause and Chao
bow-shock reference curves. The layout/colours mirror the reference
ARTEMIS-PS three-row time-series figure.

``matplotlib`` is a required dependency of ``artemis-cmmae`` but is imported
LAZILY inside :func:`plot_classification_timeseries`, so ``import artemis_cmmae``
(and even ``import artemis_cmmae.plotting``) does not pull it in until you
actually render a figure.

The Shue and Chao boundary formulas are PORTED (copied, self-contained) from the
private ``geospacefronts`` package (``geospacefronts/shue.py`` and
``geospacefronts/chao.py``); ``geospacefronts`` is NOT a runtime dependency and
is never imported here.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .features import N_MODEL_CHANNELS, REF_ENERGY_GRID_ASC
from .utils import region_timeline

# ---------------------------------------------------------------------------
# Region colours / names (match the reference three-row figure COLOR_MAP)
# ---------------------------------------------------------------------------
#: region_id -> hex colour. SW=green, MSH=orange, Lobe=blue, PS=red, Unknown=grey.
_REGION_COLORS = {
    0: "#2ca02c",   # Solar Wind   - green
    1: "#ff7f0e",   # Magnetosheath - orange
    2: "#1f77b4",   # Lobe         - blue
    3: "#d62728",   # Plasma Sheet - red
    -1: "#7f7f7f",  # Unknown      - grey
}
#: region_id -> short label used in the strip legend.
_REGION_ABBR = {0: "SW", 1: "MSH", 2: "Lobe", 3: "PS", -1: "Unknown"}
#: legend/colour ordering matching the reference figure (SW, MSH, PS, Lobe, Unknown).
_LEGEND_ORDER = [0, 1, 3, 2, -1]
#: id order used to build the raw-strip ListedColormap (any stable order works).
_STRIP_IDS = [0, 1, 2, 3, -1]

# B-component line colours (Bx/By/Bz), independent of the region colours.
_BX_COLOR, _BY_COLOR, _BZ_COLOR = "#1f77b4", "#2ca02c", "#d62728"
_N_COLOR = "black"
_T_COLOR = "#d62728"

# Gap thresholds (seconds): break pcolormesh cells / plot lines across real gaps.
_PCOLOR_MAX_GAP_SEC = 1200.0
_LINE_MAX_GAP_SEC = 300.0

_KM_PER_RE = 6371.2

# ---------------------------------------------------------------------------
# Spectrogram energy-bin edges from the 31 model channel centres (ascending eV)
# ---------------------------------------------------------------------------
_E_CENTERS = np.asarray(REF_ENERGY_GRID_ASC[:N_MODEL_CHANNELS], dtype=float)
_log_c = np.log10(_E_CENTERS)
_mid = (_log_c[:-1] + _log_c[1:]) / 2.0
_left = _log_c[0] - (_mid[0] - _log_c[0])
_right = _log_c[-1] + (_log_c[-1] - _mid[-1])
#: 32 log-spaced energy-bin edges (eV) bracketing the 31 channel centres.
_E_BIN_EDGES = 10.0 ** np.concatenate([[_left], _mid, [_right]])


# ===========================================================================
# Ported Shue magnetopause + Chao bow shock (self-contained; see module docs)
# ===========================================================================
def _shue_r0_alpha(bz: float, dp: float):
    """Shue (1997/1998) sub-solar standoff r0 [R_E] and flaring alpha.

    Ported from ``geospacefronts/shue.py::shue_r0_alpha`` (DEFAULT_COEFFS):
        r0 = (10.22 + 1.29 * tanh(0.184 * (Bz + 8.14))) * Dp**(-1/6.6)
        alpha = (0.58 - 0.007 * Bz) * (1 + 0.024 * ln Dp)
    Bz in nT, Dp in nPa.
    """
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * dp ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * bz) * (1.0 + 0.024 * np.log(dp))
    return r0, alpha


def _shue_magnetopause_xy(bz: float = 0.0, dp: float = 2.0, *,
                          n_theta: int = 361, x_min: Optional[float] = None):
    """Ported Shue magnetopause as a closed GSE X-Y curve (R_E).

    r(theta) = r0 * (2 / (1 + cos theta))**alpha, evaluated on theta in
    [0, 179.5] deg then mirrored about y=0 (matching
    ``geospacefronts/shue.py::shue_xy`` with ``mirror_y=True``). If ``x_min`` is
    given, points with x < x_min are dropped before mirroring.

    Returns:
        ``(x, y)`` 1-D numpy arrays tracing the boundary from the tail flank
        through the sub-solar nose and back down the opposite flank.
    """
    theta = np.deg2rad(np.linspace(0.0, 179.5, int(n_theta)))
    r0, alpha = _shue_r0_alpha(bz, dp)
    denom = 1.0 + np.cos(theta)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    r = r0 * (2.0 / denom) ** alpha
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if x_min is not None:
        keep = x >= x_min
        x, y = x[keep], y[keep]
    x_full = np.concatenate([x[::-1], x])
    y_full = np.concatenate([y[::-1], -y])
    return x_full, y_full


# Chao (2002) coefficients, verbatim from geospacefronts/chao.py::ChaoCoeffs.
_CHAO_COEFFS = (
    11.1266, 0.001, -0.0005, 2.5966, 0.8182, -0.017, -0.0122, 1.3007,
    -0.0049, -0.0328, 6.047, 1.029, 0.0231, -0.002,
)


def _chao_r0_alpha(bz: float, dp: float, mgs: float, beta: float):
    """Chao (2002) bow-shock r0 [R_E] and alpha (ported from geospacefronts).

    Ported from ``geospacefronts/chao.py::chao_r0_alpha`` (scalar-parameter
    form; the Bz>=0 vs Bz<0 branch is selected by the sign of ``bz``).
    """
    (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) = _CHAO_COEFFS
    r0_mhd = ((a8 - 1.0) * mgs ** 2 + 2.0) / ((a8 + 1.0) * mgs ** 2)
    base_r0 = a1 * (1.0 + a9 * beta) * (1.0 + a4 * r0_mhd) * dp ** (-1.0 / a11)
    base_a = a5 * (1.0 + a7 * dp) * (1.0 + a10 * np.log1p(beta)) * (1.0 + a14 * mgs)
    if bz >= 0.0:
        r0 = base_r0 * (1.0 + a2 * bz)
        alpha = base_a * (1.0 + a13 * bz)
    else:
        r0 = base_r0 * (1.0 + a3 * bz)
        alpha = base_a * (1.0 + a6 * bz)
    return r0, alpha


def _chao_bowshock_xy(bz: float = 0.2, dp: float = 2.0, mgs: float = 6.0,
                      beta: float = 1.0, *, n_theta: int = 321,
                      x_min: Optional[float] = -90.0):
    """Ported Chao bow shock as a closed GSE X-Y curve (R_E).

    r(theta) = r0 * ((1 + eps) / (1 + eps * cos theta))**alpha with
    eps = a12 = 1.029, evaluated on theta in [0, 160] deg (kept below the
    eps>1 asymptote near ~166 deg) then mirrored about y=0
    (matching ``geospacefronts/chao.py::chao_xy``). Points with x < x_min are
    dropped before mirroring to clip the far-tail flare.

    Returns:
        ``(x, y)`` 1-D numpy arrays tracing the bow shock.
    """
    eps = _CHAO_COEFFS[11]  # a12
    theta = np.deg2rad(np.linspace(0.0, 160.0, int(n_theta)))
    r0, alpha = _chao_r0_alpha(bz, dp, mgs, beta)
    denom = 1.0 + eps * np.cos(theta)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    r = r0 * ((1.0 + eps) / denom) ** alpha
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if x_min is not None:
        keep = x >= x_min
        x, y = x[keep], y[keep]
    x_full = np.concatenate([x[::-1], x])
    y_full = np.concatenate([y[::-1], -y])
    return x_full, y_full


def _aberration_rotate(x, y, vx: float = 400.0, vy: float = 30.0):
    """Rotate boundary curves into the aberrated solar-wind frame.

    Matches the reference figure: nominal Vx_sw = -400 km/s, Vy_sw = +30 km/s
    gives an aberration angle atan(Vy/|Vx|) ~ 4.3 deg; the nose tilts toward
    -Y_GSE (clockwise about +Z).
    """
    ang = np.arctan(vy / vx)
    ca, sa = np.cos(ang), np.sin(ang)
    return ca * x + sa * y, -sa * x + ca * y


# ===========================================================================
# Small plotting helpers (gap-aware chunking; pure numpy)
# ===========================================================================
def _make_chunks(tnum, max_gap_days):
    """Split into contiguous index chunks wherever the time gap exceeds a limit."""
    if len(tnum) > 1:
        gap_idx = np.where(np.diff(tnum) > max_gap_days)[0]
        starts = np.concatenate([[0], gap_idx + 1])
        ends = np.concatenate([gap_idx + 1, [len(tnum)]])
        return starts, ends
    return np.array([0]), np.array([len(tnum)])


def _chunk_edges(t_chunk, fallback_half_width):
    """Cell edges (midpoints) for a pcolormesh chunk of sample-centre times."""
    if len(t_chunk) < 2:
        return np.array([t_chunk[0] - fallback_half_width,
                         t_chunk[0] + fallback_half_width])
    mid = (t_chunk[:-1] + t_chunk[1:]) / 2.0
    first = t_chunk[0] - (mid[0] - t_chunk[0])
    last = t_chunk[-1] + (t_chunk[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _break_at_gaps(t, vals, max_gap_days):
    """Insert NaNs so line plots break across real data gaps."""
    if len(t) < 2:
        return t, vals
    gap_idx = np.where(np.diff(t) > max_gap_days)[0]
    if len(gap_idx) == 0:
        return t, vals
    insert_at = gap_idx + 1
    new_t = np.insert(t, insert_at, t[gap_idx] + max_gap_days * 0.5)
    new_v = np.insert(np.asarray(vals, dtype=float), insert_at, np.nan)
    return new_t, new_v


def _require_matplotlib():
    """Lazy-import matplotlib, raising a clear missing-dependency error otherwise."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, ListedColormap
        from matplotlib.dates import date2num, DateFormatter, AutoDateLocator
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "plotting requires matplotlib, which is a required dependency of "
            "artemis-cmmae; reinstall it with: pip install artemis-cmmae"
        ) from exc
    return {
        "plt": plt, "LogNorm": LogNorm, "ListedColormap": ListedColormap,
        "date2num": date2num, "DateFormatter": DateFormatter,
        "AutoDateLocator": AutoDateLocator, "Patch": Patch, "Rectangle": Rectangle,
    }


# ===========================================================================
# Public figure
# ===========================================================================
def plot_classification_timeseries(
    result,
    *,
    min_duration: str = "1h",
    position=None,
    title: Optional[str] = None,
    figsize=None,
    save_path: Optional[str] = None,
    temperature=None,
):
    """Render the multi-panel classification showcase figure.

    Panels (top-to-bottom, on the right when a trajectory is shown):

    * Ion energy spectrogram: ``pcolormesh`` of the C0..C30 linear energy flux
      (log colour), energy on a log y-axis using the 31 model channel centres.
    * B (nT): Bx/By/Bz vs time.
    * n (cm^-3): density on a log y-axis (with an optional T_i overlay).
    * "raw" strip: per-sample ``region_id`` as a colour-coded row.
    * persistence strip: the :func:`~artemis_cmmae.utils.region_timeline`
      subset -- regions held continuously for at least ``min_duration`` -- as a
      colour-coded row (the y-axis label is the ``min_duration`` string).

    Colours: Solar Wind green, Magnetosheath orange, Plasma Sheet red, Lobe
    blue, Unknown grey (shared legend under the strips).

    If ``position`` is given -- or ``result`` itself carries the GSE position
    columns ``X_GSE``/``Y_GSE``/``Z_GSE`` -- a GSE X-Y trajectory panel is added
    on the LEFT: the orbit path (coloured by region) with midnight date markers,
    plus the ported Shue magnetopause and Chao bow-shock reference curves. If no
    position is available the trajectory panel is omitted (time-series-only
    figure).

    Args:
        result: A prediction-output ``DataFrame`` from
            ``predict_from_pyspedas(..., return_features=True)`` (or
            ``predict_from_ds``): a ``DatetimeIndex`` with ``region_id`` plus the
            feature columns ``C0..C30``, ``n``, ``BX_GSE``/``BY_GSE``/``BZ_GSE``
            (a panel is skipped gracefully if its columns are absent). If it also
            carries the GSE position ``X_GSE``/``Y_GSE``/``Z_GSE`` those are used
            for the trajectory panel automatically (no separate ``position`` arg).
        min_duration: Minimum continuous dwell time for the persistence strip
            (passed to :func:`~artemis_cmmae.utils.region_timeline`); also used
            as that strip's y-axis label. Default ``'1h'``.
        position: Optional override ``DataFrame`` with GSE ``X_GSE``/``Y_GSE``/
            ``Z_GSE`` in R_E and a ``DatetimeIndex`` (e.g. from
            :func:`artemis_cmmae.loaders.load_position_frame`). When omitted, the
            position columns in ``result`` are used if present.
        title: Figure suptitle; a default derived from the time range is used if
            None.
        figsize: ``(w, h)`` inches; a sensible default is chosen per layout.
        save_path: If given, the figure is also saved here (``dpi=150``,
            ``bbox_inches='tight'``).
        temperature: Optional array-like (aligned to ``result`` rows) of ion
            temperature (eV) to overlay on the density panel.

    Returns:
        The ``matplotlib.figure.Figure``.

    Raises:
        ImportError: If matplotlib is not importable (it is a required
            dependency of ``artemis-cmmae``).
        ValueError: If ``result`` lacks ``region_id`` or the spectrum columns.
    """
    mpl = _require_matplotlib()
    plt = mpl["plt"]
    LogNorm = mpl["LogNorm"]
    ListedColormap = mpl["ListedColormap"]
    date2num = mpl["date2num"]
    DateFormatter = mpl["DateFormatter"]
    AutoDateLocator = mpl["AutoDateLocator"]
    Patch = mpl["Patch"]
    Rectangle = mpl["Rectangle"]

    # --- Validate + sort ----------------------------------------------------
    if not isinstance(result, pd.DataFrame):
        raise TypeError("result must be a pandas DataFrame (prediction output).")
    if "region_id" not in result.columns:
        raise ValueError("result must contain a 'region_id' column (predict output).")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("result must be indexed by a pandas DatetimeIndex.")

    ch_cols = [f"C{i}" for i in range(N_MODEL_CHANNELS)]
    missing_spec = [c for c in ch_cols if c not in result.columns]
    if missing_spec:
        raise ValueError(
            "result is missing spectrum columns "
            f"{missing_spec[:3]}...; call the predictor with return_features=True."
        )

    order = np.argsort(result.index.values, kind="stable")
    df = result.iloc[order]
    times = df.index
    tnum = date2num(np.asarray(times.to_pydatetime()))
    region_id = df["region_id"].to_numpy(dtype=int)

    temp_arr = None
    if temperature is not None:
        if isinstance(temperature, pd.Series):
            temp_arr = temperature.reindex(times).to_numpy(dtype=float)
        else:
            t_in = np.asarray(temperature, dtype=float)
            if t_in.shape[0] == len(result):
                temp_arr = t_in[order]
    elif "T" in df.columns:
        # Auto-use the ion temperature carried by the predictor output (present
        # when return_features=True), so the density panel gets its T_i twin
        # axis with no extra arguments.
        temp_arr = df["T"].to_numpy(dtype=float)

    b_present = all(c in df.columns for c in ("BX_GSE", "BY_GSE", "BZ_GSE"))
    n_present = "n" in df.columns

    pc_gap = _PCOLOR_MAX_GAP_SEC / 86400.0
    line_gap = _LINE_MAX_GAP_SEC / 86400.0
    pc_starts, pc_ends = _make_chunks(tnum, pc_gap)

    # --- Layout -------------------------------------------------------------
    # Auto-use the GSE position columns from `result` (X_GSE/Y_GSE/Z_GSE) for the
    # trajectory panel when no explicit `position` override was supplied.
    if position is None and all(c in df.columns for c in ("X_GSE", "Y_GSE", "Z_GSE")):
        position = df
    have_traj = position is not None
    if figsize is None:
        figsize = (15, 9) if have_traj else (12, 9)
    fig = plt.figure(figsize=figsize)
    h_ratios = [3.0, 2.0, 2.2, 1.4]
    if have_traj:
        gs = fig.add_gridspec(
            4, 2, width_ratios=[0.30, 0.70], height_ratios=h_ratios,
            hspace=0.10, wspace=0.16,
            left=0.055, right=0.915, top=0.93, bottom=0.10,
        )
        ax_traj = fig.add_subplot(gs[0:4, 0])
        col = 1
    else:
        gs = fig.add_gridspec(
            4, 1, height_ratios=h_ratios, hspace=0.10,
            left=0.09, right=0.90, top=0.93, bottom=0.10,
        )
        ax_traj = None
        col = 0
    ax_spec = fig.add_subplot(gs[0, col])
    ax_B = fig.add_subplot(gs[1, col], sharex=ax_spec)
    ax_n = fig.add_subplot(gs[2, col], sharex=ax_spec)
    ax_cls = fig.add_subplot(gs[3, col], sharex=ax_spec)

    # Manual colorbar axis next to the spectrogram (keeps panel widths aligned).
    fig.canvas.draw()
    spec_pos = ax_spec.get_position()
    cax = fig.add_axes([spec_pos.x1 + 0.006, spec_pos.y0, 0.010, spec_pos.height])

    # --- Spectrogram --------------------------------------------------------
    C = df[ch_cols].to_numpy(dtype=float)
    eflux = np.where(np.isfinite(C) & (C > 0), C, np.nan)
    pm = None
    for s, e in zip(pc_starts, pc_ends):
        if e - s < 1:
            continue
        edges = _chunk_edges(tnum[s:e], pc_gap / 4.0)
        m = ax_spec.pcolormesh(
            edges, _E_BIN_EDGES, eflux[s:e].T,
            norm=LogNorm(vmin=1e3, vmax=1e8), cmap="jet", shading="flat",
        )
        if pm is None:
            pm = m
    ax_spec.set_yscale("log")
    ax_spec.set_ylabel("Ion energy (eV)")
    if pm is not None:
        cb = fig.colorbar(pm, cax=cax)
        cb.set_label("eflux (eV/cm^2/s/sr/eV)", fontsize=9)
    else:
        cax.set_visible(False)

    # --- Line helper --------------------------------------------------------
    def _lineplot(ax, vals, color, lw=0.7, label=None, ms=1.6):
        v = np.asarray(vals, dtype=float)
        t2, v2 = _break_at_gaps(tnum, v, line_gap)
        ax.plot(t2, v2, color=color, lw=lw, label=label, zorder=2)
        finite = np.isfinite(v)
        if finite.any():
            ax.plot(tnum[finite], v[finite], linestyle="None", marker="o",
                    markersize=ms, markerfacecolor=color, markeredgecolor="none",
                    zorder=3, alpha=0.85)

    # --- B panel ------------------------------------------------------------
    if b_present:
        _lineplot(ax_B, df["BX_GSE"].to_numpy(float), _BX_COLOR, label="Bx")
        _lineplot(ax_B, df["BY_GSE"].to_numpy(float), _BY_COLOR, label="By")
        _lineplot(ax_B, df["BZ_GSE"].to_numpy(float), _BZ_COLOR, label="Bz")
        ax_B.axhline(0, color="k", ls=":", lw=0.5)
        ax_B.legend(loc="upper right", ncol=3, fontsize=9)
    else:
        ax_B.text(0.5, 0.5, "no B field in result", transform=ax_B.transAxes,
                  ha="center", va="center", color="0.5", fontsize=10)
    ax_B.set_ylabel("B (nT)")
    ax_B.grid(True, which="major", axis="y", color="0.7", lw=0.5, alpha=0.45)

    # --- n panel (+ optional T overlay) ------------------------------------
    if n_present:
        ax_n.set_yscale("log")
        _lineplot(ax_n, df["n"].to_numpy(float), _N_COLOR, lw=0.8)
        ax_n.set_ylabel("n (cm^-3)", color=_N_COLOR)
        ax_n.grid(True, which="both", axis="y", color="0.7", lw=0.5, alpha=0.45)
    else:
        ax_n.set_ylabel("n (cm^-3)")
    if temp_arr is not None:
        ax_T = ax_n.twinx()
        ax_T.set_yscale("log")
        Ti = np.where(temp_arr > 0, temp_arr, np.nan)
        _lineplot(ax_T, Ti, _T_COLOR, lw=0.8)
        ax_T.set_ylabel("T_i (eV)", color=_T_COLOR)
        ax_T.tick_params(axis="y", labelcolor=_T_COLOR)

    # --- Classification strips (raw + 1-hour) -------------------------------
    id_to_code = {rid: i for i, rid in enumerate(_STRIP_IDS)}
    strip_cmap = ListedColormap([_REGION_COLORS[r] for r in _STRIP_IDS])
    strip_cmap.set_bad("white")
    code_raw = np.array([id_to_code.get(int(r), np.nan) for r in region_id],
                        dtype=float)

    Y_RAW = (0.55, 1.00)
    Y_1H = (0.00, 0.45)
    y_raw_c = (Y_RAW[0] + Y_RAW[1]) / 2.0
    y_1h_c = (Y_1H[0] + Y_1H[1]) / 2.0

    for s, e in zip(pc_starts, pc_ends):
        if e - s < 1:
            continue
        edges = _chunk_edges(tnum[s:e], pc_gap / 4.0)
        ax_cls.pcolormesh(
            edges, np.array(Y_RAW), code_raw[s:e][np.newaxis, :],
            cmap=strip_cmap, vmin=0, vmax=len(_STRIP_IDS) - 1, shading="flat",
        )

    # Persistence strip: the consecutive-region filter. region_timeline keeps
    # only samples whose contiguous same-region run lasts at least
    # `min_duration` (short-lived / flickering labels are dropped; runs are
    # broken only by a region change, not by data gaps), so the strip is a STRICT
    # SUBSET of raw. It is still drawn the SAME gap-aware way as the raw strip:
    # removed samples become NaN and render white via the colormap's set_bad, and
    # real data gaps break the pcolormesh chunks so absences still show white.
    kept = region_timeline(df, min_duration=min_duration)
    kept_mask = df.index.isin(kept.index)
    code_1h = np.where(kept_mask, code_raw, np.nan)
    for s, e in zip(pc_starts, pc_ends):
        if e - s < 1:
            continue
        edges = _chunk_edges(tnum[s:e], pc_gap / 4.0)
        ax_cls.pcolormesh(
            edges, np.array(Y_1H), code_1h[s:e][np.newaxis, :],
            cmap=strip_cmap, vmin=0, vmax=len(_STRIP_IDS) - 1, shading="flat",
        )

    ax_cls.set_yticks([y_1h_c, y_raw_c])
    ax_cls.set_yticklabels([str(min_duration), "raw"])
    ax_cls.set_ylim(0, 1)

    handles = [Patch(facecolor=_REGION_COLORS[r], label=_REGION_ABBR[r])
               for r in _LEGEND_ORDER]
    ax_cls.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.45), ncol=len(handles), fontsize=10,
                  frameon=False, handlelength=1.6, columnspacing=1.8)

    # --- Trajectory panel ---------------------------------------------------
    if have_traj:
        _plot_trajectory(ax_traj, position, df, date2num)

    # --- Shared time axis ---------------------------------------------------
    all_axes = [ax_spec, ax_B, ax_n, ax_cls]
    for ax in all_axes:
        ax.set_xlim(tnum[0], tnum[-1])
        ax.grid(True, which="major", axis="x", color="0.55", lw=0.6, alpha=0.5)
    ax_cls.xaxis.set_major_locator(AutoDateLocator())
    ax_cls.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
    ax_cls.set_xlabel("UT")
    for ax in all_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    if title is None:
        title = (
            "ARTEMIS plasma-region classification: "
            f"{times[0].strftime('%Y-%m-%d %H:%M')} to "
            f"{times[-1].strftime('%Y-%m-%d %H:%M')}"
        )
    fig.suptitle(title, fontsize=15, y=0.975)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _plot_trajectory(ax, position, label_df, date2num):
    """Draw the GSE X-Y trajectory panel with ported Shue MP + Chao BS curves."""
    if not isinstance(position, pd.DataFrame):
        raise TypeError("position must be a DataFrame with GSE X_GSE/Y_GSE/Z_GSE in R_E.")
    for c in ("X_GSE", "Y_GSE"):
        if c not in position.columns:
            raise ValueError(f"position is missing required column {c!r} (R_E).")

    posdf = position.sort_index()
    # Drop rows with a non-finite GSE X/Y (e.g. samples with no state match).
    finite_xy = (
        np.isfinite(posdf["X_GSE"].to_numpy(dtype=float))
        & np.isfinite(posdf["Y_GSE"].to_numpy(dtype=float))
    )
    posdf = posdf.loc[finite_xy]
    X = posdf["X_GSE"].to_numpy(dtype=float)
    Y = posdf["Y_GSE"].to_numpy(dtype=float)

    # Ported boundary curves (nominal solar-wind conditions): faint GSE-aligned
    # reference + full-strength aberrated (matches the reference figure).
    mp_x, mp_y = _shue_magnetopause_xy(bz=0.0, dp=2.0, x_min=-90.0)
    bs_x, bs_y = _chao_bowshock_xy(bz=0.2, dp=2.0, mgs=6.0, beta=1.0, x_min=-90.0)
    mp_xa, mp_ya = _aberration_rotate(mp_x, mp_y)
    bs_xa, bs_ya = _aberration_rotate(bs_x, bs_y)
    ax.plot(mp_x, mp_y, color="grey", lw=1.0, ls="--", alpha=0.5)
    ax.plot(bs_x, bs_y, color="grey", lw=1.0, ls="-", alpha=0.5)
    ax.plot(mp_xa, mp_ya, color="grey", lw=1.4, ls="--", label="Shue MP")
    ax.plot(bs_xa, bs_ya, color="grey", lw=1.4, ls="-", label="Chao BS")
    ax.plot(0, 0, marker="o", color="black", markersize=5)

    # Colour the orbit by the nearest-in-time region label.
    reg_at_pos = _nearest_region(label_df, posdf.index)
    colors = np.array([_REGION_COLORS.get(int(r), "#cccccc") for r in reg_at_pos])
    ax.plot(X, Y, color="lightgrey", lw=0.3, zorder=2)
    ax.scatter(X, Y, c=colors, s=4, alpha=0.75, zorder=3)

    # Midnight date markers along the orbit.
    t_ns = posdf.index.values.astype("datetime64[ns]").astype("int64")
    t0 = posdf.index[0].normalize()
    t1 = posdf.index[-1].normalize() + pd.Timedelta(days=1)
    six_h_ns = int(6 * 3600 * 1e9)
    mr_x, mr_y, mr_lbl = [], [], []
    for m in pd.date_range(t0, t1, freq="D"):
        m_ns = m.value
        if m_ns < t_ns[0] or m_ns > t_ns[-1]:
            continue
        idx = int(np.argmin(np.abs(t_ns - m_ns)))
        if abs(t_ns[idx] - m_ns) <= six_h_ns:
            mr_x.append(X[idx]); mr_y.append(Y[idx]); mr_lbl.append(m.strftime("%m-%d"))
    if mr_x:
        ax.scatter(mr_x, mr_y, marker="s", facecolor="white", edgecolor="black",
                   s=40, zorder=5, linewidth=0.8)
        for lab, xx, yy in zip(mr_lbl, mr_x, mr_y):
            ax.annotate(lab, (xx, yy), xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="black",
                        bbox=dict(facecolor="white", edgecolor="lightgrey",
                                  alpha=0.85, pad=1.0, lw=0.3))

    pad = 5.0
    if X.size:
        ax.set_xlim(min(X.min(), -65) - pad, max(X.max(), 15) + pad)
        ax.set_ylim(min(Y.min(), -25) - pad, max(Y.max(), 25) + pad)
    else:
        ax.set_xlim(-70, 20)
        ax.set_ylim(-30, 30)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X_GSE (R_E)")
    ax.set_ylabel("Y_GSE (R_E)")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_title("Trajectory (XY GSE, Abr 4°)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def _nearest_region(label_df, pos_index) -> np.ndarray:
    """Nearest-in-time ``region_id`` for each position timestamp (merge_asof)."""
    left = pd.DataFrame({"time": pd.DatetimeIndex(pos_index)}).sort_values("time")
    right = pd.DataFrame(
        {"time": label_df.index, "region_id": label_df["region_id"].to_numpy(int)}
    ).sort_values("time")
    merged = pd.merge_asof(left, right, on="time", direction="nearest")
    return merged["region_id"].fillna(-1).to_numpy(dtype=int)
