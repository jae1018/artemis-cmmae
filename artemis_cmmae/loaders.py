"""
pyspedas / cdflib I/O layer for the ARTEMIS plasma classifier.

This module downloads (or reads local) THEMIS/ARTEMIS ESA and FGM L2 CDFs and
assembles them into the duck-typed ``ds`` mappings consumed by
:func:`artemis_cmmae.pipeline.build_features_from_ds`. ``pyspedas`` and
``cdflib`` are required dependencies of ``artemis-cmmae`` but are imported
lazily inside the functions, so merely importing :mod:`artemis_cmmae` does not
pull them in until a loader actually runs.

Two entry points, :func:`load_esa` and :func:`load_fgm`, each support an offline
``files=`` bypass (read the given local CDF paths directly with ``cdflib`` and
skip pyspedas entirely) as well as the network path (``pyspedas.projects.themis``
``downloadonly=True`` -> read the returned paths with ``cdflib``).

Only ``numpy`` and ``pandas`` are hard top-level imports.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .features import epoch_to_datetime64

#: km per Earth radius used to convert GSE position (km) -> R_E.
_KM_PER_RE = 6371.2

# Actionable hint shown if a required dependency is somehow not importable.
_EXTRA_HINT = (
    "pyspedas / cdflib could not be imported. They are required dependencies "
    "of artemis-cmmae; reinstall it with: pip install artemis-cmmae"
)


# ---------------------------------------------------------------------------
# Lazy-import guards
# ---------------------------------------------------------------------------
def _require_cdflib():
    """Import and return ``cdflib`` or raise a clear missing-dependency error."""
    try:
        import cdflib
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_EXTRA_HINT) from exc
    return cdflib


def _require_pyspedas():
    """Import and return ``(themis, cdflib)`` or raise a clear error.

    Raises:
        ImportError: If either ``pyspedas`` or ``cdflib`` is not importable
            (they are required dependencies of ``artemis-cmmae``).
    """
    try:
        from pyspedas.projects import themis
        import cdflib
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc
    return themis, cdflib


# ---------------------------------------------------------------------------
# Probe normalization
# ---------------------------------------------------------------------------
def _normalize_probe(probe: str):
    """Normalize a probe token to ``(pyspedas_probe, var_prefix)``.

    Accepts 'thb'/'thc' or bare 'b'/'c' (and any THEMIS letter a-e). Returns the
    single-letter pyspedas probe id (e.g. 'b') and the CDF variable prefix
    (e.g. 'thb').

    Args:
        probe: Probe token, case-insensitive.

    Returns:
        Tuple ``(pyspedas_probe, var_prefix)``, e.g. ``('b', 'thb')``.

    Raises:
        ValueError: If the token is not a recognizable THEMIS probe.
    """
    p = str(probe).strip().lower()
    letter = p[2:] if p.startswith("th") else p
    if letter not in ("a", "b", "c", "d", "e"):
        raise ValueError(
            f"Unrecognized probe {probe!r}; expected 'thb'/'thc' or 'b'/'c' "
            "(THEMIS letters a-e)."
        )
    return letter, "th" + letter


# ---------------------------------------------------------------------------
# CDF read helpers
# ---------------------------------------------------------------------------
def _cdf_has_var(cdf, name: str) -> bool:
    """True if ``name`` is a variable in an open ``cdflib.CDF`` object."""
    try:
        info = cdf.cdf_info()
        return name in getattr(info, "zVariables", []) or name in getattr(
            info, "rVariables", []
        )
    except Exception:  # noqa: BLE001 - defensive; treat unknown as absent
        return False


def _gse_to_gsm(gse: np.ndarray, t_epoch: np.ndarray) -> np.ndarray:
    """Rotate a GSE (T, 3) vector array to GSM using pyspedas cotrans.

    Fallback used only when a ``*_gsm`` variable is NOT present directly in the
    CDF (the ARTEMIS L2 FGM / L1 state products carry GSM natively, so this path
    is normally never taken). Uses ``pyspedas``'s cotrans routines.

    Args:
        gse: ``(T, 3)`` GSE vectors (nT for B, km for position).
        t_epoch: ``(T,)`` Unix epoch seconds paired row-wise with ``gse``.

    Returns:
        ``(T, 3)`` numpy array of the same vectors expressed in GSM.
    """
    try:
        from pyspedas.cotrans_tools.cotrans_lib import subgse2gsm
    except ImportError as exc:  # pragma: no cover - pyspedas import guard
        raise ImportError(_EXTRA_HINT) from exc
    gse = np.asarray(gse, dtype=float)
    t = np.asarray(t_epoch, dtype=float)
    out = subgse2gsm(t, gse)
    return np.asarray(out, dtype=float).reshape(gse.shape)


def _read_time(cdf, time_name: str, ref_var: str) -> np.ndarray:
    """Read a time variable (Unix epoch seconds), falling back to DEPEND_0."""
    try:
        t = np.asarray(cdf.varget(time_name), dtype=np.float64)
    except Exception:  # noqa: BLE001 - varget raises heterogeneous errors
        atts = cdf.varattsget(ref_var)
        t = np.asarray(cdf.varget(atts["DEPEND_0"]), dtype=np.float64)
    return t.ravel()


def _time_clip_mask(t_epoch: np.ndarray, trange) -> np.ndarray:
    """Boolean mask selecting epoch-second records within ``trange`` inclusive."""
    t = np.asarray(t_epoch, dtype=float)
    t0 = pd.Timestamp(trange[0]).timestamp()
    t1 = pd.Timestamp(trange[1]).timestamp()
    return (t >= t0) & (t <= t1)


def _resolve_paths(files, themis_fn, pyspedas_probe, trange, no_update):
    """Return the list of CDF paths to read (offline ``files`` or a download).

    When downloading, the location is chosen entirely by pyspedas from its own
    config (the ``SPEDAS_DATA_DIR`` env var, read at pyspedas import time).
    """
    if files is not None:
        cdflib = _require_cdflib()
        return list(files), cdflib
    themis, cdflib = _require_pyspedas()
    dl = themis_fn(
        themis,
        trange=trange,
        probe=pyspedas_probe,
        level="l2",
        downloadonly=True,
        no_update=no_update,
    )
    paths = [str(p) for p in (dl or []) if str(p).lower().endswith(".cdf")]
    return paths, cdflib


# ---------------------------------------------------------------------------
# ESA loader
# ---------------------------------------------------------------------------
def load_esa(
    trange=None,
    probe: str = "thb",
    *,
    files: Optional[Sequence[str]] = None,
    time_res: str = "peif",
    no_update: bool = False,
):
    """Load a THEMIS/ARTEMIS ESA L2 product into a duck-typed ``ds`` dict.

    If ``files`` is given, those CDF paths are read directly with ``cdflib`` and
    pyspedas is skipped entirely (offline / power-user path). Otherwise
    ``pyspedas.projects.themis.esa(..., downloadonly=True)`` resolves the file
    paths, which are then read with ``cdflib``.

    The returned mapping carries exactly the variables
    :func:`~artemis_cmmae.pipeline.build_features_from_ds` expects:
    ``<p>_<time_res>_en_eflux`` (T, 32), ``<p>_<time_res>_en_eflux_yaxis`` (T, 32),
    ``<p>_<time_res>_density`` (T,), ``<p>_<time_res>_sc_pot`` (T,),
    ``<p>_<time_res>_avgtemp`` (T, ion temperature eV), and
    ``<p>_<time_res>_time`` (T,) as Unix epoch seconds. Records are concatenated
    across day-files, time-sorted, and clipped to ``trange`` if given.

    Args:
        trange: ``[start, end]`` used for download and time-clipping (or None).
        probe: 'thb'/'thc' or 'b'/'c' (or a THEMIS letter a-e).
        files: Optional local CDF paths (bypasses pyspedas).
        time_res: ESA product prefix (only 'peif' is supported downstream).
        no_update: If True, use only local (already-downloaded) files; default
            False downloads any missing files.

    The download directory follows pyspedas's own config: set the
    ``SPEDAS_DATA_DIR`` environment variable (read at pyspedas import time)
    before first use to control where CDFs are cached.

    Returns:
        ``dict`` of numpy arrays keyed by the native CDF variable names.
    """
    pyspedas_probe, var_prefix = _normalize_probe(probe)
    paths, cdflib = _resolve_paths(
        files, lambda th, **kw: th.esa(**kw), pyspedas_probe, trange, no_update
    )
    if not paths:
        raise FileNotFoundError(
            f"load_esa: no ESA CDF files to read for probe {probe!r} "
            f"(trange={trange}). Check the trange or pass files=."
        )

    efs, yxs, dns, sps, tps, tms = [], [], [], [], [], []
    for pth in sorted(paths):
        cdf = cdflib.CDF(pth)
        efs.append(np.asarray(cdf.varget(f"{var_prefix}_{time_res}_en_eflux"), dtype=float))
        yxs.append(np.asarray(cdf.varget(f"{var_prefix}_{time_res}_en_eflux_yaxis"), dtype=float))
        dns.append(np.asarray(cdf.varget(f"{var_prefix}_{time_res}_density"), dtype=float).ravel())
        sps.append(np.asarray(cdf.varget(f"{var_prefix}_{time_res}_sc_pot"), dtype=float).ravel())
        # Ion average temperature (eV): output-only context, not a model input.
        # Optional -- a file/product without it just contributes NaN.
        try:
            tps.append(np.asarray(cdf.varget(f"{var_prefix}_{time_res}_avgtemp"), dtype=float).ravel())
        except Exception:  # noqa: BLE001 - varget raises heterogeneous errors
            tps.append(np.full(dns[-1].shape[0], np.nan))
        tms.append(_read_time(cdf, f"{var_prefix}_{time_res}_time", f"{var_prefix}_{time_res}_en_eflux"))

    ef = np.vstack(efs)
    yax = np.vstack(yxs)
    dens = np.concatenate(dns)
    scp = np.concatenate(sps)
    temp = np.concatenate(tps)
    t = np.concatenate(tms)

    order = np.argsort(t, kind="stable")
    ef, yax, dens, scp, temp, t = (
        ef[order], yax[order], dens[order], scp[order], temp[order], t[order]
    )
    if trange is not None:
        keep = _time_clip_mask(t, trange)
        ef, yax, dens, scp, temp, t = (
            ef[keep], yax[keep], dens[keep], scp[keep], temp[keep], t[keep]
        )

    return {
        f"{var_prefix}_{time_res}_en_eflux": ef,
        f"{var_prefix}_{time_res}_en_eflux_yaxis": yax,
        f"{var_prefix}_{time_res}_density": dens,
        f"{var_prefix}_{time_res}_sc_pot": scp,
        f"{var_prefix}_{time_res}_avgtemp": temp,
        f"{var_prefix}_{time_res}_time": t,
    }


# ---------------------------------------------------------------------------
# FGM loader
# ---------------------------------------------------------------------------
def load_fgm(
    trange=None,
    probe: str = "thb",
    *,
    files: Optional[Sequence[str]] = None,
    b_source: str = "fgs",
    no_update: bool = False,
):
    """Load a THEMIS/ARTEMIS FGM L2 product into a duck-typed ``ds`` dict.

    Same download/offline pattern as :func:`load_esa`. The returned mapping
    carries ``<p>_<b_source>_gse`` (T, 3) GSE magnetic field (nT),
    ``<p>_<b_source>_gsm`` (T, 3) GSM magnetic field (nT), and
    ``<p>_<b_source>_time`` (T,) Unix epoch seconds, concatenated across files,
    time-sorted, and clipped to ``trange`` if given.

    The GSM field is read DIRECTLY from the CDF (``<p>_<b_source>_gsm``) when
    present -- the ARTEMIS L2 FGM product carries it natively -- and is otherwise
    computed from the GSE field via pyspedas ``gse2gsm`` cotrans.

    Args:
        trange: ``[start, end]`` used for download and time-clipping (or None).
        probe: 'thb'/'thc' or 'b'/'c' (or a THEMIS letter a-e).
        files: Optional local CDF paths (bypasses pyspedas).
        b_source: FGM cadence: 'fgs' (default), 'fgl', or 'fgh'.
        no_update: If True, use only local (already-downloaded) files; default
            False downloads any missing files.

    The download directory follows pyspedas's own config: set the
    ``SPEDAS_DATA_DIR`` environment variable (read at pyspedas import time)
    before first use to control where CDFs are cached.

    Returns:
        ``dict`` of numpy arrays keyed by the native CDF variable names
        (``<p>_<b_source>_gse``, ``<p>_<b_source>_gsm``, ``<p>_<b_source>_time``).
    """
    if b_source not in ("fgs", "fgl", "fgh"):
        raise ValueError(f"b_source must be 'fgs', 'fgl', or 'fgh', got {b_source!r}")

    pyspedas_probe, var_prefix = _normalize_probe(probe)
    paths, cdflib = _resolve_paths(
        files, lambda th, **kw: th.fgm(**kw), pyspedas_probe, trange, no_update
    )
    if not paths:
        raise FileNotFoundError(
            f"load_fgm: no FGM CDF files to read for probe {probe!r} "
            f"(trange={trange}). Check the trange or pass files=."
        )

    gsm_name = f"{var_prefix}_{b_source}_gsm"
    gse_list, gsm_list, t_list = [], [], []
    have_gsm_direct = True
    for pth in sorted(paths):
        cdf = cdflib.CDF(pth)
        gse_list.append(np.asarray(cdf.varget(f"{var_prefix}_{b_source}_gse"), dtype=float))
        if have_gsm_direct and _cdf_has_var(cdf, gsm_name):
            gsm_list.append(np.asarray(cdf.varget(gsm_name), dtype=float))
        else:
            have_gsm_direct = False
        t_list.append(_read_time(cdf, f"{var_prefix}_{b_source}_time", f"{var_prefix}_{b_source}_gse"))

    gse = np.vstack(gse_list)
    t = np.concatenate(t_list)

    order = np.argsort(t, kind="stable")
    gse, t = gse[order], t[order]
    if have_gsm_direct:
        gsm = np.vstack(gsm_list)[order]
    else:
        gsm = _gse_to_gsm(gse, t)
    if trange is not None:
        keep = _time_clip_mask(t, trange)
        gse, gsm, t = gse[keep], gsm[keep], t[keep]

    return {
        f"{var_prefix}_{b_source}_gse": gse,
        f"{var_prefix}_{b_source}_gsm": gsm,
        f"{var_prefix}_{b_source}_time": t,
    }


# ---------------------------------------------------------------------------
# State / position loader
# ---------------------------------------------------------------------------
def _resolve_state_paths(files, pyspedas_probe, trange, no_update):
    """Return the list of state CDF paths to read (offline ``files`` or a download).

    Like :func:`_resolve_paths` but calls ``pyspedas.projects.themis.state`` with
    its native ``level='l1'`` (THEMIS spacecraft ephemeris lives in the L1 state
    product, not L2). When downloading, the location is chosen entirely by
    pyspedas from its own config (the ``SPEDAS_DATA_DIR`` env var).
    """
    if files is not None:
        cdflib = _require_cdflib()
        return list(files), cdflib
    themis, cdflib = _require_pyspedas()
    dl = themis.state(
        trange=trange,
        probe=pyspedas_probe,
        level="l1",
        downloadonly=True,
        no_update=no_update,
    )
    paths = [str(p) for p in (dl or []) if str(p).lower().endswith(".cdf")]
    return paths, cdflib


def load_position(
    trange=None,
    probe: str = "thb",
    *,
    files: Optional[Sequence[str]] = None,
    no_update: bool = False,
):
    """Load THEMIS/ARTEMIS spacecraft GSE position (L1 state) into a ds dict.

    Same download/offline pattern as :func:`load_esa` / :func:`load_fgm`, but
    reads the L1 *state* product via
    ``pyspedas.projects.themis.state(..., downloadonly=True)`` (or the given
    ``files``) with ``cdflib``. The returned mapping carries
    ``<p>_pos_gse`` (T, 3) GSE position and ``<p>_pos_gsm`` (T, 3) GSM position,
    both in **kilometres** (the native CDF unit), and ``<p>_state_time`` (T,)
    Unix epoch seconds, concatenated across files, time-sorted, and clipped to
    ``trange`` if given.

    The GSM position is read DIRECTLY from the CDF (``<p>_pos_gsm``) when present
    -- the ARTEMIS L1 state product carries it natively -- and is otherwise
    computed from the GSE position via pyspedas ``gse2gsm`` cotrans.

    Use :func:`load_position_frame` for a ready-to-plot ``DataFrame`` with the
    position already converted to Earth radii (R_E).

    Args:
        trange: ``[start, end]`` used for download and time-clipping (or None).
        probe: 'thb'/'thc' or 'b'/'c' (or a THEMIS letter a-e).
        files: Optional local state CDF paths (bypasses pyspedas).
        no_update: If True, use only local (already-downloaded) files; default
            False downloads any missing files.

    The download directory follows pyspedas's own config: set the
    ``SPEDAS_DATA_DIR`` environment variable (read at pyspedas import time)
    before first use to control where CDFs are cached.

    Returns:
        ``dict`` with ``<p>_pos_gse`` (T, 3) GSE position in km,
        ``<p>_pos_gsm`` (T, 3) GSM position in km, and ``<p>_state_time`` (T,)
        Unix epoch seconds.
    """
    pyspedas_probe, var_prefix = _normalize_probe(probe)
    paths, cdflib = _resolve_state_paths(
        files, pyspedas_probe, trange, no_update
    )
    if not paths:
        raise FileNotFoundError(
            f"load_position: no state CDF files to read for probe {probe!r} "
            f"(trange={trange}). Check the trange or pass files=."
        )

    gsm_name = f"{var_prefix}_pos_gsm"
    pos_list, posgsm_list, t_list = [], [], []
    have_gsm_direct = True
    for pth in sorted(paths):
        cdf = cdflib.CDF(pth)
        pos_list.append(np.asarray(cdf.varget(f"{var_prefix}_pos_gse"), dtype=float))
        if have_gsm_direct and _cdf_has_var(cdf, gsm_name):
            posgsm_list.append(np.asarray(cdf.varget(gsm_name), dtype=float))
        else:
            have_gsm_direct = False
        # <p>_state_time is Unix epoch seconds; fall back to the pos DEPEND_0 if
        # the name is ever absent.
        t_list.append(
            _read_time(cdf, f"{var_prefix}_state_time", f"{var_prefix}_pos_gse")
        )

    pos = np.vstack(pos_list)
    t = np.concatenate(t_list)

    order = np.argsort(t, kind="stable")
    pos, t = pos[order], t[order]
    if have_gsm_direct:
        pos_gsm = np.vstack(posgsm_list)[order]
    else:
        pos_gsm = _gse_to_gsm(pos, t)
    if trange is not None:
        keep = _time_clip_mask(t, trange)
        pos, pos_gsm, t = pos[keep], pos_gsm[keep], t[keep]

    return {
        f"{var_prefix}_pos_gse": pos,
        f"{var_prefix}_pos_gsm": pos_gsm,
        f"{var_prefix}_state_time": t,
    }


def load_position_frame(
    trange=None,
    probe: str = "thb",
    *,
    files: Optional[Sequence[str]] = None,
    no_update: bool = False,
) -> pd.DataFrame:
    """Load spacecraft position as a clean ``DataFrame`` in Earth radii.

    Thin convenience wrapper over :func:`load_position` that converts the native
    kilometre position to R_E (dividing by ``6371.2 km``) and returns exactly the
    shape :func:`artemis_cmmae.plotting.plot_classification_timeseries` expects
    for its ``position`` argument: a ``DatetimeIndex`` (named 'time') with columns
    ``X_GSE``/``Y_GSE``/``Z_GSE`` (GSE, R_E) and ``X_GSM``/``Y_GSM``/``Z_GSM``
    (GSM, R_E).

    Args:
        trange: ``[start, end]`` used for download and time-clipping (or None).
        probe: 'thb'/'thc' or 'b'/'c' (or a THEMIS letter a-e).
        files: Optional local state CDF paths (bypasses pyspedas).
        no_update: If True, use only local files; default False downloads missing.

    The download directory follows pyspedas's own config (the
    ``SPEDAS_DATA_DIR`` environment variable, read at pyspedas import time).

    Returns:
        ``DataFrame`` indexed by a ``DatetimeIndex`` named 'time' with columns
        ``X_GSE``/``Y_GSE``/``Z_GSE`` and ``X_GSM``/``Y_GSM``/``Z_GSM`` (position
        in R_E).
    """
    _, var_prefix = _normalize_probe(probe)
    ds = load_position(
        trange=trange,
        probe=probe,
        files=files,
        no_update=no_update,
    )
    gse_re = np.asarray(ds[f"{var_prefix}_pos_gse"], dtype=float) / _KM_PER_RE
    gsm_re = np.asarray(ds[f"{var_prefix}_pos_gsm"], dtype=float) / _KM_PER_RE
    times = epoch_to_datetime64(ds[f"{var_prefix}_state_time"])
    return pd.DataFrame(
        {
            "X_GSE": gse_re[:, 0], "Y_GSE": gse_re[:, 1], "Z_GSE": gse_re[:, 2],
            "X_GSM": gsm_re[:, 0], "Y_GSM": gsm_re[:, 1], "Z_GSM": gsm_re[:, 2],
        },
        index=pd.DatetimeIndex(times, name="time"),
    )
