"""
xarray-Dataset ("ds") entry points for the ARTEMIS plasma classifier.

These functions turn a THEMIS/ARTEMIS PEIF ESA L2 dataset (plus optional FGM
magnetic-field data) into a time-indexed feature frame, and run the
``PlasmaClassifier`` end-to-end.

The core spectrum-prep recipe lives in :mod:`artemis_cmmae.features`; this module
adds the ds plumbing (variable discovery, time conversion, and the FGM->ESA merge)
and the two public entry points ``build_features_from_ds`` and
``predict_from_ds``.

xarray is imported lazily / duck-typed: there is no hard top-level ``import
xarray``. A ds is any mapping that supports ``ds[var]`` (returning either a numpy
array or an object with a ``.values`` attribute, such as an xarray ``DataArray``)
and membership testing. So a plain ``dict`` of numpy arrays works too.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from .features import (
    N_MODEL_CHANNELS,
    epoch_to_datetime64,
    flag_lunar_wake,
    prepare_ion_spectra,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime xarray dependency
    import xarray as xr


#: km per Earth radius used to convert GSE/GSM position (km) -> R_E.
_KM_PER_RE = 6371.2

# GSE magnetic-field component columns -- the ONLY B that ever reaches the model
# -- and their GSM counterparts (output-only context). GSM B is never a model
# input.
_B_COMPONENTS_GSE = ("BX_GSE", "BY_GSE", "BZ_GSE")
_B_COMPONENTS_GSM = ("BX_GSM", "BY_GSM", "BZ_GSM")

# Position columns (output-only context; never a model input), both frames in
# Earth radii (R_E).
_POS_COLS_GSE = ("X_GSE", "Y_GSE", "Z_GSE")
_POS_COLS_GSM = ("X_GSM", "Y_GSM", "Z_GSM")
_POS_COLS = _POS_COLS_GSE + _POS_COLS_GSM


# ---------------------------------------------------------------------------
# ds duck-typing helpers
# ---------------------------------------------------------------------------
def _all_var_names(ds) -> list:
    """Return all variable names in a ds (xarray Dataset or dict-like)."""
    names: list = []
    for attr in ("data_vars", "coords"):
        sub = getattr(ds, attr, None)
        if sub is not None:
            names.extend(list(sub))
    if not names:
        try:
            names = list(ds.keys())
        except AttributeError:
            names = list(ds)
    return names


def _has_var(ds, name: str) -> bool:
    """Membership test that works for xarray Datasets and dict-likes."""
    try:
        return name in ds
    except TypeError:
        return name in _all_var_names(ds)


def _get_values(ds, name: str) -> np.ndarray:
    """Read a variable's values as a numpy array (duck-typed)."""
    obj = ds[name]
    if hasattr(obj, "values"):  # xarray DataArray / pandas Series
        return np.asarray(obj.values)
    return np.asarray(obj)


def _detect_probe(ds) -> str:
    """Auto-detect the probe prefix (e.g. 'thb'/'thc') from ds variable names.

    Looks for a variable ending in ``_peif_en_eflux`` and returns its prefix.
    """
    for name in _all_var_names(ds):
        if name.endswith("_peif_en_eflux"):
            return name[: -len("_peif_en_eflux")]
    raise ValueError(
        "Could not auto-detect probe: no variable ending in '_peif_en_eflux' "
        "found in the dataset. Pass probe=... explicitly (e.g. 'thb' or 'thc')."
    )


# ---------------------------------------------------------------------------
# FGM -> ESA merge
# ---------------------------------------------------------------------------
def merge_fgm_onto_esa(
    esa_df: pd.DataFrame,
    fgm_ds,
    *,
    probe: Optional[str] = None,
    b_source: str = "fgs",
    tolerance: str = "60s",
) -> pd.DataFrame:
    """Merge magnetic-field components from an FGM ds onto an ESA feature frame.

    Uses a nearest-time ``merge_asof`` with the given tolerance. The FGM ds is
    expected to carry ``<probe>_<b_source>_gse`` (T, 3) and
    ``<probe>_<b_source>_time`` (Unix epoch seconds), and OPTIONALLY
    ``<probe>_<b_source>_gsm`` (T, 3) for the GSM field.

    Args:
        esa_df: Feature frame with a ``DatetimeIndex`` named 'time'.
        fgm_ds: FGM dataset (xarray Dataset or dict-like).
        probe: Probe prefix; auto-detected from ``fgm_ds`` if None.
        b_source: One of 'fgs', 'fgl', 'fgh' (the FGM cadence to use).
        tolerance: ``merge_asof`` nearest-time tolerance (e.g. '60s').

    Returns:
        A copy of ``esa_df`` with ``BX_GSE``/``BY_GSE``/``BZ_GSE`` columns added
        (GSE nT), plus ``BX_GSM``/``BY_GSM``/``BZ_GSM`` (GSM nT) when the FGM ds
        provides the GSM field. Rows without a match within ``tolerance`` get
        NaN B. Only the GSE columns are ever used as model input.
    """
    if b_source not in ("fgs", "fgl", "fgh"):
        raise ValueError(f"b_source must be 'fgs', 'fgl', or 'fgh', got {b_source!r}")

    if probe is None:
        # Detect from an FGM variable, e.g. '<p>_fgs_gse'.
        probe = None
        for name in _all_var_names(fgm_ds):
            if name.endswith(f"_{b_source}_gse"):
                probe = name[: -len(f"_{b_source}_gse")]
                break
        if probe is None:
            raise ValueError(
                f"Could not auto-detect probe from fgm_ds: no '*_{b_source}_gse' "
                "variable found. Pass probe=... explicitly."
            )

    gse_name = f"{probe}_{b_source}_gse"
    time_name = f"{probe}_{b_source}_time"
    if not _has_var(fgm_ds, gse_name):
        raise KeyError(f"fgm_ds is missing '{gse_name}'")
    if not _has_var(fgm_ds, time_name):
        raise KeyError(f"fgm_ds is missing '{time_name}'")

    gse = np.asarray(_get_values(fgm_ds, gse_name), dtype=float)
    if gse.ndim != 2 or gse.shape[1] < 3:
        raise ValueError(f"{gse_name} must be (T, 3); got shape {gse.shape}")
    b_times = epoch_to_datetime64(_get_values(fgm_ds, time_name))

    right_cols = {
        "time": pd.DatetimeIndex(b_times),
        "BX_GSE": gse[:, 0],
        "BY_GSE": gse[:, 1],
        "BZ_GSE": gse[:, 2],
    }
    # GSM field is optional context (never a model input); include it if present.
    gsm_name = f"{probe}_{b_source}_gsm"
    if _has_var(fgm_ds, gsm_name):
        gsm = np.asarray(_get_values(fgm_ds, gsm_name), dtype=float)
        if gsm.ndim == 2 and gsm.shape[1] >= 3:
            right_cols["BX_GSM"] = gsm[:, 0]
            right_cols["BY_GSM"] = gsm[:, 1]
            right_cols["BZ_GSM"] = gsm[:, 2]
    right = pd.DataFrame(right_cols).sort_values("time").reset_index(drop=True)
    return _asof_merge(esa_df, right, tolerance)


def _asof_merge(esa_df: pd.DataFrame, right: pd.DataFrame, tolerance: str) -> pd.DataFrame:
    """Nearest-time ``merge_asof`` of a time-sorted ``right`` onto ``esa_df``.

    ``right`` must have a ``'time'`` column (already time-sorted). The ESA frame's
    ORIGINAL row order is preserved on return (the merge is done on a temporary
    time-sorted copy, then un-sorted back). Rows with no ``right`` match within
    ``tolerance`` get NaN in the added columns.
    """
    left = esa_df.copy()
    left_index_name = left.index.name or "time"
    left = left.reset_index().rename(columns={left_index_name: "time"})
    order = np.argsort(left["time"].values, kind="stable")
    left_sorted = left.iloc[order].reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        right,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    # Restore original row order.
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    merged = merged.iloc[inv].reset_index(drop=True)

    merged = merged.set_index("time")
    merged.index.name = "time"
    return merged


def merge_position_onto_esa(
    esa_df: pd.DataFrame,
    state_ds,
    *,
    probe: Optional[str] = None,
    tolerance: str = "60s",
) -> pd.DataFrame:
    """Merge GSE + GSM spacecraft position (R_E) from a state ds onto an ESA frame.

    Uses a nearest-time ``merge_asof`` with the given tolerance. The state ds is
    expected to carry ``<probe>_pos_gse`` (T, 3, km), ``<probe>_state_time``
    (Unix epoch seconds), and OPTIONALLY ``<probe>_pos_gsm`` (T, 3, km). Position
    is converted to Earth radii (dividing by ``6371.2 km``).

    Args:
        esa_df: Feature frame with a ``DatetimeIndex`` named 'time'.
        state_ds: State/position dataset (xarray Dataset or dict-like).
        probe: Probe prefix; auto-detected from ``state_ds`` if None.
        tolerance: ``merge_asof`` nearest-time tolerance (e.g. '60s').

    Returns:
        A copy of ``esa_df`` with ``X_GSE``/``Y_GSE``/``Z_GSE`` columns added
        (R_E), plus ``X_GSM``/``Y_GSM``/``Z_GSM`` (R_E) when the state ds provides
        the GSM position. Position is output-only context -- never a model input.
    """
    if probe is None:
        for name in _all_var_names(state_ds):
            if name.endswith("_pos_gse"):
                probe = name[: -len("_pos_gse")]
                break
        if probe is None:
            raise ValueError(
                "Could not auto-detect probe from state_ds: no '*_pos_gse' "
                "variable found. Pass probe=... explicitly."
            )

    gse_name = f"{probe}_pos_gse"
    time_name = f"{probe}_state_time"
    if not _has_var(state_ds, gse_name):
        raise KeyError(f"state_ds is missing '{gse_name}'")
    if not _has_var(state_ds, time_name):
        raise KeyError(f"state_ds is missing '{time_name}'")

    gse = np.asarray(_get_values(state_ds, gse_name), dtype=float)
    if gse.ndim != 2 or gse.shape[1] < 3:
        raise ValueError(f"{gse_name} must be (T, 3); got shape {gse.shape}")
    p_times = epoch_to_datetime64(_get_values(state_ds, time_name))
    gse_re = gse / _KM_PER_RE

    right_cols = {
        "time": pd.DatetimeIndex(p_times),
        "X_GSE": gse_re[:, 0],
        "Y_GSE": gse_re[:, 1],
        "Z_GSE": gse_re[:, 2],
    }
    gsm_name = f"{probe}_pos_gsm"
    if _has_var(state_ds, gsm_name):
        gsm = np.asarray(_get_values(state_ds, gsm_name), dtype=float)
        if gsm.ndim == 2 and gsm.shape[1] >= 3:
            gsm_re = gsm / _KM_PER_RE
            right_cols["X_GSM"] = gsm_re[:, 0]
            right_cols["Y_GSM"] = gsm_re[:, 1]
            right_cols["Z_GSM"] = gsm_re[:, 2]
    right = pd.DataFrame(right_cols).sort_values("time").reset_index(drop=True)
    return _asof_merge(esa_df, right, tolerance)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def build_features_from_ds(
    ds,
    *,
    probe: Optional[str] = None,
    fgm_ds=None,
    state_ds=None,
    b_source: str = "fgs",
    time_res: str = "peif",
    merge_tolerance: str = "60s",
    drop_top_channel: bool = True,
    filter_out_wake: bool = True,
    wake_scpot_threshold: float = 1.0,
    wake_pad: int = 1,
    drop_wake: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a classifier-ready feature frame from a PEIF ESA dataset.

    Expected ``ds`` variables (pyspedas / ``cdflib.cdf_to_xarray`` native names,
    ``<p>`` = probe prefix such as 'thb'):

    * ``<p>_peif_en_eflux`` (T, 32) linear ion energy flux (descending energy).
    * ``<p>_peif_en_eflux_yaxis`` (T, 32) per-record energy axis (eV).
    * ``<p>_peif_density`` (T,) density (cm^-3).
    * ``<p>_peif_sc_pot`` (T,) spacecraft potential (V).
    * ``<p>_peif_time`` (T,) Unix epoch seconds (converted via ``arr*1e9 ->
      int64 -> view datetime64[ns]``), or an already-``datetime64`` array.

    For the magnetic field either the GSE components ``BX_GSE``/``BY_GSE``/
    ``BZ_GSE`` (nT) are already present in ``ds`` (as those columns, or a combined
    ``<p>_<b_source>_gse`` variable), or a separate ``fgm_ds`` is supplied carrying
    ``<p>_<b_source>_gse`` (+ ``<p>_<b_source>_time``) and merged internally with
    a nearest-time ``merge_asof`` (tolerance ``merge_tolerance``). The GSM field
    (``BX_GSM``/``BY_GSM``/``BZ_GSM``, or ``<p>_<b_source>_gsm``) is carried
    through as OUTPUT-ONLY context when available; only the GSE B ever feeds the
    model. Likewise, spacecraft position (``X_GSE``/``Y_GSE``/``Z_GSE`` and
    ``X_GSM``/``Y_GSM``/``Z_GSM``, R_E) is added as output-only context when a
    ``state_ds`` is supplied (or those columns are already in ``ds``).

    Two data-hygiene steps run after the spectra are assembled:

    * **Lunar-wake flag** (``in_wake`` column). The paper removes wake-affected
      data via ``SCPot <= wake_scpot_threshold`` (1 V) dilated by ``wake_pad``
      adjacent time steps each side (:func:`~artemis_cmmae.features.flag_lunar_wake`).
      The flag is computed on the TIME-SORTED records (so adjacency reflects true
      temporal neighbours) and BEFORE the NaN-drop below.
    * **NaN / invalid-scalar drop.** Rows with a non-finite value in any of
      ``{SCPot, n, BX_GSE, BY_GSE, BZ_GSE}`` (whichever are present; GSM B and
      position never gate the drop) or with ``n <= 0`` are
      DROPPED: the classifier applies ``log10(n)`` and PCA, which cannot consume
      NaN or non-positive density. Spectrum-channel NaNs are NOT a drop reason
      (``prepare_ion_spectra`` fills them with 0). Dropped rows are simply removed
      -- they are never relabelled; ``region_id = -1`` is reserved for the MODEL's
      own "Unknown" (a point mapping to an unlabeled SOM node).

    ``filter_out_wake`` vs ``drop_wake`` -- what each does:

    * ``filter_out_wake`` (default True) toggles whether the ``in_wake`` column is
      actually POPULATED with the wake detection. The column is always returned
      (schema stability); when ``filter_out_wake=False`` it is present but
      all-False (wake detection disabled, informational only).
    * ``drop_wake`` (default False) is a convenience that REMOVES flagged rows from
      the returned frame. Setting ``drop_wake=True`` forces the wake flag to be
      computed even if ``filter_out_wake=False`` (so the drop is never a silent
      no-op).

    Args:
        ds: The PEIF ESA dataset (xarray Dataset or dict-like).
        probe: Probe prefix; auto-detected from the ds variable names if None.
        fgm_ds: Optional separate FGM dataset for the B-field merge.
        state_ds: Optional separate state/position dataset for the GSE+GSM
            position merge (output-only context; never a model input).
        b_source: FGM cadence to merge from: 'fgs' (default), 'fgl', or 'fgh'.
        time_res: ESA product to use; only 'peif' is currently supported.
        merge_tolerance: Nearest-time tolerance for the FGM merge.
        drop_top_channel: Drop the highest-energy channel (C31) to yield C0..C30.
        filter_out_wake: If True (default), populate the ``in_wake`` column via the
            lunar-wake rule; if False, ``in_wake`` is returned all-False.
        wake_scpot_threshold: Wake potential threshold (V); ``SCPot <=`` this is
            wake. Paper value: 1.0.
        wake_pad: Adjacent time steps flagged on EACH side of a wake sample. Paper
            value: 1.
        drop_wake: If True, drop the wake-flagged rows from the returned frame.
        verbose: If True (default), print one line with the NaN/invalid drop count.

    Returns:
        A ``DataFrame`` indexed by a ``DatetimeIndex`` named 'time' with columns
        C0..C30 (linear eflux, ascending energy), ``n`` (linear cm^-3),
        ``SCPot`` (V), (if available) ``BX_GSE``/``BY_GSE``/``BZ_GSE`` (GSE nT,
        the model input), optional ``BX_GSM``/``BY_GSM``/``BZ_GSM`` (GSM nT) and
        position ``X_GSE``/``Y_GSE``/``Z_GSE`` + ``X_GSM``/``Y_GSM``/``Z_GSM``
        (R_E) as output-only context, and a boolean ``in_wake`` column.

    Raises:
        NotImplementedError: If ``time_res`` is not 'peif'.
        ValueError: On unknown ``b_source`` or if the probe cannot be detected.
        KeyError: If a required PEIF variable is missing from ``ds``.
    """
    if time_res != "peif":
        raise NotImplementedError(
            f"time_res={time_res!r} is not supported yet; only 'peif' is "
            "implemented."
        )
    if b_source not in ("fgs", "fgl", "fgh"):
        raise ValueError(f"b_source must be 'fgs', 'fgl', or 'fgh', got {b_source!r}")

    if probe is None:
        probe = _detect_probe(ds)

    eflux_name = f"{probe}_peif_en_eflux"
    energy_name = f"{probe}_peif_en_eflux_yaxis"
    density_name = f"{probe}_peif_density"
    scpot_name = f"{probe}_peif_sc_pot"
    time_name = f"{probe}_peif_time"
    for name in (eflux_name, energy_name, density_name, scpot_name, time_name):
        if not _has_var(ds, name):
            raise KeyError(
                f"Dataset is missing required variable '{name}' for probe "
                f"'{probe}'."
            )

    eflux = np.asarray(_get_values(ds, eflux_name), dtype=float)
    energy = np.asarray(_get_values(ds, energy_name), dtype=float)
    density = np.asarray(_get_values(ds, density_name), dtype=float).ravel()
    scpot = np.asarray(_get_values(ds, scpot_name), dtype=float).ravel()
    times = epoch_to_datetime64(_get_values(ds, time_name))

    C = prepare_ion_spectra(eflux, energy, drop_top_channel=drop_top_channel)
    n_channels = C.shape[1]
    ch_cols = [f"C{i}" for i in range(n_channels)]

    df = pd.DataFrame(C, columns=ch_cols)
    df["n"] = density
    df["SCPot"] = scpot
    # Ion temperature (eV) -- output-only context for plotting / analysis, NOT a
    # model input and NOT a drop criterion. Present when the ds comes from
    # load_esa (which reads <p>_peif_avgtemp); skipped for minimal datasets.
    temp_name = f"{probe}_peif_avgtemp"
    if _has_var(ds, temp_name):
        df["T"] = np.asarray(_get_values(ds, temp_name), dtype=float).ravel()
    df.index = pd.DatetimeIndex(times, name="time")

    # --- Magnetic field ------------------------------------------------------
    if fgm_ds is not None:
        df = merge_fgm_onto_esa(
            df, fgm_ds, probe=probe, b_source=b_source, tolerance=merge_tolerance
        )
    else:
        # GSE B (the model input) may already be present in the ds as
        # BX_GSE/BY_GSE/BZ_GSE columns, or as a combined <p>_<b_source>_gse
        # variable.
        if all(_has_var(ds, comp) for comp in _B_COMPONENTS_GSE):
            for comp in _B_COMPONENTS_GSE:
                df[comp] = np.asarray(_get_values(ds, comp), dtype=float).ravel()
        else:
            gse_name = f"{probe}_{b_source}_gse"
            if _has_var(ds, gse_name):
                gse = np.asarray(_get_values(ds, gse_name), dtype=float)
                df["BX_GSE"] = gse[:, 0]
                df["BY_GSE"] = gse[:, 1]
                df["BZ_GSE"] = gse[:, 2]
        # GSM B is output-only context: BX_GSM/BY_GSM/BZ_GSM columns, or a
        # combined <p>_<b_source>_gsm variable. Never a model input.
        if all(_has_var(ds, comp) for comp in _B_COMPONENTS_GSM):
            for comp in _B_COMPONENTS_GSM:
                df[comp] = np.asarray(_get_values(ds, comp), dtype=float).ravel()
        else:
            gsm_name = f"{probe}_{b_source}_gsm"
            if _has_var(ds, gsm_name):
                gsm = np.asarray(_get_values(ds, gsm_name), dtype=float)
                df["BX_GSM"] = gsm[:, 0]
                df["BY_GSM"] = gsm[:, 1]
                df["BZ_GSM"] = gsm[:, 2]

    # --- Position (output-only context; GSE + GSM, R_E) ----------------------
    if state_ds is not None:
        df = merge_position_onto_esa(
            df, state_ds, probe=probe, tolerance=merge_tolerance
        )
    else:
        # Position columns (X_GSE.. / X_GSM..) may already be present in the ds.
        for comp in _POS_COLS:
            if _has_var(ds, comp) and comp not in df.columns:
                df[comp] = np.asarray(_get_values(ds, comp), dtype=float).ravel()

    # --- Lunar-wake flag -----------------------------------------------------
    # Computed on the TIME-SORTED order so adjacency reflects true temporal
    # neighbours, then mapped back to the original row order. Done BEFORE the
    # NaN-drop below so wake adjacency is not corrupted by row removals.
    scpot_vals = df["SCPot"].to_numpy(dtype=float)
    in_wake = np.zeros(len(df), dtype=bool)
    if filter_out_wake or drop_wake:
        time_vals = df.index.values
        order = np.argsort(time_vals, kind="stable")
        wake_sorted = flag_lunar_wake(
            scpot_vals[order], threshold=wake_scpot_threshold, pad=wake_pad
        )
        in_wake[order] = wake_sorted
    df["in_wake"] = in_wake

    # --- Drop un-runnable rows (NaN / invalid scalars) -----------------------
    # The classifier applies log10(n) and PCA, which cannot consume NaN or
    # n <= 0. Spectrum-channel NaNs are NOT a drop reason (prepare_ion_spectra
    # already filled them with 0). Dropped rows are removed, never relabelled.
    scalar_cols = [
        c for c in (("SCPot", "n") + _B_COMPONENTS_GSE) if c in df.columns
    ]
    finite_mask = np.ones(len(df), dtype=bool)
    for c in scalar_cols:
        finite_mask &= np.isfinite(df[c].to_numpy(dtype=float))
    if "n" in df.columns:
        finite_mask &= df["n"].to_numpy(dtype=float) > 0.0
    n_total = len(df)
    n_dropped = int((~finite_mask).sum())
    if verbose:
        print(
            f"artemis-cmmae: dropped {n_dropped}/{n_total} points with "
            "NaN/invalid scalars"
        )
    df = df.loc[finite_mask].copy()

    # --- Optional wake removal ----------------------------------------------
    if drop_wake:
        df = df.loc[~df["in_wake"].to_numpy(dtype=bool)].copy()

    return df


def predict_from_ds(
    ds,
    *,
    fgm_ds=None,
    state_ds=None,
    probe: Optional[str] = None,
    alpha: int = 1,
    classifier=None,
    return_features: bool = False,
    **build_kwargs,
) -> pd.DataFrame:
    """Build features from a ds and run the classifier end-to-end.

    Args:
        ds: The PEIF ESA dataset (xarray Dataset or dict-like).
        fgm_ds: Optional separate FGM dataset for the B-field merge.
        state_ds: Optional state/position dataset. When supplied, the GSE + GSM
            spacecraft position (``X_GSE``..``Z_GSM``, R_E) is merged in and
            included in the output (output-only context; never a model input).
        probe: Probe prefix; auto-detected if None.
        alpha: SOM map-size penalty (1, 5, or 10) used to construct a classifier
            when ``classifier`` is not supplied.
        classifier: A pre-loaded ``PlasmaClassifier`` to reuse (avoids re-loading
            the model weights). Its ``alpha`` takes precedence over the ``alpha``
            argument.
        return_features: If True, include the feature columns (C0..C30, n, SCPot,
            ``BX_GSE``/``BY_GSE``/``BZ_GSE`` plus any ``BX_GSM``/``BY_GSM``/
            ``BZ_GSM``) used for the prediction in the returned frame.
        **build_kwargs: Extra keyword arguments forwarded to
            :func:`build_features_from_ds` (e.g. ``b_source``, ``merge_tolerance``,
            ``drop_top_channel``, ``time_res``).

    Returns:
        A ``DataFrame`` indexed by a ``DatetimeIndex`` named 'time' with columns
        ``region_id`` (int), ``region_name`` (str), and ``in_wake`` (bool), plus
        the position columns ``X_GSE``..``Z_GSM`` whenever they are available. If
        ``return_features`` is True, the model input feature columns (C0..C30, n,
        SCPot, GSE B) and any GSM B are included as well.

    Notes:
        :func:`build_features_from_ds` DROPS un-runnable rows (NaN / invalid
        scalars) up front, so every surviving row is fed to the model. The MODEL
        is the ONLY source of ``region_id``, INCLUDING ``-1`` == "Unknown" -- a
        point that maps to a SOM node with no majority training label (the paper's
        ~4% Unknown). ``-1`` therefore means "the model ran and the node is
        unlabeled", never "we could not run the model". Un-runnable rows are
        dropped, never relabelled ``-1``.

    Raises:
        ValueError: If the required model input columns (C0..C30, n, SCPot,
            ``BX_GSE``/``BY_GSE``/``BZ_GSE``) are not all present (e.g. no B was
            supplied).
    """
    # Lazy import so this module (and features.py) stay importable without torch
    # being loaded until a prediction is actually requested.
    from .core import PlasmaClassifier

    features = build_features_from_ds(
        ds,
        probe=probe,
        fgm_ds=fgm_ds,
        state_ds=state_ds,
        **build_kwargs,
    )

    clf = classifier if classifier is not None else PlasmaClassifier(alpha=alpha)
    label_map = clf.LABEL_MAP

    ch_cols = [f"C{i}" for i in range(N_MODEL_CHANNELS)]
    tab_cols = ["n", "SCPot", *_B_COMPONENTS_GSE]
    missing = [c for c in ch_cols + tab_cols if c not in features.columns]
    if missing:
        raise ValueError(
            f"predict_from_ds needs model input columns {missing}; supply a "
            "fgm_ds (or BX_GSE/BY_GSE/BZ_GSE in the ds) so the GSE B-field is "
            "present."
        )

    # Every surviving row is runnable (build_features_from_ds already dropped
    # NaN / invalid-scalar rows). Run the model on all of them; region_id in
    # {-1, 0, 1, 2, 3} is emitted entirely by the model. Only CHANNEL_COLS +
    # TABULAR_COLS (GSE B) reach the model -- GSM B and position are ignored by
    # clf.predict.
    if len(features):
        predictions, label_map = clf.predict(features)
        region_id = np.asarray(predictions, dtype=int)
    else:
        region_id = np.zeros(0, dtype=int)
    region_name = [label_map.get(int(r), "Unknown") for r in region_id]

    out = pd.DataFrame(
        {"region_id": region_id, "region_name": region_name},
        index=features.index,
    )
    if "in_wake" in features.columns:
        out["in_wake"] = features["in_wake"].to_numpy(dtype=bool)

    # Position (output-only context) is ALWAYS included when available -- it does
    # not require return_features.
    pos_present = [c for c in _POS_COLS if c in features.columns]
    for c in pos_present:
        out[c] = features[c].to_numpy(dtype=float)

    if return_features:
        # Everything else the model / context used, minus what we already placed
        # (in_wake + position) so columns are not duplicated.
        already = set(["in_wake", *pos_present])
        feats = features.drop(columns=already, errors="ignore")
        out = pd.concat([out, feats], axis=1)
    return out


def predict_from_pyspedas(
    probe: str,
    trange,
    *,
    time_res: str = "peif",
    b_source: str = "fgs",
    filter_out_wake: bool = True,
    wake_scpot_threshold: float = 1.0,
    wake_pad: int = 1,
    drop_wake: bool = False,
    alpha: int = 1,
    classifier=None,
    return_features: bool = False,
    verbose: bool = True,
    no_update: bool = False,
    files=None,
    fgm_files=None,
    state_files=None,
    merge_tolerance: str = "60s",
) -> pd.DataFrame:
    """Load THEMIS/ARTEMIS ESA + FGM + state via pyspedas and run the classifier.

    Thin download-and-delegate wrapper: it loads the PEIF ESA product, the FGM
    field, AND the L1 state (spacecraft ephemeris) with
    :mod:`artemis_cmmae.loaders`, then hands the assembled datasets to the proven
    :func:`predict_from_ds` path (spectrum prep, lunar-wake flag, NaN-drop, and
    prediction). pyspedas / cdflib are imported lazily inside the loaders, so
    importing :mod:`artemis_cmmae` does not pull them in until a loader runs.

    The state product is ALWAYS loaded and merged, so the returned frame ALWAYS
    carries the GSE + GSM spacecraft position (``X_GSE``..``Z_GSM``, R_E) as
    output-only context, independent of ``return_features``.

    Offline / power-user path: pass ``files`` (ESA CDF paths), ``fgm_files`` (FGM
    CDF paths) and ``state_files`` (L1 state CDF paths) to read local CDFs
    directly and skip all network access.

    The download directory follows pyspedas's own config: set the
    ``SPEDAS_DATA_DIR`` environment variable (read at pyspedas import time)
    before first use to control where CDFs are cached.

    Args:
        probe: 'thb'/'thc' or 'b'/'c' (ARTEMIS), or any THEMIS letter a-e.
        trange: ``[start, end]`` time range (anything ``pandas.Timestamp`` parses).
            Used to download and to time-clip the loaded records. May be None when
            ``files`` are supplied and no clipping is wanted.
        time_res: ESA product; only 'peif' is supported downstream.
        b_source: FGM cadence: 'fgs' (default), 'fgl', or 'fgh'.
        filter_out_wake: Populate the ``in_wake`` column (default True).
        wake_scpot_threshold: Wake ``SCPot`` threshold in V (paper: 1.0).
        wake_pad: Adjacent steps flagged each side of a wake sample (paper: 1).
        drop_wake: If True, drop wake-flagged rows from the returned frame.
        alpha: SOM map-size penalty (1, 5, or 10) for a freshly-built classifier.
        classifier: A pre-loaded ``PlasmaClassifier`` to reuse.
        return_features: Include the model input feature columns in the output.
        verbose: Print the NaN/invalid-drop line.
        no_update: If True, use only local files; default False downloads missing files.
        files: Optional list of local ESA CDF paths (bypasses pyspedas download).
        fgm_files: Optional list of local FGM CDF paths (bypasses download).
        state_files: Optional list of local L1 state CDF paths (bypasses
            download) used for the always-on position merge.
        merge_tolerance: Nearest-time tolerance for the FGM->ESA merge.

    Returns:
        A ``DataFrame`` indexed by a ``DatetimeIndex`` named 'time' with
        ``region_id`` (in {-1, 0, 1, 2, 3}, all model-emitted), ``region_name``,
        ``in_wake``, and ALWAYS the position columns
        ``X_GSE``/``Y_GSE``/``Z_GSE``/``X_GSM``/``Y_GSM``/``Z_GSM`` (R_E). With
        ``return_features=True`` the model input columns (C0..C30, n, SCPot, GSE
        B) and the GSM B columns are included too.
    """
    from .loaders import load_esa, load_fgm, load_position, _normalize_probe

    _, var_prefix = _normalize_probe(probe)

    esa_ds = load_esa(
        trange=trange,
        probe=probe,
        files=files,
        time_res=time_res,
        no_update=no_update,
    )
    fgm_ds = load_fgm(
        trange=trange,
        probe=probe,
        files=fgm_files,
        b_source=b_source,
        no_update=no_update,
    )
    # State/position is ALWAYS loaded so the output always carries GSE + GSM
    # position (output-only context; never a model input).
    state_ds = load_position(
        trange=trange,
        probe=probe,
        files=state_files,
        no_update=no_update,
    )

    return predict_from_ds(
        esa_ds,
        fgm_ds=fgm_ds,
        state_ds=state_ds,
        probe=var_prefix,
        alpha=alpha,
        classifier=classifier,
        return_features=return_features,
        b_source=b_source,
        time_res=time_res,
        merge_tolerance=merge_tolerance,
        filter_out_wake=filter_out_wake,
        wake_scpot_threshold=wake_scpot_threshold,
        wake_pad=wake_pad,
        drop_wake=drop_wake,
        verbose=verbose,
    )
