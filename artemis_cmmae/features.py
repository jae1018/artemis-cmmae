"""
Ion energy-flux spectrum feature preparation for the ARTEMIS plasma classifier.

This module contains the ion omnidirectional flux feature-prep recipe used to turn 
raw THEMIS/ARTEMIS PEIF ESA L2 energy-flux spectra into the C0..C30 channel columns 
consumed by ``PlasmaClassifier.predict``.

Recipe summary (validated to float32 storage precision against the bundled
``sample.csv``):

* Fixed target grid = the REAL bin-type-0 instrument energy table (32 values,
  ascending), hard-coded here as ``REF_ENERGY_GRID_ASC``.
* Per record: keep channels where energy AND eflux are finite; if the record's
  energy grid equals the bin-0 grid (magnetosphere mode) it is an identity copy
  of the ascending-sorted eflux; otherwise interpolate ``log10(eflux + 1)``
  LINEARLY IN ENERGY onto the bin-0 targets that fall within the source span,
  then back-transform ``max(0, 10**y - 1)``. Targets outside the source span are
  filled with 0.0.
* Assemble ascending C0..C31, then DROP C31 (the highest-energy channel) to yield
  C0..C30 (31 channels, LINEAR eflux, ascending energy).

The raw CDF energy axis is DESCENDING, so a sort is required.
"""

import numpy as np
import pandas as pd

# Number of energy channels the model consumes (C0..C30).
N_MODEL_CHANNELS = 31

# ---------------------------------------------------------------------------
# The REAL bin-type-0 energy grid (32 values), lifted verbatim from the parity
# proof (parity_prep.py / PARITY_REPORT.md). Extracted from the instrument
# energy table (bin type 0, magnetosphere mode, 32 channels) and cross-checked
# against a full-32-channel magnetosphere-mode raw CDF record
# (thb_peif_en_eflux_yaxis): max abs diff 2.8e-5 eV. Stored here DESCENDING
# (raw CDF order) then sorted ascending for the model target grid.
# ---------------------------------------------------------------------------
_BIN0_DESC = np.array([
    24590.8828, 20770.4629, 15778.8877, 11986.9863, 9106.19238, 6917.73389,
    5255.37646, 3992.28491, 3032.77051, 2304.41040, 1750.16602, 1329.51074,
    1009.79779, 767.761536, 582.763245, 442.419769, 336.599274, 255.545258,
    194.379501, 147.473251, 112.199753, 85.5569992, 64.9182510, 49.1577530,
    37.5250015, 28.5190010, 21.7645016, 16.1357498, 12.3832502, 9.75650024,
    7.12975025, 5.62875032,
], dtype=float)

#: Fixed model target energy grid (eV), ascending: C0 = 5.629 eV .. C31 = 24591 eV.
#: The model consumes C0..C30; C31 (highest energy) is dropped by the recipe.
REF_ENERGY_GRID_ASC = np.sort(_BIN0_DESC)

# Valid energy domain spanned by the bin-0 grid.
_BIN0_DOMAIN = (float(REF_ENERGY_GRID_ASC.min()), float(REF_ENERGY_GRID_ASC.max()))


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------
def epoch_to_datetime64(epoch_arr) -> np.ndarray:
    """Convert Unix epoch seconds to ``datetime64[ns]``.

    Matches ``artemis_multimodal_ssl/eval/pseudo_labels.py::_epoch_to_datetime64``:
    ``arr * 1e9 -> int64 -> view datetime64[ns]``. If ``epoch_arr`` is already a
    ``datetime64`` array it is returned unchanged.

    Args:
        epoch_arr: Array-like of Unix epoch seconds (float64), or a datetime64
            array (passed through).

    Returns:
        ``numpy.ndarray`` of dtype ``datetime64[ns]``.
    """
    arr = np.asarray(epoch_arr)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")
    arr = arr.astype(np.float64)
    ns = (arr * 1e9).astype(np.int64)
    return ns.view("datetime64[ns]")


# ---------------------------------------------------------------------------
# Lunar-wake flag
# ---------------------------------------------------------------------------
def flag_lunar_wake(scpot, *, threshold: float = 1.0, pad: int = 1) -> np.ndarray:
    """Flag lunar-wake time steps from spacecraft potential.

    Implements the paper's lunar-wake criterion: a record is flagged as wake when
    its spacecraft potential ``SCPot <= threshold`` (1 V in the paper), *plus* the
    ``pad`` adjacent time steps on each side. This mirrors the wording in
    Figure "Lunar Wake Filtering" of the ARTEMIS clustering paper
    ("``SCPot <= 1`` V plus adjacent time steps") and the reference implementation
    ``filter_lunar_wake.py`` (which flags ``idx - 1`` and ``idx + 1`` around every
    wake sample, i.e. ``pad = 1``).

    The dilation is purely INDEX-BASED on the *given* order of ``scpot``, so
    callers must pass a TIME-SORTED array for the adjacency to reflect true
    temporal neighbours. Non-finite ``SCPot`` entries are never flagged (returned
    False), even when adjacent to a genuine wake sample -- they carry no usable
    potential and are dropped downstream regardless.

    Args:
        scpot: 1-D array-like of spacecraft potential (V), in the order on which
            adjacency should be computed (typically time-sorted).
        threshold: Wake potential threshold; ``SCPot <= threshold`` is wake.
        pad: Number of adjacent time steps to flag on EACH side of every wake
            sample. The paper/code value is 1 (one step before, one after).

    Returns:
        Boolean ``numpy.ndarray`` (same length as ``scpot``); True == in wake.
    """
    sp = np.asarray(scpot, dtype=float).ravel()
    n = sp.shape[0]
    finite = np.isfinite(sp)
    base = finite & (sp <= threshold)
    wake = base.copy()
    if pad and int(pad) > 0 and base.any():
        idx = np.flatnonzero(base)
        for k in range(1, int(pad) + 1):
            left = idx - k
            wake[left[left >= 0]] = True
            right = idx + k
            wake[right[right < n]] = True
    # Non-finite SCPot is never wake (even if a neighbour dilated onto it).
    wake[~finite] = False
    return wake


# ---------------------------------------------------------------------------
# Core spectrum-prep recipe (Candidate A)
# ---------------------------------------------------------------------------
def prepare_ion_spectra(eflux, energy, *, drop_top_channel: bool = True) -> np.ndarray:
    """
    Interpolate raw ion energy-flux spectra onto the model's channel grid.

    Args:
        eflux: ``(T, 32)`` or ``(32,)`` linear ion energy flux. The raw THEMIS
            PEIF CDF stores energy DESCENDING with NaN in inactive channels
            (e.g. solar-wind mode has ~16 finite channels).
        energy: ``(T, 32)`` or ``(32,)`` per-record energy axis (eV), same shape
            and ordering as ``eflux``.
        drop_top_channel: If True (default) drop C31 (highest energy) to return
            31 channels (C0..C30). If False, return all 32 ascending channels.

    Returns:
        ``(T, 31)`` (or ``(T, 32)`` if ``drop_top_channel`` is False) LINEAR,
        ASCENDING channel array, ready to become the C-columns of a feature
        frame. NaNs are replaced with 0.0.
    """
    eflux = np.asarray(eflux, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if eflux.ndim == 1:
        eflux = eflux[None, :]
    if energy.ndim == 1:
        energy = energy[None, :]
    if eflux.shape != energy.shape:
        raise ValueError(
            f"eflux shape {eflux.shape} != energy shape {energy.shape}"
        )

    T = eflux.shape[0]
    out32 = np.full((T, 32), np.nan, dtype=float)  # ascending C0..C31

    tgt = REF_ENERGY_GRID_ASC
    tgt_in_bin0 = (
        (tgt >= _BIN0_DOMAIN[0]) & (tgt <= _BIN0_DOMAIN[1]) & np.isfinite(tgt)
    )

    for i in range(T):
        e = energy[i]
        f = eflux[i]
        e_finite = np.isfinite(e)

        # Reference (magnetosphere / bin-0) mode: all 32 energies finite and the
        # sorted grid matches the bin-0 grid -> identity copy of ascending eflux.
        is_ref = (e_finite.sum() == 32) and np.allclose(
            np.sort(e), REF_ENERGY_GRID_ASC, rtol=1e-4, atol=1e-2
        )
        if is_ref:
            order = np.argsort(e)  # ascending energy
            out32[i, :] = f[order]  # keep NaN where flux is NaN
            continue

        # Non-reference (e.g. solar-wind) mode: interpolate log10(f + 1) linearly
        # in energy onto the bin-0 targets that lie within the source span.
        src_mask = e_finite & np.isfinite(f)
        if not np.any(src_mask):
            continue
        xs = e[src_mask]
        fs = f[src_mask]
        o = np.argsort(xs)
        xp = xs[o]
        Fp = fs[o]
        xmin, xmax = xp[0], xp[-1]
        tgt_in_src = (tgt >= xmin) & (tgt <= xmax)
        tmask = tgt_in_src & tgt_in_bin0
        if not np.any(tmask):
            continue
        # numpy.interp is linear interpolation; targets are within [xmin, xmax]
        # so no extrapolation happens (matches interp1d on the interior).
        ylog = np.interp(tgt[tmask], xp, np.log10(Fp + 1.0))
        ylin = np.maximum(0.0, (10.0 ** ylog) - 1.0)
        out32[i, tmask] = ylin

    # Fill unfilled targets (out of span / bin0) with 0.
    out32 = np.nan_to_num(out32, nan=0.0)

    if drop_top_channel:
        return out32[:, :N_MODEL_CHANNELS]  # drop C31 (highest energy)
    return out32