"""
User-facing utilities for the ARTEMIS plasma classifier.

This module is the home for small, dependency-light helpers that operate on the
prediction-output frames produced by :mod:`artemis_cmmae.pipeline`. It has no
runtime dependency on torch / xarray / pyspedas -- only ``pandas`` -- so it is
safe to import anywhere.

Currently it hosts :func:`region_timeline`, the consecutive-run ("N-hour")
persistence filter.
"""

import pandas as pd


def region_timeline(labels, min_duration="1h"):
    """Keep only samples in contiguous same-region runs lasting >= ``min_duration``.

    This is a dwell-time / persistence filter, NOT a downsampling. A sample is
    kept only if it belongs to a run of CONSECUTIVE samples that share the same
    ``region_id`` and whose span (last minus first sample time) is at least
    ``min_duration``. Short-lived or flickering labels are dropped, so the result
    is a STRICT SUBSET of the input rows -- labels are never changed or merged.
    This mirrors ``filter_continuous_intervals`` from the reference
    ``consecutive_regions.py``: the "1-hour" classification is "the regions that
    were classified the same, continuously, for at least an hour".

    A run is broken ONLY by a change in ``region_id``. Data gaps do not break a
    run and no maximum gap is imposed: a same-region run spans its full time
    extent regardless of how sparsely it is sampled, so it counts as
    >= ``min_duration`` as long as its first and last samples are that far apart.

    Args:
        labels: A prediction-output :class:`~pandas.DataFrame` with a
            ``DatetimeIndex`` and a ``region_id`` column (as returned by
            :func:`~artemis_cmmae.pipeline.predict_from_pyspedas` /
            :func:`~artemis_cmmae.pipeline.predict_from_ds`).
        min_duration: Minimum run span to keep, as a ``pandas`` timedelta string
            or :class:`~pandas.Timedelta` (default ``'1h'``).

    Returns:
        A subset of ``labels`` (same columns, ``DatetimeIndex``, time-sorted)
        containing only the samples that lie in runs lasting >= ``min_duration``.
        May be empty if no run is long enough.

    Raises:
        ValueError: If ``labels`` lacks a ``region_id`` column.
        TypeError: If ``labels`` is not indexed by a ``DatetimeIndex``.
    """
    if not isinstance(labels, pd.DataFrame) or "region_id" not in labels.columns:
        raise ValueError(
            "labels must be a prediction-output DataFrame containing a "
            "'region_id' column (e.g. from predict_from_pyspedas)."
        )
    if not isinstance(labels.index, pd.DatetimeIndex):
        raise TypeError(
            "labels must be indexed by a pandas DatetimeIndex (the prediction "
            "output is)."
        )

    min_dur = pd.Timedelta(min_duration)
    df = labels.sort_index()
    if len(df) == 0:
        return df.copy()

    rid = df["region_id"].astype(int)
    # A new run starts wherever the region changes; data gaps do NOT break runs.
    change = rid.ne(rid.shift())
    run_id = change.cumsum()

    times = df.index.to_series()
    grp = times.groupby(run_id)
    span = grp.transform("max") - grp.transform("min")
    keep = (span >= min_dur).to_numpy()
    return df.loc[keep]
