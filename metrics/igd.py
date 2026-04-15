"""Inverted Generational Distance and related metrics."""

import numpy as np


def _normalize_fronts(pareto_front, reference_front):
    """Normalize both fronts to [0, 1] based on reference front bounds.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).
    reference_front : numpy.ndarray
        Reference Pareto front of shape (n_ref, n_obj).

    Returns
    -------
    tuple of numpy.ndarray
        Normalized (pareto_front, reference_front).
    """
    combined = np.vstack([pareto_front, reference_front])
    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # avoid division by zero for constant objectives
    pf_norm = (pareto_front - min_vals) / ranges
    rf_norm = (reference_front - min_vals) / ranges
    return pf_norm, rf_norm


def compute_igd(pareto_front, reference_front, normalize=False):
    """Compute the Inverted Generational Distance (IGD).

    IGD measures the average distance from each reference point to
    its nearest point in the obtained Pareto front.

    IGD = (1 / |R|) * sum_{r in R} min_{p in P} d(r, p)

    where d is the Euclidean distance.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).
    reference_front : numpy.ndarray
        Reference (true) Pareto front of shape (n_ref, n_obj).
    normalize : bool, optional
        If True, normalize objectives to [0, 1] before computing. Default False.

    Returns
    -------
    float
        The IGD value. Lower is better. Returns inf for an empty obtained front,
        and 0 for an empty reference front.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)
    if reference_front.ndim == 1 and reference_front.size > 0:
        reference_front = reference_front.reshape(1, -1)

    if reference_front.size == 0:
        return 0.0
    if pareto_front.size == 0:
        return float("inf")

    if normalize:
        pareto_front, reference_front = _normalize_fronts(pareto_front, reference_front)

    # For each reference point, find minimum Euclidean distance to obtained front
    # Use broadcasting: (n_ref, 1, n_obj) - (1, n_points, n_obj) -> (n_ref, n_points, n_obj)
    diff = reference_front[:, np.newaxis, :] - pareto_front[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_ref, n_points)
    min_distances = np.min(distances, axis=1)  # (n_ref,)

    return float(np.mean(min_distances))


def compute_igd_plus(pareto_front, reference_front, normalize=False):
    """Compute the IGD+ indicator.

    IGD+ is a weakly Pareto-compliant variant of IGD that uses a modified
    distance function which only considers worsening in each objective.

    d+(r, p) = sqrt(sum_i max(p_i - r_i, 0)^2)

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).
    reference_front : numpy.ndarray
        Reference (true) Pareto front of shape (n_ref, n_obj).
    normalize : bool, optional
        If True, normalize objectives to [0, 1] before computing. Default False.

    Returns
    -------
    float
        The IGD+ value. Lower is better. Returns inf for an empty obtained front,
        and 0 for an empty reference front.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)
    if reference_front.ndim == 1 and reference_front.size > 0:
        reference_front = reference_front.reshape(1, -1)

    if reference_front.size == 0:
        return 0.0
    if pareto_front.size == 0:
        return float("inf")

    if normalize:
        pareto_front, reference_front = _normalize_fronts(pareto_front, reference_front)

    # d+(r, p) = sqrt(sum_i max(p_i - r_i, 0)^2)
    diff = pareto_front[np.newaxis, :, :] - reference_front[:, np.newaxis, :]  # (n_ref, n_points, n_obj)
    diff_plus = np.maximum(diff, 0.0)
    distances = np.sqrt(np.sum(diff_plus ** 2, axis=2))  # (n_ref, n_points)
    min_distances = np.min(distances, axis=1)  # (n_ref,)

    return float(np.mean(min_distances))


def compute_gd(pareto_front, reference_front, normalize=False):
    """Compute the Generational Distance (GD).

    GD measures the average distance from each obtained point to
    its nearest point in the reference front.

    GD = (1 / |P|) * sum_{p in P} min_{r in R} d(p, r)

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).
    reference_front : numpy.ndarray
        Reference (true) Pareto front of shape (n_ref, n_obj).
    normalize : bool, optional
        If True, normalize objectives to [0, 1] before computing. Default False.

    Returns
    -------
    float
        The GD value. Lower is better. Returns 0 for an empty obtained front,
        and inf for an empty reference front.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)
    if reference_front.ndim == 1 and reference_front.size > 0:
        reference_front = reference_front.reshape(1, -1)

    if pareto_front.size == 0:
        return 0.0
    if reference_front.size == 0:
        return float("inf")

    if normalize:
        pareto_front, reference_front = _normalize_fronts(pareto_front, reference_front)

    # For each obtained point, find minimum Euclidean distance to reference front
    diff = pareto_front[:, np.newaxis, :] - reference_front[np.newaxis, :, :]  # (n_points, n_ref, n_obj)
    distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_points, n_ref)
    min_distances = np.min(distances, axis=1)  # (n_points,)

    return float(np.mean(min_distances))
