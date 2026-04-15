"""Diversity metrics for multi-objective optimization."""

import numpy as np


def compute_spread(pareto_front, reference_front=None):
    """Compute the spread (Delta) metric.

    The Delta metric measures the extent of spread achieved by the
    obtained Pareto front. A value of 0 indicates ideal uniformity.

    Delta = (d_f + d_l + sum(|d_i - d_mean|)) / (d_f + d_l + (N-1) * d_mean)

    where d_f and d_l are distances from the extreme solutions of the
    reference front to the boundary solutions of the obtained front,
    and d_i are consecutive distances between adjacent solutions.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).
    reference_front : numpy.ndarray, optional
        Reference Pareto front for determining extreme points.
        If None, extreme points are taken from pareto_front itself.

    Returns
    -------
    float
        The Delta spread value. Lower is better (0 = perfect uniformity).
        Returns 0 for fronts with fewer than 2 points.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)

    n_points = pareto_front.shape[0]
    if n_points < 2:
        return 0.0

    n_obj = pareto_front.shape[1]

    # Determine extreme points from reference front or obtained front
    if reference_front is not None:
        reference_front = np.asarray(reference_front, dtype=float)
        if reference_front.ndim == 1 and reference_front.size > 0:
            reference_front = reference_front.reshape(1, -1)
        extremes = _get_extreme_points(reference_front)
    else:
        extremes = _get_extreme_points(pareto_front)

    # Sort the Pareto front using crowding-distance ordering:
    # For 2-D, sort by first objective; for higher dimensions, use
    # lexicographic sort by first objective.
    sorted_indices = np.lexsort(pareto_front[:, ::-1].T)
    sorted_front = pareto_front[sorted_indices]

    # Compute consecutive distances between adjacent solutions
    consecutive_dists = np.sqrt(
        np.sum((sorted_front[1:] - sorted_front[:-1]) ** 2, axis=1)
    )

    if len(consecutive_dists) == 0:
        return 0.0

    d_mean = np.mean(consecutive_dists)

    if d_mean == 0.0:
        return 0.0

    # Compute d_f and d_l: distances from extreme points to nearest boundary
    # solutions of the obtained front
    d_f = np.sqrt(np.sum((sorted_front[0] - extremes[0]) ** 2))
    d_l = np.sqrt(np.sum((sorted_front[-1] - extremes[-1]) ** 2))

    # Delta metric
    numerator = d_f + d_l + np.sum(np.abs(consecutive_dists - d_mean))
    denominator = d_f + d_l + (n_points - 1) * d_mean

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def _get_extreme_points(front):
    """Get extreme points of a front (min for each objective).

    Parameters
    ----------
    front : numpy.ndarray
        Array of shape (n_points, n_obj).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_obj, n_obj) where row i is the point with
        minimum value in objective i. For 2-obj case, returns the two
        boundary points.
    """
    n_obj = front.shape[1]
    extremes = np.empty((n_obj, n_obj))
    for i in range(n_obj):
        idx = np.argmin(front[:, i])
        extremes[i] = front[idx]
    return extremes


def compute_pure_diversity(pareto_front):
    """Compute the Pure Diversity (PD) metric.

    Pure Diversity measures the minimum pairwise Euclidean distance
    among all solutions in the normalized objective space. A larger
    value indicates better spread.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).

    Returns
    -------
    float
        The Pure Diversity value. Higher is better.
        Returns 0 for fronts with fewer than 2 points.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)

    n_points = pareto_front.shape[0]
    if n_points < 2:
        return 0.0

    # Normalize objectives to [0, 1]
    min_vals = np.min(pareto_front, axis=0)
    max_vals = np.max(pareto_front, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    normalized = (pareto_front - min_vals) / ranges

    # Compute pairwise distances
    # diff[i, j] = normalized[i] - normalized[j]
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    pairwise_dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(pairwise_dists, np.inf)

    # PD = minimum pairwise distance
    return float(np.min(pairwise_dists))


def compute_spacing(pareto_front):
    """Compute the Spacing metric.

    Spacing measures the uniformity of the distribution of solutions.
    A value of 0 indicates perfectly uniform spacing.

    SP = sqrt((1 / (N-1)) * sum((d_i - d_mean)^2))

    where d_i is the distance from point i to its nearest neighbor.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Obtained Pareto front of shape (n_points, n_obj).

    Returns
    -------
    float
        The Spacing value. Lower is better (0 = perfect uniformity).
        Returns 0 for fronts with fewer than 2 points.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)

    n_points = pareto_front.shape[0]
    if n_points < 2:
        return 0.0

    # Compute pairwise distances
    diff = pareto_front[:, np.newaxis, :] - pareto_front[np.newaxis, :, :]
    pairwise_dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Set diagonal to infinity
    np.fill_diagonal(pairwise_dists, np.inf)

    # Nearest neighbor distance for each point
    nearest_dists = np.min(pairwise_dists, axis=1)

    d_mean = np.mean(nearest_dists)

    # Spacing = standard deviation of nearest-neighbor distances (with N-1)
    spacing = np.sqrt(np.sum((nearest_dists - d_mean) ** 2) / (n_points - 1))

    return float(spacing)
