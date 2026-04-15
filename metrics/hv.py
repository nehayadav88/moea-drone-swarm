"""Hypervolume indicator for multi-objective optimization."""

import numpy as np


def compute_hypervolume(pareto_front, reference_point):
    """Compute the hypervolume dominated by the Pareto front.

    The hypervolume indicator measures the volume of the objective space
    dominated by the Pareto front and bounded above by a reference point.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Array of shape (n_points, n_obj) containing the objective values
        of the Pareto front. All objectives are assumed to be minimized.
    reference_point : numpy.ndarray
        Array of shape (n_obj,) specifying the reference (upper bound) point.

    Returns
    -------
    float
        The hypervolume indicator value. Returns 0 for an empty front.
    """
    pareto_front = np.asarray(pareto_front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if pareto_front.ndim == 1 and pareto_front.size > 0:
        pareto_front = pareto_front.reshape(1, -1)

    if pareto_front.size == 0:
        return 0.0

    n_points, n_obj = pareto_front.shape

    # Filter out points that do not dominate the reference point
    mask = np.all(pareto_front < reference_point, axis=1)
    pareto_front = pareto_front[mask]

    if pareto_front.shape[0] == 0:
        return 0.0

    if n_obj == 1:
        return float(np.max(reference_point[0] - pareto_front[:, 0]))

    if n_obj == 2:
        return _hv_2d(pareto_front, reference_point)

    return _hv_recursive(pareto_front, reference_point)


def _hv_2d(pareto_front, reference_point):
    """Sweep-line hypervolume for 2-D objective space.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Array of shape (n_points, 2).
    reference_point : numpy.ndarray
        Array of shape (2,).

    Returns
    -------
    float
        Hypervolume value.
    """
    # Sort by first objective ascending
    sorted_indices = np.argsort(pareto_front[:, 0])
    front = pareto_front[sorted_indices]

    hv = 0.0
    prev_x = reference_point[0]

    # Sweep from right to left (largest x first) accumulating rectangles
    # We iterate in reverse order of x (descending first objective)
    for i in range(len(front) - 1, -1, -1):
        width = prev_x - front[i, 0]
        height = reference_point[1] - front[i, 1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = front[i, 0]

    return float(hv)


def _hv_recursive(pareto_front, reference_point):
    """Hypervolume computation for general dimensions.

    For small fronts (≤ 8 points) uses exact inclusion-exclusion.
    For larger fronts with 3+ objectives uses Monte Carlo estimation
    for computational tractability.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Array of shape (n_points, n_obj) with n_obj >= 2.
    reference_point : numpy.ndarray
        Array of shape (n_obj,).

    Returns
    -------
    float
        Hypervolume value (exact for small fronts, estimated for large).
    """
    n_points, n_obj = pareto_front.shape

    if n_points == 0:
        return 0.0

    if n_obj == 1:
        return float(reference_point[0] - np.min(pareto_front[:, 0]))

    if n_obj == 2:
        return _hv_2d(pareto_front, reference_point)

    if n_points == 1:
        return float(np.prod(reference_point - pareto_front[0]))

    # Use exact inclusion-exclusion for very small fronts
    if n_points <= 8:
        return _hv_inclusion_exclusion(pareto_front, reference_point)

    # For larger fronts with 3+ objectives, use Monte Carlo estimation
    return _hv_monte_carlo(pareto_front, reference_point)


def _hv_monte_carlo(pareto_front, reference_point, n_samples=100000):
    """Monte Carlo hypervolume estimation for higher dimensions.

    Samples random points in the bounding box [ideal, reference] and
    counts the fraction dominated by at least one Pareto front member.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Array of shape (n_points, n_obj).
    reference_point : numpy.ndarray
        Array of shape (n_obj,).
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    float
        Estimated hypervolume value.
    """
    n_points, n_obj = pareto_front.shape
    ideal = np.min(pareto_front, axis=0)

    # Total bounding-box volume
    box_vol = float(np.prod(reference_point - ideal))
    if box_vol <= 0:
        return 0.0

    rng = np.random.default_rng(0)
    samples = rng.uniform(ideal, reference_point, size=(n_samples, n_obj))

    # A sample is dominated if there exists at least one front point
    # that is <= the sample in every objective.
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pareto_front:
        dominated |= np.all(samples >= p, axis=1)

    fraction = np.mean(dominated)
    return box_vol * fraction


def _hv_inclusion_exclusion(pareto_front, reference_point):
    """Inclusion-exclusion based hypervolume for small dimensions.

    Computes the hypervolume using the inclusion-exclusion principle
    over the hyper-boxes defined by each point and the reference point.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Array of shape (n_points, n_obj).
    reference_point : numpy.ndarray
        Array of shape (n_obj,).

    Returns
    -------
    float
        Hypervolume value.
    """
    n_points = pareto_front.shape[0]

    if n_points == 0:
        return 0.0
    if n_points == 1:
        return float(np.prod(reference_point - pareto_front[0]))

    # Recursive inclusion-exclusion:
    # HV(S) = HV(S \ {p}) + vol(box(p, ref)) - HV_overlap
    # where the overlap is computed with respect to the intersection boxes.
    #
    # We use an iterative approach for efficiency.
    hv = 0.0
    for i in range(n_points):
        # Volume of box from point i to reference
        box_vol = float(np.prod(reference_point - pareto_front[i]))
        if box_vol <= 0:
            continue

        # Subtract overlaps with previously counted points using inclusion-exclusion
        overlap = _compute_overlap(pareto_front[:i], pareto_front[i], reference_point)
        hv += box_vol - overlap

    return hv


def _compute_overlap(previous_points, current_point, reference_point):
    """Compute the overlap volume between current point's box and previous points' boxes.

    Uses recursive inclusion-exclusion on the intersection boxes.

    Parameters
    ----------
    previous_points : numpy.ndarray
        Array of shape (n_prev, n_obj).
    current_point : numpy.ndarray
        Array of shape (n_obj,).
    reference_point : numpy.ndarray
        Array of shape (n_obj,).

    Returns
    -------
    float
        Total overlap volume.
    """
    if previous_points.shape[0] == 0:
        return 0.0

    # Intersection boxes: for each previous point, the intersection of its box
    # with the current point's box is defined by max(prev, current) to reference.
    intersections = np.maximum(previous_points, current_point)

    # Filter valid intersections (all dims must be less than reference)
    valid = np.all(intersections < reference_point, axis=1)
    intersections = intersections[valid]

    if intersections.shape[0] == 0:
        return 0.0

    # Recursively compute the hypervolume of the intersection boxes
    # using the same inclusion-exclusion approach
    return _hv_inclusion_exclusion(intersections, reference_point)


def _filter_dominated(points):
    """Remove dominated points from a set.

    A point p dominates q if p_i <= q_i for all i and p_j < q_j for some j.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n_points, n_obj).

    Returns
    -------
    numpy.ndarray
        Non-dominated subset.
    """
    if points.shape[0] <= 1:
        return points

    n = points.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(i + 1, n):
            if is_dominated[j]:
                continue
            if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                is_dominated[j] = True
            elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated[i] = True
                break

    return points[~is_dominated]
