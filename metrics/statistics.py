"""Statistical analysis utilities for multi-objective optimization experiments."""

import numpy as np
from scipy import stats


def wilcoxon_rank_sum_test(data1, data2, alpha=0.05):
    """Perform the Wilcoxon rank-sum (Mann-Whitney U) test.

    Tests whether two independent samples come from the same distribution.

    Parameters
    ----------
    data1 : array-like
        First sample of metric values.
    data2 : array-like
        Second sample of metric values.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'statistic': float, the U statistic.
        - 'p_value': float, the two-sided p-value.
        - 'significant': bool, True if p_value < alpha.
        - 'effect_size': float, rank-biserial correlation r = 1 - 2U/(n1*n2).
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)

    n1, n2 = len(data1), len(data2)

    if n1 == 0 or n2 == 0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
        }

    result = stats.mannwhitneyu(data1, data2, alternative="two-sided")
    u_stat = result.statistic
    p_value = result.pvalue

    # Rank-biserial correlation as effect size
    effect_size = 1.0 - (2.0 * u_stat) / (n1 * n2)

    return {
        "statistic": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "effect_size": float(effect_size),
    }


def compute_improvement_rate(baseline_values, method_values):
    """Compute the percentage improvement of a method over a baseline.

    Improvement = 100 * (baseline - method) / |baseline| for each pair
    of values. A positive improvement means the method is better (lower)
    than the baseline.

    Parameters
    ----------
    baseline_values : array-like
        Metric values from the baseline algorithm.
    method_values : array-like
        Metric values from the method being evaluated.

    Returns
    -------
    float
        Mean percentage improvement. Positive means the method is better.
        Returns 0.0 if inputs are empty or baseline values are all zero.
    """
    baseline_values = np.asarray(baseline_values, dtype=float)
    method_values = np.asarray(method_values, dtype=float)

    if baseline_values.size == 0 or method_values.size == 0:
        return 0.0

    # Align lengths to the shorter array
    n = min(len(baseline_values), len(method_values))
    baseline_values = baseline_values[:n]
    method_values = method_values[:n]

    abs_baseline = np.abs(baseline_values)
    # Avoid division by zero
    valid = abs_baseline > 0
    if not np.any(valid):
        return 0.0

    improvements = np.zeros(n)
    improvements[valid] = (
        100.0 * (baseline_values[valid] - method_values[valid]) / abs_baseline[valid]
    )

    return float(np.mean(improvements))


def compute_statistics_table(results_dict):
    """Compute summary statistics for multiple algorithms.

    Parameters
    ----------
    results_dict : dict
        Mapping of algorithm_name (str) to a list/array of metric values
        from multiple independent runs.

    Returns
    -------
    dict
        Mapping of algorithm_name to a dict with keys:
        'mean', 'std', 'median', 'min', 'max', 'iqr'.
    """
    stats_table = {}

    for algo_name, values in results_dict.items():
        values = np.asarray(values, dtype=float)

        if values.size == 0:
            stats_table[algo_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "iqr": float("nan"),
            }
            continue

        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))

        stats_table[algo_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "iqr": q3 - q1,
        }

    return stats_table


def pairwise_comparison(results_dict, alpha=0.05):
    """Perform pairwise Wilcoxon rank-sum tests between all algorithms.

    Parameters
    ----------
    results_dict : dict
        Mapping of algorithm_name (str) to a list/array of metric values.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'algorithms': list of algorithm names.
        - 'p_values': 2D numpy array of p-values where entry (i, j) is the
          p-value of the test between algorithms i and j.
        - 'symbols': 2D list of strings where entry (i, j) is '+' if
          algorithm i is significantly better (lower) than j, '-' if
          significantly worse, or '=' if no significant difference.
    """
    algo_names = list(results_dict.keys())
    n = len(algo_names)

    p_values = np.ones((n, n))
    symbols = [["=" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            data_i = np.asarray(results_dict[algo_names[i]], dtype=float)
            data_j = np.asarray(results_dict[algo_names[j]], dtype=float)

            if data_i.size == 0 or data_j.size == 0:
                p_values[i, j] = float("nan")
                p_values[j, i] = float("nan")
                continue

            result = wilcoxon_rank_sum_test(data_i, data_j, alpha=alpha)
            p_values[i, j] = result["p_value"]
            p_values[j, i] = result["p_value"]

            if result["significant"]:
                median_i = float(np.median(data_i))
                median_j = float(np.median(data_j))
                if median_i < median_j:
                    symbols[i][j] = "+"
                    symbols[j][i] = "-"
                else:
                    symbols[i][j] = "-"
                    symbols[j][i] = "+"

    return {
        "algorithms": algo_names,
        "p_values": p_values,
        "symbols": symbols,
    }


def format_results_table(stats_dict):
    """Format statistics into a printable aligned table.

    Parameters
    ----------
    stats_dict : dict
        Output of ``compute_statistics_table``: mapping of algorithm_name
        to a dict with 'mean', 'std', 'median', 'min', 'max', 'iqr'.

    Returns
    -------
    str
        Formatted table string.
    """
    if not stats_dict:
        return ""

    headers = ["Algorithm", "Mean", "Std", "Median", "Min", "Max", "IQR"]
    metric_keys = ["mean", "std", "median", "min", "max", "iqr"]

    # Build rows
    rows = []
    for algo_name, metrics in stats_dict.items():
        row = [algo_name]
        for key in metric_keys:
            val = metrics[key]
            if np.isnan(val):
                row.append("N/A")
            else:
                row.append(f"{val:.6f}")
        rows.append(row)

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "  ".join("-" * col_widths[i] for i in range(len(headers)))

    # Format rows
    row_lines = []
    for row in rows:
        line = "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        row_lines.append(line)

    return "\n".join([header_line, separator] + row_lines)
