"""Visualization functions for MOEA drone-swarm experiments.

All plotting uses the non-interactive ``Agg`` backend so the module works in
headless environments (CI, remote servers, containers).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401 — import needed for 3D projection

# ──────────────────────────────────────────────────────────────
# Shared colour palette – 10 distinct colours, cycled if needed
# ──────────────────────────────────────────────────────────────
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]


def _get_color(index: int) -> str:
    """Return a colour from the palette, cycling if *index* exceeds length."""
    return COLORS[index % len(COLORS)]


def _get_linestyle(index: int) -> str:
    return LINE_STYLES[index % len(LINE_STYLES)]


def _save_and_close(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to *save_path* (if given) and close it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 1. Pareto-front visualisation
# ──────────────────────────────────────────────────────────────


def plot_pareto_front(
    objectives_dict: Dict[str, np.ndarray],
    obj_names: Optional[List[str]] = None,
    title: str = "Pareto Front",
    save_path: Optional[str] = None,
) -> None:
    """Plot Pareto-front objective values for one or more algorithms.

    Args:
        objectives_dict: Mapping *algorithm_name* → array of shape
            ``(n_points, n_objectives)``.
        obj_names: Optional list of objective names used for axis labels.
        title: Figure title.
        save_path: If provided the figure is written to this path.
    """
    if not objectives_dict:
        return

    # Determine number of objectives from the first non-empty entry.
    n_obj: int = 0
    for arr in objectives_dict.values():
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] > 0:
            n_obj = arr.shape[1]
            break
    if n_obj == 0:
        return

    if obj_names is None:
        obj_names = [f"Objective {i + 1}" for i in range(n_obj)]

    # --- 2-objective scatter ---
    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, (name, data) in enumerate(objectives_dict.items()):
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[0] == 0:
                continue
            ax.scatter(
                data[:, 0],
                data[:, 1],
                label=name,
                color=_get_color(idx),
                alpha=0.7,
                edgecolors="k",
                linewidths=0.3,
                s=40,
            )
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save_and_close(fig, save_path)

    # --- 3-objective 3-D scatter ---
    elif n_obj == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for idx, (name, data) in enumerate(objectives_dict.items()):
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[0] == 0:
                continue
            ax.scatter(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                label=name,
                color=_get_color(idx),
                alpha=0.7,
                s=40,
            )
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_zlabel(obj_names[2])
        ax.set_title(title)
        ax.legend()
        _save_and_close(fig, save_path)

    # --- 4+ objectives → parallel coordinates ---
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_ticks = list(range(n_obj))
        for idx, (name, data) in enumerate(objectives_dict.items()):
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[0] == 0:
                continue
            # Normalise each objective to [0, 1] for comparability.
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normed = (data - mins) / ranges
            color = _get_color(idx)
            for row in normed:
                ax.plot(x_ticks, row, color=color, alpha=0.25, linewidth=0.8)
            # Draw one invisible line for the legend entry.
            ax.plot([], [], color=color, label=name, linewidth=2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(obj_names[: n_obj])
        ax.set_ylabel("Normalised value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 2. Convergence plot
# ──────────────────────────────────────────────────────────────


def plot_convergence(
    history_dict: Dict[str, List[float]],
    metric_name: str = "Hypervolume",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Line plot of a metric's value over generations for each algorithm.

    Args:
        history_dict: Mapping *algorithm_name* → list of metric values (one
            per generation).
        metric_name: Label for the y-axis.
        title: Optional figure title; defaults to ``'{metric_name} Convergence'``.
        save_path: If provided the figure is written to this path.
    """
    if not history_dict:
        return

    if title is None:
        title = f"{metric_name} Convergence"

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (name, values) in enumerate(history_dict.items()):
        values = list(values)
        if not values:
            continue
        generations = list(range(1, len(values) + 1))
        ax.plot(
            generations,
            values,
            label=name,
            color=_get_color(idx),
            linestyle=_get_linestyle(idx),
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 3. Drone-path visualisation
# ──────────────────────────────────────────────────────────────


def plot_drone_paths(
    environment,
    drones,
    paths_dict: Dict[int, np.ndarray],
    title: str = "Drone Paths",
    save_path: Optional[str] = None,
) -> None:
    """Plot drone flight paths inside an environment with obstacles.

    Automatically chooses 2-D or 3-D visualisation based on the
    dimensionality of the first path in *paths_dict*.

    Args:
        environment: An :class:`Environment` or :class:`Environment3D`.
        drones: Iterable of :class:`Drone` objects (used for IDs).
        paths_dict: Mapping *drone_id* → ``(n, 2)`` or ``(n, 3)`` array
            of waypoints describing the flight path.
        title: Figure title.
        save_path: If provided the figure is written to this path.
    """
    if not paths_dict:
        return

    # Auto-detect dimensionality.
    first_path = np.asarray(next(iter(paths_dict.values())))
    if first_path.ndim == 2 and first_path.shape[1] >= 3:
        _plot_drone_paths_3d(environment, drones, paths_dict, title, save_path)
    else:
        _plot_drone_paths_2d(environment, drones, paths_dict, title, save_path)


def _plot_drone_paths_2d(
    environment,
    drones,
    paths_dict: Dict[int, np.ndarray],
    title: str = "Drone Paths",
    save_path: Optional[str] = None,
) -> None:
    """2-D drone path visualisation (original behaviour)."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles as filled grey circles.
    if environment is not None:
        for obs in environment.get_obstacle_positions():
            cx, cy = obs[0], obs[1]
            r = obs[2]
            circle = Circle(
                (cx, cy), r, color="gray", alpha=0.5, zorder=1
            )
            ax.add_patch(circle)

    for idx, (drone_id, path) in enumerate(paths_dict.items()):
        path = np.asarray(path)
        if path.ndim != 2 or path.shape[0] == 0:
            continue
        color = _get_color(idx)

        ax.plot(
            path[:, 0], path[:, 1],
            color=color, linewidth=1.5, alpha=0.8, zorder=2,
            label=f"Drone {drone_id}",
        )
        ax.plot(path[0, 0], path[0, 1], "o", color="green", markersize=8, zorder=3)
        ax.plot(path[-1, 0], path[-1, 1], "X", color="red", markersize=10, zorder=3)
        ax.annotate(
            str(drone_id), xy=(path[0, 0], path[0, 1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold", color=color, zorder=4,
        )

    if environment is not None:
        ax.set_xlim(0, environment.width)
        ax.set_ylim(0, environment.height)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    _save_and_close(fig, save_path)


def _plot_drone_paths_3d(
    environment,
    drones,
    paths_dict: Dict[int, np.ndarray],
    title: str = "Drone Paths (3D)",
    save_path: Optional[str] = None,
) -> None:
    """3-D drone path visualisation with spherical obstacles."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw obstacles as wireframe spheres (subset for performance).
    if environment is not None:
        obstacles = environment.get_obstacle_positions()
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)
        for obs in obstacles[:30]:  # limit to 30 for readability
            cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
            xs = cx + r * np.outer(np.cos(u), np.sin(v))
            ys = cy + r * np.outer(np.sin(u), np.sin(v))
            zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.3)

    for idx, (drone_id, path) in enumerate(paths_dict.items()):
        path = np.asarray(path)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 3:
            continue
        color = _get_color(idx)

        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color=color, linewidth=1.5, alpha=0.8,
            label=f"Drone {drone_id}",
        )
        ax.scatter(*path[0, :3], color="green", s=50, marker="o", zorder=5)
        ax.scatter(*path[-1, :3], color="red", s=60, marker="X", zorder=5)

    if environment is not None:
        ax.set_xlim(0, environment.width)
        ax.set_ylim(0, environment.height)
        ax.set_zlim(0, environment.depth)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 4. Metrics box-plot comparison
# ──────────────────────────────────────────────────────────────


def plot_metrics_comparison(
    stats_dict: Dict[str, List[float]],
    metric_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Box plot comparing a metric across multiple algorithms.

    Args:
        stats_dict: Mapping *algorithm_name* → list of metric values across
            independent runs.
        metric_name: Name of the metric (used for y-label).
        title: Figure title; defaults to ``'{metric_name} Comparison'``.
        save_path: If provided the figure is written to this path.
    """
    if not stats_dict:
        return

    if title is None:
        title = f"{metric_name} Comparison"

    names = list(stats_dict.keys())
    data = [list(stats_dict[n]) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=False)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_get_color(idx))
        patch.set_alpha(0.6)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 5. Heatmap
# ──────────────────────────────────────────────────────────────


def plot_heatmap(
    data_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Heatmap",
    save_path: Optional[str] = None,
) -> None:
    """Annotated heatmap with a colour bar.

    Args:
        data_matrix: 2-D array of values (rows × cols).
        row_labels: Labels for the rows.
        col_labels: Labels for the columns.
        title: Figure title.
        save_path: If provided the figure is written to this path.
    """
    data_matrix = np.asarray(data_matrix, dtype=float)
    if data_matrix.ndim != 2 or data_matrix.size == 0:
        return

    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 1.2), max(4, len(row_labels) * 0.8))
    )
    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Annotate each cell with its numeric value.
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            val = data_matrix[i, j]
            text_color = "white" if val > (data_matrix.max() + data_matrix.min()) / 2 else "black"
            ax.text(
                j,
                i,
                f"{val:.3g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax.set_title(title)
    _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 6. Constraint-violation bar chart
# ──────────────────────────────────────────────────────────────


def plot_constraint_violations(
    violations_dict: Dict[str, List[float]],
    title: str = "Constraint Violations",
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart of constraint violations per algorithm and run.

    Args:
        violations_dict: Mapping *algorithm_name* → list of total violation
            values (one per independent run).
        title: Figure title.
        save_path: If provided the figure is written to this path.
    """
    if not violations_dict:
        return

    names = list(violations_dict.keys())
    all_values = [list(violations_dict[n]) for n in names]
    max_runs = max(len(v) for v in all_values) if all_values else 0
    if max_runs == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, max_runs * len(names) * 0.4), 5))
    bar_width = 0.8 / len(names)
    x_base = np.arange(max_runs)

    for idx, (name, vals) in enumerate(zip(names, all_values)):
        # Pad with zeros if some algorithms have fewer runs.
        padded = vals + [0.0] * (max_runs - len(vals))
        offsets = x_base + idx * bar_width
        ax.bar(
            offsets,
            padded,
            width=bar_width,
            label=name,
            color=_get_color(idx),
            alpha=0.8,
            edgecolor="k",
            linewidth=0.3,
        )

    ax.set_xlabel("Run")
    ax.set_ylabel("Total Violation")
    ax.set_title(title)
    ax.set_xticks(x_base + bar_width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"Run {i + 1}" for i in range(max_runs)])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_and_close(fig, save_path)


# ──────────────────────────────────────────────────────────────
# 7. Full experiment report
# ──────────────────────────────────────────────────────────────


def create_experiment_report(
    results: dict,
    output_dir: str,
) -> str:
    """Generate all experiment plots and an HTML summary page.

    Args:
        results: Dictionary with optional keys:

            * ``'pareto_fronts'`` – dict suitable for :func:`plot_pareto_front`
            * ``'convergence'``   – dict suitable for :func:`plot_convergence`
            * ``'paths'``         – dict suitable for :func:`plot_drone_paths`
            * ``'stats'``         – dict mapping *metric_name* → stats_dict
            * ``'environment'``   – :class:`Environment` instance
            * ``'drones'``        – list of :class:`Drone` instances
        output_dir: Directory where plots and the HTML file are written.

    Returns:
        Path to the generated ``report.html``.
    """
    os.makedirs(output_dir, exist_ok=True)
    images: List[Tuple[str, str]] = []  # (filename, caption)

    # --- Pareto front ---
    pareto = results.get("pareto_fronts")
    if pareto:
        fname = "pareto_front.png"
        plot_pareto_front(
            pareto,
            title="Pareto Front",
            save_path=os.path.join(output_dir, fname),
        )
        images.append((fname, "Pareto Front"))

    # --- Convergence ---
    convergence = results.get("convergence")
    if convergence:
        fname = "convergence.png"
        plot_convergence(
            convergence,
            title="Convergence",
            save_path=os.path.join(output_dir, fname),
        )
        images.append((fname, "Convergence"))

    # --- Drone paths ---
    paths = results.get("paths")
    env = results.get("environment")
    drones = results.get("drones", [])
    if paths and env:
        fname = "drone_paths.png"
        plot_drone_paths(
            env,
            drones,
            paths,
            title="Drone Paths",
            save_path=os.path.join(output_dir, fname),
        )
        images.append((fname, "Drone Paths"))

    # --- Metrics comparison (one plot per metric) ---
    stats = results.get("stats")
    if stats and isinstance(stats, dict):
        for metric_name, stats_dict in stats.items():
            safe_name = metric_name.lower().replace(" ", "_")
            fname = f"metrics_{safe_name}.png"
            plot_metrics_comparison(
                stats_dict,
                metric_name=metric_name,
                save_path=os.path.join(output_dir, fname),
            )
            images.append((fname, f"{metric_name} Comparison"))

    # --- Build HTML report ---
    html_path = os.path.join(output_dir, "report.html")
    img_tags = "\n".join(
        f'    <div style="margin-bottom:24px;">'
        f"<h2>{caption}</h2>"
        f'<img src="{fname}" style="max-width:100%;" />'
        f"</div>"
        for fname, caption in images
    )
    html = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        "<title>Experiment Report</title></head>\n"
        '<body style="font-family:sans-serif;max-width:960px;margin:auto;padding:20px;">\n'
        "  <h1>MOEA Drone-Swarm Experiment Report</h1>\n"
        f"{img_tags}\n"
        "  <p><em>Report generated by visualization.plots</em></p>\n"
        "</body></html>\n"
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return html_path
