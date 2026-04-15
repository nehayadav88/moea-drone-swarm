"""Visualization utilities for Pareto fronts, convergence, and paths."""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_pareto(fronts: Dict[str, Sequence[Sequence[float]]], out_path: str) -> None:
    plt.figure(figsize=(7, 5))
    for name, front in fronts.items():
        if not front:
            continue
        x = [f[0] for f in front]
        y = [f[2] for f in front]
        plt.scatter(x, y, s=24, label=name, alpha=0.8)
    plt.xlabel("Total Path Length")
    plt.ylabel("Total Energy")
    plt.title("Pareto Front Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence(histories: Dict[str, Sequence[dict]], out_path: str) -> None:
    plt.figure(figsize=(7, 5))
    for name, history in histories.items():
        xs = [h["generation"] for h in history]
        ys = [h.get("best_sum", 0.0) for h in history]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Generation")
    plt.ylabel("Best Aggregated Objective")
    plt.title("Convergence Traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_paths(paths: Sequence[Sequence[tuple]], obstacles: Sequence[object], out_path: str) -> None:
    plt.figure(figsize=(7, 7))
    for path in paths:
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        plt.plot(x, y, marker="o", linewidth=1.5)
    for obs in obstacles:
        if hasattr(obs, "radius") and hasattr(obs, "center"):
            circ = plt.Circle(obs.center, obs.radius, color="red", alpha=0.25)
            plt.gca().add_patch(circ)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Drone Path Visualization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
