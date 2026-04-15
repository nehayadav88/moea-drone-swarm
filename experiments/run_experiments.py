"""Run reproducible multi-run experiments for heterogeneous drone swarm MOP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from algorithms.moead import MOEAD, MOEADConfig
from algorithms.nsga2 import NSGA2, NSGA2Config
from algorithms.spea2 import SPEA2, SPEA2Config
from core.energy_model import EnergyModel, EnergyModelConfig
from core.problem_definition import ProblemConfig, SwarmPathPlanningProblem
from metrics.diversity import pure_diversity, spread_delta
from metrics.hv import hypervolume
from metrics.igd import igd
from metrics.statistics import convergence_trace, improvement_rate, wilcoxon_rank_sum
from simulation.drone import Drone
from simulation.environment import CircleObstacle, Environment2D
from visualization.plots import plot_convergence, plot_pareto

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("Please install pyyaml to run experiments: pip install pyyaml") from exc


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_problem(cfg: dict) -> SwarmPathPlanningProblem:
    env_cfg = cfg["environment"]
    x_min, y_min, x_max, y_max = env_cfg["bounds"]
    static_circles = [CircleObstacle(tuple(o["center"]), o["radius"]) for o in env_cfg.get("static_circles", [])]
    dynamic_circles = [
        CircleObstacle(tuple(o["center"]), o["radius"], tuple(o.get("velocity", [0.0, 0.0])))
        for o in env_cfg.get("dynamic_circles", [])
    ]
    env = Environment2D(x_min, y_min, x_max, y_max, static_circles=static_circles, dynamic_circles=dynamic_circles)

    drones = []
    for i, d in enumerate(cfg["drones"]):
        drones.append(
            Drone(
                drone_id=i,
                start=tuple(d["start"]),
                goal=tuple(d["goal"]),
                battery_capacity=d["battery_capacity"],
                speed=d["speed"],
                sensing_range=d["sensing_range"],
                payload=d["payload"],
                mass=d.get("mass", 2.0),
                comm_range=d.get("comm_range", 20.0),
            )
        )

    problem_cfg = ProblemConfig(**cfg.get("problem", {}))
    energy_model = EnergyModel(EnergyModelConfig())
    return SwarmPathPlanningProblem(drones=drones, environment=env, energy_model=energy_model, config=problem_cfg)


def non_dominated(front: Sequence[Sequence[float]]) -> List[List[float]]:
    nd = []
    for i, p in enumerate(front):
        dominated = False
        for j, q in enumerate(front):
            if i == j:
                continue
            if all(q[k] <= p[k] for k in range(len(p))) and any(q[k] < p[k] for k in range(len(p))):
                dominated = True
                break
        if not dominated:
            nd.append(list(p))
    return nd


def run() -> None:
    root = Path(__file__).resolve().parent
    cfg = load_config(str(root / "configs.yaml"))
    exp = cfg["experiment"]
    output_dir = Path(exp["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = exp["runs"]
    seed = exp["seed"]

    alg_constructors = {
        "nsga2": lambda s: NSGA2(NSGA2Config(seed=s, **cfg["algorithms"]["nsga2"])),
        "moead": lambda s: MOEAD(MOEADConfig(seed=s, **cfg["algorithms"]["moead"])),
        "spea2": lambda s: SPEA2(SPEA2Config(seed=s, **cfg["algorithms"]["spea2"])),
    }

    all_results: Dict[str, List[dict]] = {name: [] for name in alg_constructors}
    run_fronts: Dict[str, List[List[List[float]]]] = {name: [] for name in alg_constructors}
    all_fronts: Dict[str, List[List[float]]] = {name: [] for name in alg_constructors}
    best_front_hv: Dict[str, float] = {name: float("-inf") for name in alg_constructors}
    histories = {}
    problem = build_problem(cfg)

    for run_id in range(runs):
        ref_point = [500.0, 80.0, 150000.0, 20.0]

        for name, ctor in alg_constructors.items():
            algo = ctor(seed + run_id)
            result = algo.run(problem)
            front = result.pareto_objectives

            hv = hypervolume(front, ref_point, samples=5000, seed=seed + run_id)
            div = pure_diversity(front)
            delta = spread_delta(front)
            all_results[name].append({"hv": hv, "pd": div, "delta": delta})
            run_fronts[name].append([list(p) for p in front])
            if hv > best_front_hv[name]:
                best_front_hv[name] = hv
                all_fronts[name] = [list(p) for p in front]
            histories[name] = result.history

    reference_front = non_dominated([p for fronts in run_fronts.values() for front in fronts for p in front])
    for name in all_results:
        for i in range(len(all_results[name])):
            all_results[name][i]["igd"] = igd(run_fronts[name][i], reference_front)

    summary = {}
    for name, records in all_results.items():
        summary[name] = {
            "hv_mean": mean([r["hv"] for r in records]),
            "pd_mean": mean([r["pd"] for r in records]),
            "delta_mean": mean([r["delta"] for r in records]),
            "igd_mean": mean([r["igd"] for r in records]),
        }

    baseline_name = "moead"
    for name in summary:
        if name == baseline_name:
            continue
        summary[name]["hv_improvement_over_baseline_pct"] = improvement_rate(
            summary[name]["hv_mean"], summary[baseline_name]["hv_mean"], maximize=True
        )
        stat = wilcoxon_rank_sum([r["hv"] for r in all_results[name]], [r["hv"] for r in all_results[baseline_name]])
        summary[name]["wilcoxon_p"] = stat["p_value"]

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "convergence.json", "w", encoding="utf-8") as f:
        json.dump({k: convergence_trace(v) for k, v in histories.items()}, f, indent=2)

    plot_pareto(all_fronts, str(output_dir / "pareto.png"))
    plot_convergence(histories, str(output_dir / "convergence.png"))

    print(f"Saved results in {output_dir}")


if __name__ == "__main__":
    run()
