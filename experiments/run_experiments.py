#!/usr/bin/env python3
"""
MOEA Drone Swarm Path Planning - Experiment Runner
====================================================
Runs multi-objective optimization experiments for heterogeneous drone swarm
path planning, comparing NSGA-II, MOEA/D, and SPEA2.

Usage:
    python -m experiments.run_experiments [--config experiments/configs.yaml] [--quick]
"""

import sys
import os
import time
import argparse
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import Environment
from simulation.drone import Drone
from core.problem_definition import DroneSwarmMOP
from algorithms.nsga2 import NSGA2
from algorithms.moead import MOEAD
from algorithms.spea2 import SPEA2
from metrics.hv import compute_hypervolume
from metrics.igd import compute_igd, compute_gd
from metrics.diversity import compute_spread, compute_spacing, compute_pure_diversity
from metrics.statistics import (
    compute_statistics_table, pairwise_comparison,
    format_results_table, compute_improvement_rate
)
from visualization.plots import (
    plot_pareto_front, plot_convergence, plot_drone_paths,
    plot_metrics_comparison, create_experiment_report
)


def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _drone_to_dict(drone):
    """Convert a Drone object to the dict format expected by DroneSwarmMOP."""
    return {
        "start": tuple(drone.start_position),
        "target": tuple(drone.target_position),
        "mass": drone.mass,
        "payload": drone.payload,
        "speed": drone.max_speed,
        "battery_capacity": drone.battery_capacity,
        "comm_range": drone.comm_range,
        "drone_radius": drone.drone_radius,
    }


def setup_problem(config, seed=None):
    """Create environment, drones, and problem instance."""
    env_cfg = config['environment']
    env = Environment(width=env_cfg['width'], height=env_cfg['height'])
    env.generate_random_obstacles(
        n_static=env_cfg['n_static_obstacles'],
        n_dynamic=env_cfg['n_dynamic_obstacles'],
        min_radius=env_cfg['obstacle_min_radius'],
        max_radius=env_cfg['obstacle_max_radius'],
        seed=seed
    )

    swarm_cfg = config['swarm']
    drones = Drone.create_heterogeneous_swarm(
        n_drones=swarm_cfg['n_drones'],
        env_width=env_cfg['width'],
        env_height=env_cfg['height'],
        seed=seed
    )

    prob_cfg = config['problem']
    drone_dicts = [_drone_to_dict(d) for d in drones]
    task_positions = [(d.target_position[0], d.target_position[1]) for d in drones]

    problem = DroneSwarmMOP(
        environment=env,
        drones=drone_dicts,
        task_positions=task_positions,
        config={
            "num_waypoints_per_drone": prob_cfg['num_waypoints_per_drone'],
            "safety_distance": prob_cfg.get('safety_distance', 3.0),
        },
    )

    return env, drones, problem


def run_single_experiment(algorithm_class, problem, algo_config, seed):
    """Run a single algorithm execution and return results."""
    algo = algorithm_class(problem=problem, seed=seed, **algo_config)

    start_time = time.time()
    pf_solutions, pf_objectives = algo.run()
    elapsed = time.time() - start_time

    return {
        'solutions': pf_solutions,
        'objectives': pf_objectives,
        'history': algo.history,
        'time': elapsed
    }


def compute_all_metrics(pf_objectives, reference_point, reference_front=None):
    """Compute all performance metrics for a Pareto front."""
    metrics = {}

    # Hypervolume
    try:
        metrics['hv'] = compute_hypervolume(pf_objectives, reference_point)
    except Exception:
        metrics['hv'] = 0.0

    # IGD (if reference front available)
    if reference_front is not None and len(reference_front) > 0:
        metrics['igd'] = compute_igd(pf_objectives, reference_front)
        metrics['gd'] = compute_gd(pf_objectives, reference_front)

    # Diversity metrics
    if len(pf_objectives) > 1:
        metrics['spread'] = compute_spread(pf_objectives)
        metrics['spacing'] = compute_spacing(pf_objectives)
        metrics['pure_diversity'] = compute_pure_diversity(pf_objectives)
    else:
        metrics['spread'] = 0.0
        metrics['spacing'] = 0.0
        metrics['pure_diversity'] = 0.0

    # Objective statistics
    metrics['mean_path_length'] = np.mean(pf_objectives[:, 0])
    metrics['mean_flight_time'] = np.mean(pf_objectives[:, 1])
    metrics['mean_energy'] = np.mean(pf_objectives[:, 2])
    metrics['mean_safety'] = np.mean(pf_objectives[:, 3])
    metrics['n_solutions'] = len(pf_objectives)

    return metrics


def run_experiments(config_path, quick=False):
    """Main experiment loop."""
    config = load_config(config_path)

    if quick:
        config['experiment']['n_runs'] = 2
        for algo in config['algorithms']:
            config['algorithms'][algo]['n_gen'] = 30
            config['algorithms'][algo]['pop_size'] = min(
                config['algorithms'][algo].get('pop_size', 50), 50
            )

    exp_cfg = config['experiment']
    n_runs = exp_cfg['n_runs']
    base_seed = exp_cfg['base_seed']
    output_dir = exp_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    reference_point = np.array(config['metrics']['reference_point'])

    algorithm_map = {
        'nsga2': NSGA2,
        'moead': MOEAD,
        'spea2': SPEA2,
    }

    # Results storage
    all_results = {}
    all_metrics = {}

    for algo_name in config['algorithms']:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()}")
        print(f"{'='*60}")

        algo_config = config['algorithms'][algo_name].copy()
        algo_class = algorithm_map[algo_name]

        # Remove n_gen and pop_size from config to pass separately
        # (they're direct constructor args)

        all_results[algo_name] = []
        all_metrics[algo_name] = {
            'hv': [], 'spread': [], 'spacing': [], 'pure_diversity': [],
            'mean_path_length': [], 'mean_flight_time': [],
            'mean_energy': [], 'mean_safety': [],
            'time': [], 'n_solutions': []
        }

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")

            env, drones, problem = setup_problem(config, seed=seed)

            result = run_single_experiment(
                algo_class, problem, algo_config, seed=seed + 1000
            )
            all_results[algo_name].append(result)

            # Compute metrics
            metrics = compute_all_metrics(
                result['objectives'], reference_point
            )
            metrics['time'] = result['time']

            for key in all_metrics[algo_name]:
                if key in metrics:
                    all_metrics[algo_name][key].append(metrics[key])

            print(f"  HV={metrics['hv']:.4f}, "
                  f"Solutions={metrics['n_solutions']}, "
                  f"Time={result['time']:.2f}s")

    # ---- Statistical Analysis ----
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    for metric_name in ['hv', 'spread', 'mean_energy', 'mean_path_length']:
        print(f"\n--- {metric_name.upper()} ---")
        metric_data = {
            name: vals[metric_name]
            for name, vals in all_metrics.items()
            if len(vals[metric_name]) > 0
        }

        if len(metric_data) > 0:
            stats = compute_statistics_table(metric_data)
            print(format_results_table(stats))

            if len(metric_data) > 1 and all(len(v) >= 2 for v in metric_data.values()):
                comparison = pairwise_comparison(metric_data)
                algo_names = comparison['algorithms']
                p_vals = comparison['p_values']
                syms = comparison['symbols']
                print("\nPairwise Wilcoxon test p-values:")
                for i in range(len(algo_names)):
                    for j in range(i + 1, len(algo_names)):
                        print(f"  {algo_names[i]} vs {algo_names[j]}: "
                              f"p={p_vals[i, j]:.4f} [{syms[i][j]}]")

    # ---- Visualization ----
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    viz_dir = os.path.join(output_dir, 'plots')
    os.makedirs(viz_dir, exist_ok=True)

    # Pareto fronts from last run
    pf_dict = {}
    for algo_name in all_results:
        last_result = all_results[algo_name][-1]
        pf_dict[algo_name.upper()] = last_result['objectives']

    plot_pareto_front(
        pf_dict,
        obj_names=['Path Length', 'Flight Time', 'Energy', 'Safety'],
        title='Pareto Front Comparison',
        save_path=os.path.join(viz_dir, 'pareto_front.png')
    )

    # Convergence curves (best mean objective values from history)
    conv_dict = {}
    for algo_name in all_results:
        history = all_results[algo_name][-1]['history']
        if 'objectives' in history and len(history['objectives']) > 0:
            # Track mean of first objective (path length) per generation
            mean_obj1 = []
            for gen_obj in history['objectives']:
                if len(gen_obj) > 0:
                    mean_obj1.append(float(np.mean(gen_obj[:, 0])))
            if mean_obj1:
                conv_dict[algo_name.upper()] = mean_obj1

    if conv_dict:
        plot_convergence(
            conv_dict,
            metric_name='Mean Path Length (Front)',
            title='Convergence Comparison',
            save_path=os.path.join(viz_dir, 'convergence.png')
        )

    # Metric box plots
    for metric_name in ['hv', 'mean_energy', 'mean_path_length']:
        metric_data = {
            name.upper(): vals[metric_name]
            for name, vals in all_metrics.items()
        }
        plot_metrics_comparison(
            metric_data,
            metric_name=metric_name.replace('_', ' ').title(),
            save_path=os.path.join(viz_dir, f'boxplot_{metric_name}.png')
        )

    # Drone paths from best solution of last run (using first algorithm)
    first_algo = list(all_results.keys())[0]
    last_result = all_results[first_algo][-1]
    env, drones, problem = setup_problem(config, seed=base_seed + n_runs - 1)

    # Decode best solution (first in Pareto front)
    if len(last_result['solutions']) > 0:
        best_sol = last_result['solutions'][0]
        drone_paths = problem.decode_solution(best_sol)
        paths_dict = {i: path for i, path in enumerate(drone_paths)}

        plot_drone_paths(
            env, drones, paths_dict,
            title=f'Drone Paths ({first_algo.upper()})',
            save_path=os.path.join(viz_dir, 'drone_paths.png')
        )

    # ---- Save Results ----
    results_summary = {
        'config': config,
        'metrics': {
            algo: {k: [float(v) for v in vals]
                   for k, vals in metrics.items()}
            for algo, metrics in all_metrics.items()
        }
    }

    summary_path = os.path.join(output_dir, 'results_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)

    print(f"\nResults saved to {output_dir}/")
    print(f"Plots saved to {viz_dir}/")
    print("Experiment complete!")

    return all_results, all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run MOEA drone swarm experiments'
    )
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(os.path.dirname(__file__), 'configs.yaml'),
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick experiment with reduced parameters for testing'
    )
    args = parser.parse_args()

    run_experiments(args.config, quick=args.quick)
