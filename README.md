# MOEA Drone Swarm Path Planning Framework

A research-grade Python framework for **multi-objective optimization of heterogeneous drone swarm path planning**. Implements NSGA-II, MOEA/D, and SPEA2 algorithms from scratch with comprehensive performance metrics and visualization.

## Overview

This framework formulates drone swarm path planning as a multi-objective optimization problem (MOP), simultaneously minimizing:

1. **Total Path Length** — sum of Euclidean path distances for all drones
2. **Total Flight Time** — sum of (path length / drone speed) for each drone
3. **Total Energy Consumption** — realistic energy model including distance, hovering, turning, payload, and drag
4. **Safety Penalty** — proximity penalties for paths near obstacles

Subject to constraints:
- Energy feasibility (per-drone battery limits)
- Path feasibility (no obstacle collisions)
- Communication connectivity (drones must maintain network connectivity)

## Features

- **Heterogeneous drones** with varying battery, speed, sensing range, payload, and mass
- **Realistic energy model** with distance, hovering, turning, payload, and aerodynamic drag components
- **Three MOEA implementations from scratch**: NSGA-II, MOEA/D (Tchebycheff), SPEA2
- **Comprehensive metrics**: Hypervolume, IGD, GD, Spread, Spacing, Pure Diversity
- **Statistical analysis**: Wilcoxon rank-sum tests, improvement rates, comparison tables
- **Rich visualization**: Pareto fronts, convergence curves, drone path plots, box plots, heatmaps
- **Configurable experiments** via YAML configuration
- **Modular architecture** for easy extension

## Project Structure

```
moea-drone-swarm/
├── core/
│   ├── energy_model.py      # Realistic energy consumption model
│   ├── problem_definition.py # Multi-objective problem formulation
│   └── constraints.py        # Constraint checking utilities
├── algorithms/
│   ├── base.py               # Abstract MOEA base class
│   ├── nsga2.py              # NSGA-II implementation
│   ├── moead.py              # MOEA/D (Tchebycheff) implementation
│   └── spea2.py              # SPEA2 implementation
├── simulation/
│   ├── environment.py        # 2D environment with obstacles
│   ├── drone.py              # Heterogeneous drone model
│   └── path_planner.py       # Path utilities and feasibility
├── metrics/
│   ├── hv.py                 # Hypervolume indicator
│   ├── igd.py                # IGD, IGD+, GD
│   ├── diversity.py          # Spread, Spacing, Pure Diversity
│   └── statistics.py         # Statistical tests and analysis
├── visualization/
│   └── plots.py              # All plotting functions
├── experiments/
│   ├── configs.yaml          # Experiment configuration
│   └── run_experiments.py    # Main experiment runner
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/nehayadav88/moea-drone-swarm.git
cd moea-drone-swarm

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- PyYAML >= 6.0

## Quick Start

### Run a Quick Test

```bash
python -m experiments.run_experiments --quick
```

This runs a reduced experiment (2 runs, 30 generations) to verify everything works.

### Run Full Experiments

```bash
python -m experiments.run_experiments --config experiments/configs.yaml
```

For publication-quality results, set `n_runs: 30` in `experiments/configs.yaml`.

### Custom Configuration

Edit `experiments/configs.yaml` to modify:

```yaml
environment:
  width: 100.0          # Environment width (meters)
  height: 100.0         # Environment height (meters)
  n_static_obstacles: 15
  n_dynamic_obstacles: 5

swarm:
  n_drones: 5           # Number of drones in swarm

problem:
  num_waypoints_per_drone: 5  # Intermediate waypoints per drone
  safety_distance: 3.0        # Safety margin around obstacles

algorithms:
  nsga2:
    pop_size: 100
    n_gen: 150
  moead:
    pop_size: 105
    n_gen: 150
  spea2:
    pop_size: 100
    n_gen: 150

experiment:
  n_runs: 30            # Number of independent runs
  base_seed: 42         # Base random seed
  output_dir: "results" # Output directory
```

## Expected Outputs

After running experiments, the `results/` directory contains:

```
results/
├── plots/
│   ├── pareto_front.png      # Pareto front comparison
│   ├── convergence.png       # Hypervolume convergence curves
│   ├── boxplot_hv.png        # Hypervolume distribution
│   ├── boxplot_mean_energy.png
│   ├── boxplot_mean_path_length.png
│   └── drone_paths.png       # Best drone paths visualization
└── results_summary.yaml      # Numerical results
```

Console output includes:
- Per-run metrics (HV, solution count, runtime)
- Statistical comparison tables (mean, std, median)
- Pairwise Wilcoxon rank-sum test results

## Mathematical Formulation

### Decision Variables

For each drone *d* with *W* intermediate waypoints:
- x_d = [x₁, y₁, x₂, y₂, ..., x_W, y_W]

Full path: [start_d] → [x₁,y₁] → ... → [x_W,y_W] → [target_d]

### Objective Functions

1. **Path Length**: f₁ = Σ_d Σ_i ||p_{i+1} - p_i||₂
2. **Flight Time**: f₂ = Σ_d (path_length_d / speed_d)
3. **Energy**: f₃ = Σ_d E_total(d) where E_total = E_dist + E_hover + E_turn + E_payload + E_drag
4. **Safety**: f₄ = Σ_d Σ_obstacles max(0, safety_dist - dist_to_obstacle)

### Energy Model

- E_dist = k_d · mass · distance · speed_factor
- E_hover = k_h · mass · hover_time
- E_turn = k_t · Σ|Δθ|
- E_payload = k_p · payload · distance
- E_drag = k_drag · speed² · distance

### Constraints

- g₁: E_total(d) ≤ battery_capacity(d) ∀d
- g₂: No path-obstacle intersections
- g₃: Communication graph connectivity at each waypoint step

## Extending the Framework

### Adding a New Algorithm

1. Create `algorithms/my_algo.py`
2. Inherit from `MOEABase`
3. Implement the `run()` method
4. Add to `algorithm_map` in `run_experiments.py`

### Adding New Objectives

1. Modify `DroneSwarmMOP.evaluate()` in `core/problem_definition.py`
2. Update `n_obj` property
3. Adjust reference point in `configs.yaml`

### Scaling to More Drones

Set `n_drones` in config. The framework supports 20+ drones but computation time scales with swarm size. Consider reducing `n_gen` or `pop_size` for larger swarms.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{moea_drone_swarm,
  title={MOEA Drone Swarm Path Planning Framework},
  year={2024},
  url={https://github.com/nehayadav88/moea-drone-swarm}
}
```

## License

This project is provided for research and educational purposes.