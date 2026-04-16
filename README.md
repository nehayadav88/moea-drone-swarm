# MOEA Drone Swarm Path Planning Framework

A research-grade Python framework for **multi-objective optimization of heterogeneous drone swarm path planning**. Implements **11 algorithms inspired by PlatEMO** from scratch, with both 2-D and 3-D simulation environments, comprehensive performance metrics, and visualization.

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
- **2-D and 3-D environments** with static and dynamic obstacles
- **8 predefined 3-D test scenarios** (urban canyon, dense forest, corridor, etc.)
- **Realistic energy model** with distance, hovering, turning, payload, and aerodynamic drag components
- **11 MOEA implementations from scratch** (inspired by PlatEMO):
  - NSGA-II, NSGA-III, MOEA/D, SPEA2, MOPSO, RVEA, IBEA, GDE3, SMS-EMOA, AGE-MOEA, HypE
- **Comprehensive metrics**: Hypervolume, IGD, GD, Spread, Spacing, Pure Diversity
- **Statistical analysis**: Wilcoxon rank-sum tests, improvement rates, comparison tables
- **Rich visualization**: Pareto fronts, convergence curves, 2-D/3-D drone path plots, box plots, heatmaps
- **Configurable experiments** via YAML configuration
- **Modular architecture** for easy extension

## Project Structure

```
moea-drone-swarm/
├── core/
│   ├── energy_model.py        # Realistic energy consumption model (2D + 3D)
│   ├── problem_definition.py  # DroneSwarmMOP (2D) + DroneSwarmMOP3D (3D)
│   └── constraints.py         # Collision, energy, communication constraints
├── algorithms/
│   ├── base.py                # Abstract MOEA base class (SBX, polynomial mutation, NDS)
│   ├── nsga2.py               # NSGA-II (Deb et al., 2002)
│   ├── nsga3.py               # NSGA-III (Deb & Jain, 2014)
│   ├── moead.py               # MOEA/D Tchebycheff (Zhang & Li, 2007)
│   ├── spea2.py               # SPEA2 (Zitzler et al., 2001)
│   ├── mopso.py               # MOPSO (Coello Coello et al., 2004)
│   ├── rvea.py                # RVEA (Cheng et al., 2016)
│   ├── ibea.py                # IBEA (Zitzler & Künzli, 2004)
│   ├── gde3.py                # GDE3 (Kukkonen & Lampinen, 2005)
│   ├── sms_emoa.py            # SMS-EMOA (Beume et al., 2007)
│   ├── age_moea.py            # AGE-MOEA (Panichella, 2019)
│   └── hype.py                # HypE (Bader & Zitzler, 2011)
├── simulation/
│   ├── environment.py         # 2D environment with circular obstacles
│   ├── environment3d.py       # 3D environment with spherical obstacles
│   ├── drone.py               # Heterogeneous drone model
│   ├── path_planner.py        # Path utilities and feasibility
│   └── scenarios.py           # 8 predefined 3D test scenarios
├── metrics/
│   ├── hv.py                  # Hypervolume indicator
│   ├── igd.py                 # IGD, IGD+, GD
│   ├── diversity.py           # Spread, Spacing, Pure Diversity
│   └── statistics.py          # Wilcoxon rank-sum, comparison tables
├── visualization/
│   └── plots.py               # Pareto fronts, convergence, 2D/3D paths, box plots
├── experiments/
│   ├── configs.yaml           # Experiment configuration (all 11 algorithms)
│   └── run_experiments.py     # Main experiment runner
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

### Run a Quick Test (2-D)

```bash
python -m experiments.run_experiments --quick
```

This runs a reduced experiment (2 runs, 30 generations) with all 11 algorithms.

### Run a 3-D Scenario

```bash
python -m experiments.run_experiments --scenario urban_canyon --quick
```

Available 3-D scenarios: `urban_canyon`, `dense_forest`, `open_field`, `corridor`, `multi_layer`, `search_and_rescue`, `high_density`, `custom`.

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

# Comment out algorithms you don't want to run
algorithms:
  nsga2:
    pop_size: 100
    n_gen: 150
  nsga3:
    pop_size: 100
    n_gen: 150
    n_partitions: 12
  moead:
    pop_size: 105
    n_gen: 150
  spea2:
    pop_size: 100
    n_gen: 150
  mopso:
    pop_size: 100
    n_gen: 150
  rvea:
    pop_size: 100
    n_gen: 150
  ibea:
    pop_size: 100
    n_gen: 150
  gde3:
    pop_size: 100
    n_gen: 150
  sms_emoa:
    pop_size: 100
    n_gen: 150
  age_moea:
    pop_size: 100
    n_gen: 150
  hype:
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

## Algorithms (Inspired by PlatEMO)

All 11 algorithms are implemented from scratch following the `MOEABase` interface:

| Algorithm | Reference | Key Feature |
|-----------|-----------|-------------|
| NSGA-II   | Deb et al., 2002 | Crowding distance selection |
| NSGA-III  | Deb & Jain, 2014 | Das-Dennis reference points |
| MOEA/D    | Zhang & Li, 2007 | Tchebycheff decomposition |
| SPEA2     | Zitzler et al., 2001 | Strength-based fitness + archive |
| MOPSO     | Coello Coello et al., 2004 | Particle swarm + grid archive |
| RVEA      | Cheng et al., 2016 | Angle-penalized distance |
| IBEA      | Zitzler & Künzli, 2004 | Epsilon indicator fitness |
| GDE3      | Kukkonen & Lampinen, 2005 | Differential evolution |
| SMS-EMOA  | Beume et al., 2007 | Hypervolume contribution |
| AGE-MOEA  | Panichella, 2019 | Adaptive geometry estimation |
| HypE      | Bader & Zitzler, 2011 | Monte Carlo HV estimation |

## 3-D Test Scenarios

Use `--scenario` to run with predefined 3-D environments:

| Scenario | Description | Drones | Obstacles |
|----------|-------------|--------|-----------|
| `urban_canyon` | Tall building-like columns | 8 | ~56 |
| `dense_forest` | Many small tree-like obstacles | 5 | 30-50 |
| `open_field` | Large area, few scattered obstacles | 10 | ~8 |
| `corridor` | Narrow passage with wall obstacles | 4 | ~30 |
| `multi_layer` | Obstacles at ground/mid/high altitudes | 8 | ~30 |
| `search_and_rescue` | Large area, scattered obstacles | 12 | ~20 |
| `high_density` | Stress test with many drones | 20 | ~40 |
| `custom` | User-configurable via kwargs | configurable | configurable |

```python
from simulation.scenarios import create_scenario, list_scenarios

# List all available scenarios
print(list_scenarios())

# Create a specific scenario
env, drones = create_scenario("urban_canyon", seed=42)
print(f"Environment: {env.width}x{env.height}x{env.depth}")
print(f"Drones: {len(drones)}, Obstacles: {len(env.obstacles)}")
```

## Extending the Framework

### Adding a New Algorithm

1. Create `algorithms/my_algo.py`
2. Inherit from `MOEABase`
3. Implement the `run()` method
4. Add to `ALGORITHM_REGISTRY` in `algorithms/__init__.py`
5. Add config entry in `experiments/configs.yaml`

### Adding New Objectives

1. Modify `DroneSwarmMOP.evaluate()` in `core/problem_definition.py`
2. Update `n_obj` property
3. Adjust reference point in `configs.yaml`

### Scaling to More Drones

Set `n_drones` in config. The framework supports 20+ drones but computation time scales with swarm size. Consider reducing `n_gen` or `pop_size` for larger swarms. Use the `high_density` 3-D scenario for stress testing.

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