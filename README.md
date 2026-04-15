# Heterogeneous Drone Swarm Path Planning (MOEA Framework)

Research-grade Python framework for multi-objective path planning of heterogeneous drone swarms with configurable environment, constraints, energy model, MOEAs, metrics, and visualization.

## Problem Formulation

We solve:

\[
\min_{\mathbf{x}}\; \mathbf{f}(\mathbf{x}) = [f_1, f_2, f_3, f_4]
\]

where decision vector \(\mathbf{x}\) encodes waypoint coordinates for each drone.

Objectives (all minimized):

1. **Total Path Length**
   \[
   f_1 = \sum_{d=1}^{N} \sum_{k=0}^{K_d-1} \|p_{d,k+1} - p_{d,k}\|_2
   \]
2. **Total Flight Time**
   \[
   f_2 = \sum_{d=1}^{N} \frac{L_d}{v_d}
   \]
3. **Comprehensive Energy Consumption**
   \[
   f_3 = \sum_{d=1}^{N} E_d
   \]
4. **Safety / Risk Objective**
   \[
   f_4 = \text{CollisionRisk}(\mathbf{x}) + \sum_i \max(0, g_i(\mathbf{x}))
   \]

Constraints:

- Energy per drone: \(g_{1,d} = E_d - B_d \le 0\)
- Communication stability: \(g_2 = \rho_{min} - \rho_{conn} \le 0\)
- Path feasibility (obstacle avoidance): \(g_3 \le 0\) (binary violation count)
- Task completion: \(g_4 = T_{req} - T_{done} \le 0\)

## Energy Model

Implemented in `core/energy_model.py` with configurable coefficients:

- Distance-dependent propulsion energy
- Speed factor
- Mass + payload influence
- Hovering energy
- Turning-angle penalty

## Repository Structure

- `core/`
  - `problem_definition.py`
  - `energy_model.py`
  - `constraints.py`
- `algorithms/`
  - `nsga2.py`
  - `moead.py`
  - `spea2.py`
- `simulation/`
  - `environment.py`
  - `drone.py`
  - `path_planner.py`
- `metrics/`
  - `hv.py`
  - `igd.py`
  - `diversity.py`
  - `statistics.py`
- `experiments/`
  - `run_experiments.py`
  - `configs.yaml`
- `visualization/`
  - `plots.py`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pyyaml
```

## Run Experiments

```bash
python /home/runner/work/moea-drone-swarm/moea-drone-swarm/experiments/run_experiments.py
```

Outputs are written to `results/` (configurable):

- `summary.json` (HV, PD, IGD, spread, stats)
- `convergence.json`
- `pareto.png`
- `convergence.png`

## Configuration

Edit `/home/runner/work/moea-drone-swarm/moea-drone-swarm/experiments/configs.yaml`:

- Number of runs (default `30`)
- Random seeds for reproducibility
- Drone heterogeneity (battery/speed/sensing/payload)
- Environment and obstacle definitions
- Algorithm hyperparameters

## Testing

Run unit tests:

```bash
python -m unittest discover -s /home/runner/work/moea-drone-swarm/moea-drone-swarm/tests -q
```

## Notes

- Supports parallel objective evaluation in algorithms.
- Designed for scalability to larger swarms by configurable population sizes and generations.
- Easy to extend with NSGA-III, MOPSO, hybrid operators, RL-assisted selection, dynamic replanning.
