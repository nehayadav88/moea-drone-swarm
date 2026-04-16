"""HypE: Hypervolume Estimation Algorithm for Many-Objective Optimization.

Reference:
    J. Bader and E. Zitzler,
    "HypE: An Algorithm for Fast Hypervolume-Based Many-Objective
    Optimization,"
    Evolutionary Computation, vol. 19, no. 1, pp. 45-76, 2011.

HypE approximates the hypervolume contribution of each solution via
Monte Carlo sampling, making it tractable for problems with many
objectives where exact hypervolume computation is prohibitively
expensive.  Selection pressure toward both convergence and diversity
is achieved by preferring solutions with higher estimated hypervolume
contributions.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase


class HypE(MOEABase):
    """Hypervolume Estimation Algorithm (HypE).

    Uses Monte Carlo sampling to estimate the exclusive hypervolume
    contribution of each individual.  Environmental selection fills
    successive non-dominated fronts, trimming the critical front by
    estimated hypervolume contribution.  Mating selection uses binary
    tournament on the same HV fitness values.

    Parameters
    ----------
    problem : object
        Problem instance with ``evaluate(x)``, ``n_var``, ``n_obj``,
        and ``bounds`` attributes.
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
    n_samples : int
        Number of Monte Carlo samples for HV estimation.
    crossover_eta : float
        Distribution index for SBX crossover.
    mutation_eta : float
        Distribution index for polynomial mutation.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        n_samples: int = 10000,
        crossover_eta: float = 20,
        mutation_eta: float = 20,
        seed: int | None = None,
    ):
        super().__init__(problem, pop_size, n_gen, seed)
        self.n_samples = n_samples
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

    # ------------------------------------------------------------------
    # Monte Carlo HV contribution estimation
    # ------------------------------------------------------------------

    def _estimate_hv_contributions(
        self,
        objectives: np.ndarray,
        reference_point: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Estimate exclusive hypervolume contributions via Monte Carlo.

        Random points are sampled uniformly in the box
        ``[ideal, reference_point]``.  For each sample the set of
        dominating solutions is determined.  If exactly one solution
        dominates the sample, it receives a full contribution credit;
        if *k* solutions dominate it, each receives 1/k.  The raw
        counts are then scaled by the box volume / n_samples to yield
        an absolute HV contribution estimate.

        Parameters
        ----------
        objectives : np.ndarray
            Objective matrix, shape ``(n, n_obj)``.
        reference_point : np.ndarray
            Reference point for the HV computation, shape ``(n_obj,)``.
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        np.ndarray
            Estimated HV contribution per solution, shape ``(n,)``.
        """
        n, m = objectives.shape
        ideal = objectives.min(axis=0)

        # Uniform samples in [ideal, reference_point]
        samples = self.rng.uniform(ideal, reference_point, size=(n_samples, m))

        # Determine which solutions dominate each sample
        # dominated(i, j) = True iff solution j dominates sample i
        # solution j dominates sample i  <=>  all(objectives[j] <= samples[i])
        # Shape: (n_samples, n)
        dominated = np.all(
            objectives[np.newaxis, :, :] <= samples[:, np.newaxis, :], axis=2
        )

        dom_counts = dominated.sum(axis=1)  # (n_samples,)

        # Contributions: fractional credit
        contributions = np.zeros(n)
        valid = dom_counts > 0
        # For each valid sample, add 1/k to every dominating solution
        fractions = np.zeros(n_samples)
        fractions[valid] = 1.0 / dom_counts[valid]

        # Vectorised accumulation
        weighted = dominated * fractions[:, np.newaxis]  # (n_samples, n)
        contributions = weighted.sum(axis=0)  # (n,)

        # Scale by box volume / n_samples
        box_volume = float(np.prod(reference_point - ideal))
        if box_volume > 0.0:
            contributions *= box_volume / n_samples

        return contributions

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    def _environmental_selection(
        self,
        pop: np.ndarray,
        objectives: np.ndarray,
        reference_point: np.ndarray,
        target_size: int,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select *target_size* survivors using non-dominated sorting + HV.

        Complete non-dominated fronts are added in order.  The critical
        (partially accepted) front is trimmed by keeping the individuals
        with the highest estimated hypervolume contributions.

        Parameters
        ----------
        pop : np.ndarray
            Decision variable array, shape ``(n, n_var)``.
        objectives : np.ndarray
            Objective array, shape ``(n, n_obj)``.
        reference_point : np.ndarray
            Reference point for HV estimation.
        target_size : int
            Desired survivor count.
        n_samples : int
            Monte Carlo samples for HV estimation on the critical front.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(selected_pop, selected_obj)``
        """
        fronts = self.fast_non_dominated_sort(objectives)

        new_pop: list[np.ndarray] = []
        new_obj: list[np.ndarray] = []

        for front in fronts:
            if len(new_pop) + len(front) <= target_size:
                for idx in front:
                    new_pop.append(pop[idx])
                    new_obj.append(objectives[idx])
            else:
                remaining = target_size - len(new_pop)
                if remaining <= 0:
                    break

                front_obj = objectives[front]
                contrib = self._estimate_hv_contributions(
                    front_obj, reference_point, n_samples
                )
                # Select the *remaining* individuals with highest contribution
                local_order = np.argsort(-contrib)[:remaining]
                for k in local_order:
                    idx = front[k]
                    new_pop.append(pop[idx])
                    new_obj.append(objectives[idx])
                break

        return np.array(new_pop), np.array(new_obj)

    # ------------------------------------------------------------------
    # Mating tournament
    # ------------------------------------------------------------------

    def _binary_tournament(
        self, pop: np.ndarray, fitness: np.ndarray
    ) -> np.ndarray:
        """Binary tournament selection based on fitness (higher = better).

        Parameters
        ----------
        pop : np.ndarray
            Population array.
        fitness : np.ndarray
            Fitness value per individual (higher is preferred).

        Returns
        -------
        np.ndarray
            Selected individual (1-D array).
        """
        n = len(pop)
        i, j = self.rng.integers(0, n, size=2)
        return pop[i] if fitness[i] >= fitness[j] else pop[j]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute HypE.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(pareto_solutions, pareto_objectives)`` – the non-dominated
            set discovered after *n_gen* generations.
        """
        # --- Initialisation ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)

        for gen in range(self.n_gen):
            t_start = time.time()

            # --- Reference point: 1.1 × nadir of current population ---
            nadir = objectives.max(axis=0)
            reference_point = 1.1 * nadir

            # --- HV-contribution fitness for mating selection ---
            hv_fitness = self._estimate_hv_contributions(
                objectives, reference_point, self.n_samples
            )

            # --- Create offspring via binary tournament + SBX + mutation ---
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._binary_tournament(pop, hv_fitness)
                p2 = self._binary_tournament(pop, hv_fitness)
                c1, c2 = self.sbx_crossover(p1, p2, eta=self.crossover_eta)
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[: self.pop_size])

            # --- Evaluate offspring ---
            off_obj = self.evaluate_population(offspring)

            # --- Combine parent + offspring ---
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([objectives, off_obj])

            # --- Environmental selection using HV contributions ---
            # Recompute reference point from combined population
            combined_nadir = combined_obj.max(axis=0)
            combined_ref = 1.1 * combined_nadir

            pop, objectives = self._environmental_selection(
                combined_pop, combined_obj, combined_ref,
                self.pop_size, self.n_samples,
            )

            # --- History tracking ---
            pf_fronts = self.fast_non_dominated_sort(objectives)
            if pf_fronts:
                pf_obj = objectives[pf_fronts[0]]
            else:
                pf_obj = objectives
            self.history['objectives'].append(pf_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                n_pf = len(pf_obj)
                print(
                    f"HypE     gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
