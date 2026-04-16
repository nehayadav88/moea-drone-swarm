"""AGE-MOEA: Adaptive Geometry Estimation MOEA.

Reference:
    A. Panichella,
    "An Adaptive Evolutionary Algorithm based on Non-Euclidean Geometry
    for Many-objective Optimization,"
    Proceedings of the Genetic and Evolutionary Computation Conference
    (GECCO), 2019.

AGE-MOEA replaces NSGA-II's crowding distance with a *survival score*
derived from the estimated geometry (curvature) of the current Pareto
front.  A parameter *p* is computed such that the Lp-norm distance
between solutions reflects the true shape of the front (convex → p < 1,
linear → p ≈ 1, concave → p > 1).  Diversity is then maintained by
keeping solutions that are most isolated in Lp-norm space.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase


class AGEMOEA(MOEABase):
    """Adaptive Geometry Estimation MOEA for many-objective optimisation.

    Like NSGA-II the algorithm uses fast non-dominated sorting to rank
    the combined parent + offspring population, but the critical front
    (the one that is only partially accepted) is trimmed using a
    geometry-aware survival score instead of crowding distance.

    Parameters
    ----------
    problem : object
        Problem instance with ``evaluate(x)``, ``n_var``, ``n_obj``,
        and ``bounds`` attributes.
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
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
        crossover_eta: float = 20,
        mutation_eta: float = 20,
        seed: int | None = None,
    ):
        super().__init__(problem, pop_size, n_gen, seed)
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

    # ------------------------------------------------------------------
    # Geometry estimation
    # ------------------------------------------------------------------

    def _compute_geometry(self, front_obj: np.ndarray) -> float:
        """Estimate the curvature parameter *p* of the Pareto front.

        The method normalises the objectives to [0, 1], identifies the
        extreme points of the front, and uses binary search to find the
        value of *p* such that the Lp distance from each extreme to the
        ideal point equals 1 (the expected distance on a unit simplex
        under the Lp norm).

        Parameters
        ----------
        front_obj : np.ndarray
            Objective values of the front, shape ``(n, n_obj)``.

        Returns
        -------
        float
            Estimated curvature parameter *p*, clamped to [0.1, 20].
        """
        n, m = front_obj.shape
        if n <= 1 or m <= 1:
            return 1.0

        ideal = front_obj.min(axis=0)
        nadir = front_obj.max(axis=0)
        rng = nadir - ideal
        rng[rng < 1e-14] = 1e-14

        # Normalise to [0, 1]
        norm_obj = (front_obj - ideal) / rng

        # Identify extreme points (best in each objective)
        extremes = []
        for j in range(m):
            extremes.append(norm_obj[np.argmin(norm_obj[:, j])])
        extremes = np.array(extremes)

        # Target: Lp distance from extreme to the ideal ([0]*m) should
        # equal 1 on a unit simplex.  We search for p that makes the
        # *mean* Lp distance across extremes equal to 1.
        def _mean_lp_dist(p: float) -> float:
            # Lp distance from each extreme to the origin
            dists = np.sum(np.abs(extremes) ** p, axis=1) ** (1.0 / p)
            return float(np.mean(dists))

        # Binary search for p in [0.1, 20]
        lo, hi = 0.1, 20.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _mean_lp_dist(mid) < 1.0:
                hi = mid
            else:
                lo = mid

        p = (lo + hi) / 2.0
        return float(np.clip(p, 0.1, 20.0))

    # ------------------------------------------------------------------
    # Survival score
    # ------------------------------------------------------------------

    def _compute_survival_score(
        self, objectives: np.ndarray, p: float
    ) -> np.ndarray:
        """Compute the survival score for a set of solutions.

        Solutions that are more isolated in Lp-norm space receive a
        higher score.  Boundary solutions (best/worst in any single
        objective) are assigned infinity so they are always retained.

        Parameters
        ----------
        objectives : np.ndarray
            Objective values, shape ``(n, n_obj)``.
        p : float
            Geometry parameter for the Lp norm.

        Returns
        -------
        np.ndarray
            Survival scores, shape ``(n,)``.
        """
        n, m = objectives.shape
        if n <= 2:
            return np.full(n, np.inf)

        # Normalise to [0, 1]
        ideal = objectives.min(axis=0)
        nadir = objectives.max(axis=0)
        rng = nadir - ideal
        rng[rng < 1e-14] = 1e-14
        norm = (objectives - ideal) / rng

        scores = np.full(n, np.inf)

        # For each objective dimension, sort and compute pairwise Lp
        # distances between adjacent solutions.  Boundary points in each
        # sorted order keep their infinite score.
        min_lp = np.full(n, np.inf)

        for j in range(m):
            order = np.argsort(norm[:, j])
            for k in range(1, n - 1):
                prev_idx = order[k - 1]
                curr_idx = order[k]
                next_idx = order[k + 1]

                d_prev = np.sum(np.abs(norm[curr_idx] - norm[prev_idx]) ** p) ** (
                    1.0 / p
                )
                d_next = np.sum(np.abs(norm[curr_idx] - norm[next_idx]) ** p) ** (
                    1.0 / p
                )
                d_min = min(d_prev, d_next)
                min_lp[curr_idx] = min(min_lp[curr_idx], d_min)

        # Boundary solutions: best or worst in any single objective
        is_boundary = np.zeros(n, dtype=bool)
        for j in range(m):
            order = np.argsort(norm[:, j])
            is_boundary[order[0]] = True
            is_boundary[order[-1]] = True

        scores[~is_boundary] = min_lp[~is_boundary]
        # Boundary solutions already have np.inf from initialisation
        return scores

    # ------------------------------------------------------------------
    # Selection from the critical front
    # ------------------------------------------------------------------

    def _select_from_front(
        self,
        front_indices: list[int],
        objectives: np.ndarray,
        n_select: int,
    ) -> list[int]:
        """Select *n_select* individuals from a front via survival score.

        Parameters
        ----------
        front_indices : list[int]
            Indices (into the full population) of individuals on this
            front.
        objectives : np.ndarray
            Full objective array, shape ``(N, n_obj)``.
        n_select : int
            Number of individuals to select.

        Returns
        -------
        list[int]
            Selected indices (into the full population).
        """
        if n_select >= len(front_indices):
            return list(front_indices)

        front_obj = objectives[front_indices]
        p = self._compute_geometry(front_obj)
        scores = self._compute_survival_score(front_obj, p)

        # Select the n_select with the highest survival scores
        local_order = np.argsort(-scores)[:n_select]
        return [front_indices[i] for i in local_order]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute AGE-MOEA.

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

            # --- Build rank / survival arrays for mating selection ---
            fronts = self.fast_non_dominated_sort(objectives)
            ranks = np.zeros(len(pop), dtype=int)
            survival_all = np.zeros(len(pop))

            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                if len(front) > 0:
                    front_obj = objectives[front]
                    p = self._compute_geometry(front_obj)
                    ss = self._compute_survival_score(front_obj, p)
                    for k, idx in enumerate(front):
                        survival_all[idx] = ss[k]

            # --- Create offspring ---
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, ranks, survival_all)
                p2 = self._tournament(pop, ranks, survival_all)
                c1, c2 = self.sbx_crossover(p1, p2, eta=self.crossover_eta)
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[: self.pop_size])

            # --- Evaluate offspring ---
            off_obj = self.evaluate_population(offspring)

            # --- Combine parent + offspring (2N) ---
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([objectives, off_obj])

            # --- Non-dominated sort on combined population ---
            fronts = self.fast_non_dominated_sort(combined_obj)

            # --- Select next generation of size N ---
            new_pop: list[np.ndarray] = []
            new_obj: list[np.ndarray] = []

            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    # Critical front – use survival score to select
                    remaining = self.pop_size - len(new_pop)
                    selected = self._select_from_front(
                        front, combined_obj, remaining
                    )
                    for idx in selected:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                    break

            pop = np.array(new_pop)
            objectives = np.array(new_obj)

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
                    f"AGE-MOEA gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)

    # ------------------------------------------------------------------
    # Tournament helper
    # ------------------------------------------------------------------

    def _tournament(
        self,
        pop: np.ndarray,
        ranks: np.ndarray,
        survival: np.ndarray,
    ) -> np.ndarray:
        """Binary tournament: prefer lower rank, then higher survival score.

        Parameters
        ----------
        pop : np.ndarray
            Population array.
        ranks : np.ndarray
            Pareto rank per individual.
        survival : np.ndarray
            Survival score per individual.

        Returns
        -------
        np.ndarray
            Selected individual (1-D array).
        """
        n = len(pop)
        i, j = self.rng.integers(0, n, size=2)
        if ranks[i] < ranks[j]:
            return pop[i]
        elif ranks[j] < ranks[i]:
            return pop[j]
        else:
            return pop[i] if survival[i] >= survival[j] else pop[j]
