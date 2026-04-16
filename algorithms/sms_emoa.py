"""SMS-EMOA: S-Metric Selection Evolutionary Multi-Objective Algorithm.

Reference:
    N. Beume, B. Naujoks, and M. Emmerich,
    "SMS-EMOA: Multiobjective selection based on dominated hypervolume,"
    European Journal of Operational Research, vol. 181, no. 3, 2007.

SMS-EMOA is a steady-state MOEA that produces one offspring per iteration
and removes the individual with the smallest hypervolume contribution
from the worst non-dominated front.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase
from metrics.hv import compute_hypervolume


class SMSEMOA(MOEABase):
    """S-Metric Selection EMOA for multi-objective optimization.

    A steady-state algorithm that generates a single offspring per
    iteration and uses hypervolume contribution to decide which
    individual to discard.  For large worst-fronts (>50 members) the
    algorithm falls back to crowding distance for computational
    tractability.

    Parameters
    ----------
    problem : object
        Problem instance with ``evaluate(x)``, ``n_var``, ``n_obj``, and
        ``bounds`` attributes.
    pop_size : int
        Population size.
    n_gen : int
        Number of *generations*.  Each generation corresponds to
        ``pop_size`` offspring evaluations (steady-state iterations).
    crossover_eta : float
        Distribution index for SBX crossover.
    mutation_eta : float
        Distribution index for polynomial mutation.
    seed : int or None
        Random seed for reproducibility.
    """

    # Threshold above which crowding distance is used instead of HV
    _HV_FRONT_LIMIT = 50

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
    # Hypervolume-based selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hv_contribution(
        objectives: np.ndarray, reference_point: np.ndarray, idx: int
    ) -> float:
        """Compute the exclusive hypervolume contribution of solution *idx*.

        HV contribution = HV(full set) − HV(set without *idx*).

        Parameters
        ----------
        objectives : np.ndarray
            Objective matrix of shape ``(n, n_obj)`` for the front under
            consideration.
        reference_point : np.ndarray
            Reference point for the HV computation.
        idx : int
            Index of the solution whose contribution is measured.

        Returns
        -------
        float
            Exclusive hypervolume contribution (≥ 0).
        """
        hv_full = compute_hypervolume(objectives, reference_point)
        reduced = np.delete(objectives, idx, axis=0)
        if reduced.shape[0] == 0:
            return hv_full
        hv_without = compute_hypervolume(reduced, reference_point)
        return hv_full - hv_without

    def _select_for_removal(
        self,
        pop: np.ndarray,
        objectives: np.ndarray,
        reference_point: np.ndarray,
    ) -> int:
        """Identify the individual to remove from the population.

        1. Perform non-dominated sorting.
        2. Locate the worst (last) front.
        3. If the worst front has a single member, remove it.
        4. Otherwise, remove the member with the smallest hypervolume
           contribution.  For fronts larger than ``_HV_FRONT_LIMIT``,
           crowding distance is used as a faster surrogate.

        Parameters
        ----------
        pop : np.ndarray
            Population array of shape ``(n, n_var)``.
        objectives : np.ndarray
            Objective array of shape ``(n, n_obj)``.
        reference_point : np.ndarray
            Reference point for HV computation.

        Returns
        -------
        int
            Index (into *pop* / *objectives*) of the individual to remove.
        """
        fronts = self.fast_non_dominated_sort(objectives)

        # Worst front = last front in the list
        worst_front = fronts[-1]

        if len(worst_front) == 1:
            return worst_front[0]

        front_indices = np.array(worst_front)
        front_obj = objectives[front_indices]

        if len(worst_front) > self._HV_FRONT_LIMIT:
            # Fallback: crowding distance – remove the most crowded
            cd = self.crowding_distance(front_obj)
            # Among finite distances pick the smallest; if all inf pick first
            finite_mask = np.isfinite(cd)
            if np.any(finite_mask):
                local_idx = np.where(finite_mask)[0][np.argmin(cd[finite_mask])]
            else:
                local_idx = 0
            return front_indices[local_idx]

        # HV contribution for each member of the worst front
        contributions = np.array(
            [
                self._hv_contribution(front_obj, reference_point, k)
                for k in range(len(front_obj))
            ]
        )
        local_idx = int(np.argmin(contributions))
        return front_indices[local_idx]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute SMS-EMOA.

        The main loop performs ``pop_size * n_gen`` steady-state iterations.
        Every ``pop_size`` iterations the current Pareto front is recorded
        as one *generation* for history tracking.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(pareto_solutions, pareto_objectives)`` – the non-dominated
            set discovered after all iterations.
        """
        # --- Initialization ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)

        total_iterations = self.pop_size * self.n_gen
        gen_counter = 0  # tracks completed "generations"

        for iteration in range(total_iterations):
            # Track time per logical generation
            if iteration % self.pop_size == 0:
                t_start = time.time()

            # --- Generate one offspring ---
            p1 = self.tournament_selection(pop, objectives)
            p2 = self.tournament_selection(pop, objectives)
            c1, _c2 = self.sbx_crossover(p1, p2, eta=self.crossover_eta)
            offspring = self.polynomial_mutation(c1, eta=self.mutation_eta)
            offspring = np.clip(offspring, self.lower, self.upper)
            off_obj = np.asarray(self.problem.evaluate(offspring), dtype=np.float64)

            # --- Add offspring (pop_size → pop_size + 1) ---
            pop = np.vstack([pop, offspring.reshape(1, -1)])
            objectives = np.vstack([objectives, off_obj.reshape(1, -1)])

            # --- Remove worst individual via HV contribution ---
            ref_point = 1.1 * np.max(objectives, axis=0)
            remove_idx = self._select_for_removal(pop, objectives, ref_point)

            pop = np.delete(pop, remove_idx, axis=0)
            objectives = np.delete(objectives, remove_idx, axis=0)

            # --- History tracking (once per logical generation) ---
            if (iteration + 1) % self.pop_size == 0:
                gen_counter += 1
                elapsed = time.time() - t_start
                self.history['time_per_gen'].append(elapsed)

                pf_fronts = self.fast_non_dominated_sort(objectives)
                if pf_fronts:
                    pf_obj = objectives[pf_fronts[0]]
                else:
                    pf_obj = objectives
                self.history['objectives'].append(pf_obj.copy())

                if gen_counter % 10 == 0 or gen_counter == 1:
                    n_pf = len(pf_obj)
                    print(
                        f"SMS-EMOA gen {gen_counter:>4d}/{self.n_gen}  |  "
                        f"front size: {n_pf:>4d}  |  "
                        f"time: {elapsed:.3f}s"
                    )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
