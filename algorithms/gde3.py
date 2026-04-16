"""GDE3: Generalized Differential Evolution 3.

Reference:
    S. Kukkonen and J. Lampinen,
    "GDE3: The third Evolution Step of Generalized Differential Evolution,"
    2005 IEEE Congress on Evolutionary Computation (CEC), 2005.

GDE3 extends the classic DE/rand/1/bin scheme to multi-objective problems
by combining differential evolution operators with non-dominated sorting
and crowding-distance-based pruning.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase


class GDE3(MOEABase):
    """Generalized Differential Evolution 3 for multi-objective optimization.

    Uses DE/rand/1/bin mutation and crossover to generate trial vectors.
    Selection retains non-dominated solutions, and the population is pruned
    back to *pop_size* via non-dominated sorting with crowding distance.

    Parameters
    ----------
    problem : object
        Problem instance with ``evaluate(x)``, ``n_var``, ``n_obj``, and
        ``bounds`` attributes.
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
    F : float
        Differential-evolution scale factor (mutation weight).
    CR : float
        Crossover rate for binomial crossover.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        F: float = 0.5,
        CR: float = 0.9,
        seed: int | None = None,
    ):
        super().__init__(problem, pop_size, n_gen, seed)
        self.F = F
        self.CR = CR

    # ------------------------------------------------------------------
    # DE operators
    # ------------------------------------------------------------------

    def _de_mutation(self, pop: np.ndarray, target_idx: int, F: float) -> np.ndarray:
        """DE/rand/1 mutation: ``v = x_r1 + F * (x_r2 - x_r3)``.

        Three mutually exclusive indices (also distinct from *target_idx*)
        are selected uniformly at random from the population.

        Parameters
        ----------
        pop : np.ndarray
            Current population of shape ``(pop_size, n_var)``.
        target_idx : int
            Index of the target vector (excluded from donor selection).
        F : float
            Scale factor.

        Returns
        -------
        np.ndarray
            Mutant vector of shape ``(n_var,)``.
        """
        idxs = list(range(len(pop)))
        idxs.remove(target_idx)
        r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
        return pop[r1] + F * (pop[r2] - pop[r3])

    def _de_crossover(
        self, target: np.ndarray, mutant: np.ndarray, CR: float
    ) -> np.ndarray:
        """Binomial (uniform) crossover.

        Each variable is taken from the mutant with probability *CR* and
        from the target otherwise.  At least one variable is guaranteed to
        come from the mutant.

        Parameters
        ----------
        target : np.ndarray
            Target vector of shape ``(n_var,)``.
        mutant : np.ndarray
            Mutant vector of shape ``(n_var,)``.
        CR : float
            Crossover probability.

        Returns
        -------
        np.ndarray
            Trial vector of shape ``(n_var,)``.
        """
        n = len(target)
        j_rand = self.rng.integers(0, n)
        mask = self.rng.random(n) < CR
        mask[j_rand] = True  # ensure at least one variable from mutant
        trial = np.where(mask, mutant, target)
        return trial

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute GDE3.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(pareto_solutions, pareto_objectives)`` – the non-dominated
            set discovered after *n_gen* generations.
        """
        # --- Initialization ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)

        for gen in range(self.n_gen):
            t_start = time.time()

            pool_pop = []
            pool_obj = []

            for i in range(len(pop)):
                # DE/rand/1 mutation + binomial crossover
                mutant = self._de_mutation(pop, i, self.F)
                trial = self._de_crossover(pop[i], mutant, self.CR)
                trial = np.clip(trial, self.lower, self.upper)

                trial_obj = self.problem.evaluate(trial)

                # Pairwise dominance comparison
                if self._dominates(trial_obj, objectives[i]):
                    # Trial dominates target → replace
                    pool_pop.append(trial)
                    pool_obj.append(trial_obj)
                elif self._dominates(objectives[i], trial_obj):
                    # Target dominates trial → keep target
                    pool_pop.append(pop[i])
                    pool_obj.append(objectives[i])
                else:
                    # Non-dominated → keep both for later pruning
                    pool_pop.append(pop[i])
                    pool_obj.append(objectives[i])
                    pool_pop.append(trial)
                    pool_obj.append(trial_obj)

            pool_pop = np.array(pool_pop)
            pool_obj = np.array(pool_obj)

            # --- Prune back to pop_size via NDS + crowding distance ---
            if len(pool_pop) > self.pop_size:
                fronts = self.fast_non_dominated_sort(pool_obj)
                new_pop = []
                new_obj = []
                for front in fronts:
                    if len(new_pop) + len(front) <= self.pop_size:
                        for idx in front:
                            new_pop.append(pool_pop[idx])
                            new_obj.append(pool_obj[idx])
                    else:
                        remaining = self.pop_size - len(new_pop)
                        cd = self.crowding_distance(pool_obj[front])
                        sorted_indices = np.argsort(-cd)  # descending
                        for k in sorted_indices[:remaining]:
                            idx = front[k]
                            new_pop.append(pool_pop[idx])
                            new_obj.append(pool_obj[idx])
                        break
                pop = np.array(new_pop)
                objectives = np.array(new_obj)
            else:
                pop = pool_pop
                objectives = pool_obj

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
                    f"GDE3     gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"pool size: {len(pool_pop):>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
