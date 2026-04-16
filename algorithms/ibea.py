"""IBEA: Indicator-Based Evolutionary Algorithm.

Reference:
    E. Zitzler and S. Künzli,
    "Indicator-Based Selection in Multiobjective Search,"
    Parallel Problem Solving from Nature (PPSN VIII),
    Lecture Notes in Computer Science, vol. 3242,
    pp. 832-842, Springer, 2004.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase


class IBEA(MOEABase):
    """Indicator-Based Evolutionary Algorithm with additive epsilon indicator.

    IBEA drives selection entirely through a binary quality indicator,
    avoiding the need for Pareto ranking or diversity maintenance
    mechanisms.  Each individual's fitness is the aggregated pair-wise
    indicator contribution from every other member of the population.
    Environmental selection iteratively removes the individual with the
    worst fitness and updates the remaining fitness values incrementally.

    This implementation uses the *additive epsilon indicator* I_ε+:

        I_ε+(a, b) = max_j (a_j − b_j)

    which measures the minimum additive translation of *a* needed to
    weakly dominate *b*.  A smaller value indicates that *a* is closer to
    dominating *b*.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        kappa: float = 0.05,
        crossover_eta: float = 20,
        mutation_eta: float = 20,
        seed: int | None = None,
    ):
        """
        Args:
            problem: Problem instance with evaluate, n_var, n_obj, bounds.
            pop_size: Population size (N).
            n_gen: Number of generations.
            kappa: Scaling factor that controls the influence of the
                indicator in the fitness computation.  Smaller values
                amplify differences.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.kappa = kappa
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

    # ------------------------------------------------------------------
    # Additive epsilon indicator
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_epsilon_indicator(a: np.ndarray, b: np.ndarray) -> float:
        """Compute the additive epsilon indicator I_ε+(a, b).

        I_ε+(a, b) = max_j (a_j − b_j)

        A smaller (more negative) value means *a* is better relative to *b*
        because *a* already dominates *b* or needs only a small shift.

        Args:
            a: (n_obj,) objective vector of solution a.
            b: (n_obj,) objective vector of solution b.

        Returns:
            Scalar epsilon indicator value.
        """
        return float(np.max(a - b))

    # ------------------------------------------------------------------
    # Fitness computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fitness(
        objectives: np.ndarray, kappa: float
    ) -> np.ndarray:
        """Compute IBEA fitness for every individual.

        fitness(i) = Σ_{j ≠ i}  −exp( −I_ε+(j, i) / (κ · c) )

        where *c* is the range of indicator values across all pairs
        (used for scaling).  Higher fitness is better.

        Args:
            objectives: (n, n_obj) objective values.
            kappa: Scaling factor.

        Returns:
            fitness: (n,) array — higher is better.
        """
        n = len(objectives)

        # Pairwise epsilon indicator matrix: eps[j, i] = I_ε+(j, i)
        # eps[j, i] = max_m (objectives[j, m] - objectives[i, m])
        # Vectorised: (n, 1, n_obj) - (1, n, n_obj) -> max over axis 2
        diff = objectives[:, np.newaxis, :] - objectives[np.newaxis, :, :]
        eps = diff.max(axis=2)  # (n, n)

        # Scaling constant c: range of indicator values
        np.fill_diagonal(eps, np.nan)
        c = np.nanmax(eps) - np.nanmin(eps)
        if c < 1e-14:
            c = 1e-14
        np.fill_diagonal(eps, 0.0)

        # fitness(i) = Σ_{j≠i} -exp(-eps[j, i] / (kappa * c))
        # eps[j, i] is row j, col i  ->  sum over rows (axis=0)
        exponents = -eps / (kappa * c)
        # Clamp for numerical safety
        exponents = np.clip(exponents, -500.0, 500.0)
        contributions = -np.exp(exponents)  # (n, n)
        np.fill_diagonal(contributions, 0.0)
        fitness = contributions.sum(axis=0)  # (n,)

        return fitness

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    @staticmethod
    def _environmental_selection(
        pop: np.ndarray,
        objectives: np.ndarray,
        pop_size: int,
        kappa: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iteratively remove the worst individual until |pop| == pop_size.

        After removing individual *x*, the fitness of every remaining
        individual *i* is updated efficiently:

            fitness(i) += exp( −I_ε+(x, i) / (κ · c) )

        This avoids recomputing fitness from scratch at each removal step.

        Args:
            pop: (m, n_var) combined population.
            objectives: (m, n_obj) objective values.
            pop_size: Target population size.
            kappa: Scaling factor.

        Returns:
            (pop_sel, obj_sel): selected population and objectives.
        """
        n = len(pop)
        if n <= pop_size:
            return pop.copy(), objectives.copy()

        # Full pairwise epsilon indicator
        diff = (
            objectives[:, np.newaxis, :] - objectives[np.newaxis, :, :]
        )
        eps = diff.max(axis=2)  # (n, n)

        # Scaling constant
        diag_backup = eps.diagonal().copy()
        np.fill_diagonal(eps, np.nan)
        c = np.nanmax(eps) - np.nanmin(eps)
        if c < 1e-14:
            c = 1e-14
        np.fill_diagonal(eps, diag_backup)

        # Initial fitness
        exponents = -eps / (kappa * c)
        exponents = np.clip(exponents, -500.0, 500.0)
        contributions = -np.exp(exponents)
        np.fill_diagonal(contributions, 0.0)
        fitness = contributions.sum(axis=0)

        alive = list(range(n))

        while len(alive) > pop_size:
            # Find worst (smallest fitness) among alive
            worst_pos = int(np.argmin(fitness[alive]))
            worst_idx = alive[worst_pos]

            # Update fitness of remaining individuals before removal
            for i in alive:
                if i == worst_idx:
                    continue
                # Undo contribution of worst to i
                fitness[i] += np.exp(-eps[worst_idx, i] / (kappa * c))

            alive.pop(worst_pos)

        return pop[alive], objectives[alive]

    # ------------------------------------------------------------------
    # Binary tournament on fitness
    # ------------------------------------------------------------------

    def _binary_tournament(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Select one individual via binary tournament (higher fitness wins).

        Args:
            pop: (n, n_var) population.
            fitness: (n,) fitness values (higher is better).

        Returns:
            Selected individual (1-D array).
        """
        n = len(pop)
        i, j = self.rng.integers(0, n, size=2)
        return pop[i] if fitness[i] >= fitness[j] else pop[j]

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute IBEA.

        Returns:
            (pareto_solutions, pareto_objectives): numpy arrays for the
            non-dominated set found after *n_gen* generations.
        """
        # --- Initialisation ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)

        for gen in range(self.n_gen):
            t_start = time.time()

            # --- Compute fitness for current population ---
            fitness = self._compute_fitness(objectives, self.kappa)

            # --- Mating selection via binary tournament ---
            offspring: list[np.ndarray] = []
            while len(offspring) < self.pop_size:
                p1 = self._binary_tournament(pop, fitness)
                p2 = self._binary_tournament(pop, fitness)
                c1, c2 = self.sbx_crossover(
                    p1, p2, eta=self.crossover_eta
                )
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])
            offspring_arr = np.array(offspring[: self.pop_size])
            off_obj = self.evaluate_population(offspring_arr)

            # --- Combine parent + offspring ---
            combined_pop = np.vstack([pop, offspring_arr])
            combined_obj = np.vstack([objectives, off_obj])

            # --- Environmental selection ---
            pop, objectives = self._environmental_selection(
                combined_pop, combined_obj, self.pop_size, self.kappa
            )

            # --- History tracking ---
            pf_fronts = self.fast_non_dominated_sort(objectives)
            pf_obj = objectives[pf_fronts[0]] if pf_fronts else objectives
            self.history['objectives'].append(pf_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                n_pf = len(pf_obj)
                print(
                    f"IBEA     gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"pop size: {len(pop):>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
