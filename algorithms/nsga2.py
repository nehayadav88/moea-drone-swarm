"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Reference:
    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan,
    "A fast and elitist multiobjective genetic algorithm: NSGA-II,"
    IEEE Transactions on Evolutionary Computation, 2002.
"""
import numpy as np
import time

from .base import MOEABase


class NSGA2(MOEABase):
    """NSGA-II with constraint-domination principle for constrained problems.

    Uses fast non-dominated sorting, crowding distance assignment, and
    elitist selection to evolve a diverse set of Pareto-optimal solutions.
    """

    def __init__(self, problem, pop_size=100, n_gen=200, crossover_eta=20,
                 mutation_eta=20, crossover_prob=0.9, seed=None):
        """
        Args:
            problem: Problem instance with evaluate, evaluate_constraints,
                     n_var, n_obj, and bounds.
            pop_size: Population size (N).
            n_gen: Number of generations.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            crossover_prob: Probability of crossover.
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.crossover_prob = crossover_prob

        # Constraint violation cache (indexed the same as the combined pop)
        self._cv = None

    # ------------------------------------------------------------------
    # Constraint-domination helpers
    # ------------------------------------------------------------------

    def _constraint_violation(self, individual):
        """Return total constraint violation (scalar >= 0).

        A value of 0 means the solution is feasible.
        """
        cv = self.problem.evaluate_constraints(individual)
        if cv is None:
            return 0.0
        cv = np.asarray(cv, dtype=float)
        # Constraints are assumed <= 0 (feasible when <= 0)
        return float(np.sum(np.maximum(cv, 0.0)))

    def _compute_cv_population(self, pop):
        """Compute constraint-violation vector for a population."""
        return np.array([self._constraint_violation(ind) for ind in pop])

    def _dominates(self, a, b, cv_a=0.0, cv_b=0.0):
        """Constraint-domination: feasible beats infeasible; among infeasible,
        lower total violation wins; among feasible, Pareto dominance applies."""
        # Both feasible -> normal dominance
        if cv_a == 0.0 and cv_b == 0.0:
            return bool(np.all(a <= b) and np.any(a < b))
        # One feasible, one not
        if cv_a == 0.0 and cv_b > 0.0:
            return True
        if cv_a > 0.0 and cv_b == 0.0:
            return False
        # Both infeasible -> smaller violation dominates
        return cv_a < cv_b

    # ------------------------------------------------------------------
    # Override sorting to use constraint-domination
    # ------------------------------------------------------------------

    def fast_non_dominated_sort(self, objectives, cv=None):
        """Fast non-dominated sorting with optional constraint violation.

        Args:
            objectives: (n, n_obj) objective array.
            cv: Optional (n,) constraint-violation array.

        Returns:
            List of fronts, each a list of indices.
        """
        if cv is None:
            cv = np.zeros(len(objectives))

        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(objectives[i], objectives[j],
                                   cv[i], cv[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i],
                                     cv[j], cv[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return fronts[:-1]  # drop trailing empty front

    # ------------------------------------------------------------------
    # Tournament with constraint-domination
    # ------------------------------------------------------------------

    def _tournament(self, pop, objectives, cv, ranks, crowding):
        """Binary tournament using rank, then crowding distance."""
        n = len(pop)
        i, j = self.rng.integers(0, n, size=2)

        # Constraint-domination at tournament level
        if cv[i] == 0.0 and cv[j] > 0.0:
            return pop[i]
        if cv[j] == 0.0 and cv[i] > 0.0:
            return pop[j]
        if cv[i] > 0.0 and cv[j] > 0.0:
            return pop[i] if cv[i] < cv[j] else pop[j]

        # Both feasible -> rank then crowding
        if ranks[i] < ranks[j]:
            return pop[i]
        elif ranks[j] < ranks[i]:
            return pop[j]
        else:
            return pop[i] if crowding[i] >= crowding[j] else pop[j]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute NSGA-II.

        Returns:
            (pareto_solutions, pareto_objectives): numpy arrays for the
            non-dominated set found after *n_gen* generations.
        """
        # --- Initialization ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)
        cv = self._compute_cv_population(pop)

        for gen in range(self.n_gen):
            t_start = time.time()

            # Build rank / crowding arrays for current pop (needed for mating)
            fronts = self.fast_non_dominated_sort(objectives, cv)
            ranks = np.zeros(len(pop), dtype=int)
            crowding_all = np.zeros(len(pop))
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                if len(front) > 0:
                    cd = self.crowding_distance(objectives[front])
                    for k, idx in enumerate(front):
                        crowding_all[idx] = cd[k]

            # --- Create offspring ---
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, cv, ranks, crowding_all)
                p2 = self._tournament(pop, objectives, cv, ranks, crowding_all)
                c1, c2 = self.sbx_crossover(p1, p2,
                                            eta=self.crossover_eta,
                                            prob=self.crossover_prob)
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])

            # --- Evaluate offspring ---
            off_obj = self.evaluate_population(offspring)
            off_cv = self._compute_cv_population(offspring)

            # --- Combine parent + offspring (2N) ---
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([objectives, off_obj])
            combined_cv = np.concatenate([cv, off_cv])

            # --- Non-dominated sort on combined ---
            fronts = self.fast_non_dominated_sort(combined_obj, combined_cv)

            # --- Select next generation of size N ---
            new_pop = []
            new_obj = []
            new_cv = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_cv.append(combined_cv[idx])
                else:
                    # Need partial front — sort by crowding distance
                    remaining = self.pop_size - len(new_pop)
                    cd = self.crowding_distance(combined_obj[front])
                    sorted_indices = np.argsort(-cd)  # descending
                    for k in sorted_indices[:remaining]:
                        idx = front[k]
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_cv.append(combined_cv[idx])
                    break

            pop = np.array(new_pop)
            objectives = np.array(new_obj)
            cv = np.array(new_cv)

            # --- History tracking ---
            pf_fronts = self.fast_non_dominated_sort(objectives, cv)
            if pf_fronts:
                pf_obj = objectives[pf_fronts[0]]
            else:
                pf_obj = objectives
            self.history['objectives'].append(pf_obj.copy())

            feasible_mask = cv == 0.0
            self.history['feasibility_rate'].append(
                float(np.mean(feasible_mask)) if len(cv) > 0 else 0.0
            )

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                n_pf = len(pf_obj)
                feas = self.history['feasibility_rate'][-1]
                print(f"NSGA-II  gen {gen + 1:>4d}/{self.n_gen}  |  "
                      f"front size: {n_pf:>4d}  |  "
                      f"feasibility: {feas:.2%}  |  "
                      f"time: {elapsed:.3f}s")

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
