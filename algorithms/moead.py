"""MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition.

Reference:
    Q. Zhang and H. Li,
    "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
    IEEE Transactions on Evolutionary Computation, 2007.

Uses the Tchebycheff scalarisation approach with neighbourhood-based mating
and replacement.
"""
import numpy as np
import time
from itertools import combinations_with_replacement

from .base import MOEABase


class MOEAD(MOEABase):
    """MOEA/D with Tchebycheff decomposition.

    Decomposes a multi-objective problem into a set of scalar subproblems
    defined by uniformly distributed weight vectors, and solves them
    simultaneously using neighbourhood information.
    """

    def __init__(self, problem, pop_size=100, n_gen=200, n_neighbors=20,
                 crossover_eta=20, mutation_eta=20, seed=None):
        """
        Args:
            problem: Problem instance.
            pop_size: Desired population size (may be adjusted to match the
                      number of weight vectors actually generated).
            n_gen: Number of generations.
            n_neighbors: Neighbourhood size T.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            seed: Random seed for reproducibility.
        """
        # We might adjust pop_size below, so call super first with the
        # requested value, then overwrite.
        super().__init__(problem, pop_size, n_gen, seed)
        self.n_neighbors = n_neighbors
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

        # Weight vectors, neighbourhood, ideal point — initialised in run()
        self.weights = None
        self.neighborhoods = None
        self.z_ideal = None

    # ------------------------------------------------------------------
    # Weight vector generation (simplex-lattice design)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_weight_vectors(pop_size, n_obj):
        """Generate uniformly distributed weight vectors on the unit simplex.

        Uses the simplex-lattice design.  The granularity *H* is chosen so
        that C(H + n_obj - 1, n_obj - 1) is as close to *pop_size* as
        possible.  The actual number of vectors is returned along with the
        vectors themselves.

        Args:
            pop_size: Target number of vectors.
            n_obj: Number of objectives.

        Returns:
            weights: (N, n_obj) numpy array, rows sum to 1.
            actual_size: Number of vectors generated.
        """
        from math import comb

        # Find H such that C(H+m-1, m-1) ~ pop_size
        H = 1
        while comb(H + n_obj - 1, n_obj - 1) < pop_size:
            H += 1
        # H is now the smallest value giving >= pop_size vectors.
        # Check if H-1 is closer.
        n_upper = comb(H + n_obj - 1, n_obj - 1)
        n_lower = comb(H - 1 + n_obj - 1, n_obj - 1) if H > 1 else 0
        if abs(n_lower - pop_size) < abs(n_upper - pop_size) and n_lower > 0:
            H = H - 1

        # Enumerate all compositions of H into n_obj parts
        vectors = []
        for combo in combinations_with_replacement(range(H + 1), n_obj - 1):
            # combo gives n_obj-1 dividers in [0, H]
            parts = [combo[0]]
            for k in range(1, len(combo)):
                parts.append(combo[k] - combo[k - 1])
            parts.append(H - combo[-1])
            vectors.append([p / H for p in parts])

        weights = np.array(vectors)
        return weights, len(weights)

    # ------------------------------------------------------------------
    # Neighbourhood computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_neighborhoods(weights, n_neighbors):
        """For each weight vector find its *n_neighbors* nearest neighbours.

        Args:
            weights: (N, n_obj) weight matrix.
            n_neighbors: Number of neighbours T.

        Returns:
            neighborhoods: (N, T) integer index array.
        """
        n = len(weights)
        n_neighbors = min(n_neighbors, n)
        # Pairwise Euclidean distances
        dists = np.linalg.norm(weights[:, None, :] - weights[None, :, :],
                               axis=2)
        neighborhoods = np.argsort(dists, axis=1)[:, :n_neighbors]
        return neighborhoods

    # ------------------------------------------------------------------
    # Tchebycheff scalarisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tchebycheff(x_obj, weight, z_ideal, eps=1e-6):
        """Tchebycheff scalarisation value.

        g^te(x | w, z*) = max_i { (w_i + eps) * |f_i(x) - z*_i| }

        Args:
            x_obj: Objective vector of the solution.
            weight: Weight vector for the sub-problem.
            z_ideal: Current ideal point.
            eps: Small constant to avoid zero weights.

        Returns:
            Scalar Tchebycheff value.
        """
        return float(np.max((weight + eps) * np.abs(x_obj - z_ideal)))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute MOEA/D.

        Returns:
            (pareto_solutions, pareto_objectives): numpy arrays for the
            non-dominated solutions found.
        """
        # --- Weight vectors & neighbourhoods ---
        self.weights, actual_size = self._generate_weight_vectors(
            self.pop_size, self.n_obj)
        self.pop_size = actual_size
        self.neighborhoods = self._compute_neighborhoods(
            self.weights, self.n_neighbors)

        print(f"MOEA/D  weight vectors generated: {actual_size}  "
              f"(requested {actual_size}),  "
              f"neighbourhood size: {self.n_neighbors}")

        # --- Initialise population (one solution per weight vector) ---
        pop = self.rng.uniform(self.lower, self.upper,
                               size=(self.pop_size, self.n_var))
        objectives = self.evaluate_population(pop)

        # --- Ideal point ---
        self.z_ideal = np.min(objectives, axis=0).copy()

        for gen in range(self.n_gen):
            t_start = time.time()

            for i in range(self.pop_size):
                # Select two parents from neighbourhood of i
                nb = self.neighborhoods[i]
                p_idx = self.rng.choice(nb, size=2, replace=False)
                parent1 = pop[p_idx[0]]
                parent2 = pop[p_idx[1]]

                # Reproduce
                child, _ = self.sbx_crossover(parent1, parent2,
                                              eta=self.crossover_eta)
                child = self.polynomial_mutation(child,
                                                 eta=self.mutation_eta)

                # Evaluate child
                child_obj = np.asarray(self.problem.evaluate(child))

                # Update ideal point
                self.z_ideal = np.minimum(self.z_ideal, child_obj)

                # Update neighbours
                for j in nb:
                    g_child = self._tchebycheff(child_obj, self.weights[j],
                                                self.z_ideal)
                    g_current = self._tchebycheff(objectives[j],
                                                  self.weights[j],
                                                  self.z_ideal)
                    if g_child < g_current:
                        pop[j] = child.copy()
                        objectives[j] = child_obj.copy()

            # --- History tracking ---
            pf_pop, pf_obj = self.get_pareto_front(pop, objectives)
            self.history['objectives'].append(pf_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"MOEA/D   gen {gen + 1:>4d}/{self.n_gen}  |  "
                      f"front size: {len(pf_obj):>4d}  |  "
                      f"ideal: {np.array2string(self.z_ideal, precision=3)}  |  "
                      f"time: {elapsed:.3f}s")

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
