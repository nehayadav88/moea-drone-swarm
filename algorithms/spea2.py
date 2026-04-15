"""SPEA2: Strength Pareto Evolutionary Algorithm 2.

Reference:
    E. Zitzler, M. Laumanns, and L. Thiele,
    "SPEA2: Improving the Strength Pareto Evolutionary Algorithm,"
    TIK-Report 103, ETH Zurich, 2001.
"""
import numpy as np
import time

from .base import MOEABase


class SPEA2(MOEABase):
    """SPEA2 with fine-grained fitness and archive truncation.

    Maintains an external archive and uses a fitness assignment scheme
    based on strength values and k-th nearest neighbour density estimation
    to guide the search toward a well-distributed Pareto front.
    """

    def __init__(self, problem, pop_size=100, archive_size=100, n_gen=200,
                 crossover_eta=20, mutation_eta=20, seed=None):
        """
        Args:
            problem: Problem instance.
            pop_size: Population size.
            archive_size: Maximum archive size.
            n_gen: Number of generations.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.archive_size = archive_size
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

    # ------------------------------------------------------------------
    # Fitness components
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_raw_fitness(objectives):
        """Compute raw fitness R(i) for each individual.

        S(i) = number of solutions that i dominates.
        R(i) = sum of S(j) for all j that dominate i.

        Args:
            objectives: (n, n_obj) array.

        Returns:
            raw: (n,) array of raw fitness values.
        """
        n = len(objectives)
        strength = np.zeros(n, dtype=int)
        raw = np.zeros(n, dtype=float)

        # Precompute domination relationships
        # dom[i][j] = True if i dominates j
        dom = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(objectives[i] <= objectives[j]) and \
                   np.any(objectives[i] < objectives[j]):
                    dom[i, j] = True

        # Strength: how many does i dominate?
        strength = dom.sum(axis=1)  # S(i) = |{j : i dominates j}|

        # Raw fitness: sum of strengths of all dominators
        for i in range(n):
            dominators = np.where(dom[:, i])[0]  # j that dominate i
            raw[i] = float(np.sum(strength[dominators]))

        return raw

    @staticmethod
    def _compute_density(objectives, k=None):
        """k-th nearest neighbour density estimation.

        D(i) = 1 / (sigma_k(i) + 2)

        where sigma_k(i) is the Euclidean distance to the k-th nearest
        neighbour in objective space.

        Args:
            objectives: (n, n_obj) array.
            k: Neighbour index. Defaults to floor(sqrt(n)).

        Returns:
            density: (n,) array.
        """
        n = len(objectives)
        if k is None:
            k = int(np.floor(np.sqrt(n)))
        k = max(1, min(k, n - 1))

        # Pairwise distances
        dists = np.linalg.norm(
            objectives[:, None, :] - objectives[None, :, :], axis=2
        )
        # Sort distances; column 0 is self (distance 0)
        sorted_dists = np.sort(dists, axis=1)
        # col 0 = self (distance 0), so col k = k-th nearest neighbour
        sigma_k = sorted_dists[:, k]

        density = 1.0 / (sigma_k + 2.0)
        return density

    @staticmethod
    def _compute_fitness(objectives):
        """Total SPEA2 fitness F(i) = R(i) + D(i).

        Lower is better.  Non-dominated individuals have F < 1.

        Args:
            objectives: (n, n_obj) array.

        Returns:
            fitness: (n,) array.
        """
        raw = SPEA2._compute_raw_fitness(objectives)
        density = SPEA2._compute_density(objectives)
        return raw + density

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    @staticmethod
    def _environmental_selection(pop, objectives, fitness, archive_size):
        """Select next archive from combined population + archive.

        1. Copy all non-dominated (fitness < 1) to new archive.
        2. If archive too small, fill with best dominated individuals.
        3. If archive too large, iteratively remove the individual with the
           smallest distance to its nearest neighbour (truncation procedure).

        Args:
            pop: (n, n_var) decision variables.
            objectives: (n, n_obj) objective values.
            fitness: (n,) SPEA2 fitness values.
            archive_size: Maximum archive size.

        Returns:
            (archive_pop, archive_obj): selected individuals and objectives.
        """
        n = len(pop)
        non_dom_mask = fitness < 1.0
        non_dom_idx = np.where(non_dom_mask)[0]
        dom_idx = np.where(~non_dom_mask)[0]

        if len(non_dom_idx) == archive_size:
            sel = non_dom_idx
        elif len(non_dom_idx) < archive_size:
            # Fill remaining slots from dominated, sorted by fitness
            remaining = archive_size - len(non_dom_idx)
            sorted_dom = dom_idx[np.argsort(fitness[dom_idx])]
            extra = sorted_dom[:remaining]
            sel = np.concatenate([non_dom_idx, extra])
        else:
            # Truncation: iteratively remove closest-pair member
            sel = non_dom_idx.tolist()
            obj_sel = objectives[sel]

            while len(sel) > archive_size:
                m = len(sel)
                dists = np.linalg.norm(
                    obj_sel[:, None, :] - obj_sel[None, :, :], axis=2
                )
                np.fill_diagonal(dists, np.inf)

                # For each individual, compute sorted distance list
                sorted_dists = np.sort(dists, axis=1)

                # Find the individual with smallest distance sequence
                # (lexicographic comparison on sorted distance vectors)
                remove_idx = 0
                for i in range(1, m):
                    for d in range(m - 1):
                        if sorted_dists[i, d] < sorted_dists[remove_idx, d]:
                            remove_idx = i
                            break
                        elif sorted_dists[i, d] > sorted_dists[remove_idx, d]:
                            break

                sel.pop(remove_idx)
                obj_sel = np.delete(obj_sel, remove_idx, axis=0)

            sel = np.array(sel)

        return pop[sel], objectives[sel]

    # ------------------------------------------------------------------
    # Mating selection
    # ------------------------------------------------------------------

    def _mating_tournament(self, archive_pop, archive_fitness):
        """Binary tournament selection from archive (lower fitness = better).

        Args:
            archive_pop: (m, n_var) archive individuals.
            archive_fitness: (m,) fitness values.

        Returns:
            Selected individual (1-D array).
        """
        m = len(archive_pop)
        i, j = self.rng.integers(0, m, size=2)
        return archive_pop[i] if archive_fitness[i] <= archive_fitness[j] \
            else archive_pop[j]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute SPEA2.

        Returns:
            (pareto_solutions, pareto_objectives): non-dominated solutions
            from the final archive.
        """
        # --- Initialise population ---
        pop = self.initialize_population()
        pop_obj = self.evaluate_population(pop)

        # Empty archive
        archive_pop = np.empty((0, self.n_var))
        archive_obj = np.empty((0, self.n_obj))

        for gen in range(self.n_gen):
            t_start = time.time()

            # --- Combine population and archive ---
            if len(archive_pop) > 0:
                combined_pop = np.vstack([pop, archive_pop])
                combined_obj = np.vstack([pop_obj, archive_obj])
            else:
                combined_pop = pop.copy()
                combined_obj = pop_obj.copy()

            # --- Fitness assignment ---
            fitness = self._compute_fitness(combined_obj)

            # --- Environmental selection -> new archive ---
            archive_pop, archive_obj = self._environmental_selection(
                combined_pop, combined_obj, fitness, self.archive_size
            )

            # Recompute fitness for archive (needed for mating selection)
            archive_fitness = self._compute_fitness(archive_obj)

            # --- History tracking ---
            pf_pop, pf_obj = self.get_pareto_front(archive_pop, archive_obj)
            self.history['objectives'].append(pf_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"SPEA2    gen {gen + 1:>4d}/{self.n_gen}  |  "
                      f"archive: {len(archive_pop):>4d}  |  "
                      f"front size: {len(pf_obj):>4d}  |  "
                      f"time: {elapsed:.3f}s")

            # Last generation -> return archive's Pareto front
            if gen == self.n_gen - 1:
                break

            # --- Mating selection + offspring creation ---
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._mating_tournament(archive_pop, archive_fitness)
                p2 = self._mating_tournament(archive_pop, archive_fitness)
                c1, c2 = self.sbx_crossover(p1, p2,
                                            eta=self.crossover_eta)
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])

            pop = np.array(offspring[:self.pop_size])
            pop_obj = self.evaluate_population(pop)

        return self.get_pareto_front(archive_pop, archive_obj)
