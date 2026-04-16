"""NSGA-III: Reference-Point Based Non-dominated Sorting Approach.

Reference:
    K. Deb and H. Jain,
    "An Evolutionary Many-Objective Optimization Algorithm Using
    Reference-Point-Based Nondominated Sorting Approach, Part I:
    Solving Problems With Box Constraints,"
    IEEE Transactions on Evolutionary Computation, vol. 18, no. 4,
    pp. 577-601, 2014.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from .base import MOEABase


class NSGA3(MOEABase):
    """NSGA-III with Das-Dennis reference points for many-objective optimisation.

    Unlike NSGA-II which relies on crowding distance, NSGA-III maintains
    diversity by associating solutions with a structured set of reference
    points distributed on the normalised objective hyperplane.  This makes
    it better suited for problems with three or more objectives.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        n_partitions: int = 12,
        crossover_eta: float = 20,
        mutation_eta: float = 20,
        seed: int | None = None,
    ):
        """
        Args:
            problem: Problem instance with evaluate, n_var, n_obj, bounds.
            pop_size: Population size (N).
            n_gen: Number of generations.
            n_partitions: Number of divisions on each objective axis for
                Das-Dennis reference point generation.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.n_partitions = n_partitions
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

        self.reference_points = self._generate_reference_points(
            n_partitions, self.n_obj
        )

    # ------------------------------------------------------------------
    # Reference-point generation (Das-Dennis)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_reference_points(n_partitions: int, n_obj: int) -> np.ndarray:
        """Generate structured reference points using the Das-Dennis method.

        Creates uniformly spaced points on the unit simplex in *n_obj*
        dimensions.  The total number of points is C(n_partitions + n_obj - 1,
        n_obj - 1).

        Args:
            n_partitions: Number of equal divisions along each axis.
            n_obj: Number of objectives.

        Returns:
            (n_ref, n_obj) array of reference points on the unit simplex.
        """
        def _recurse(n_part, n_dim, point, depth, result):
            if depth == n_dim - 1:
                point[depth] = n_part
                result.append(point.copy())
                return
            for i in range(n_part + 1):
                point[depth] = i
                _recurse(n_part - i, n_dim, point, depth + 1, result)

        points: list[np.ndarray] = []
        _recurse(n_partitions, n_obj, np.zeros(n_obj), 0, points)
        ref = np.array(points) / n_partitions
        return ref

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_objectives(
        objectives: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
    ) -> np.ndarray:
        """Translate and scale objectives to [0, 1] using ideal / nadir.

        Args:
            objectives: (n, n_obj) raw objective values.
            ideal: (n_obj,) ideal point (minimum per objective).
            nadir: (n_obj,) nadir point (estimated maximum per objective).

        Returns:
            (n, n_obj) normalised objectives.
        """
        denom = nadir - ideal
        denom[denom < 1e-14] = 1e-14  # avoid division by zero
        return (objectives - ideal) / denom

    # ------------------------------------------------------------------
    # Association to reference points
    # ------------------------------------------------------------------

    @staticmethod
    def _associate_to_reference_points(
        normalized_obj: np.ndarray,
        reference_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Associate each solution to its closest reference line.

        For every solution the perpendicular distance to each reference line
        (from origin through each reference point) is computed; the solution
        is assigned to the nearest one.

        Args:
            normalized_obj: (n, n_obj) normalised objective values.
            reference_points: (n_ref, n_obj) reference directions.

        Returns:
            closest_ref: (n,) index of closest reference point per solution.
            distances: (n,) perpendicular distance to closest reference line.
        """
        # Unit direction vectors for each reference line
        norms = np.linalg.norm(reference_points, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1e-14
        unit_dirs = reference_points / norms

        # Projection length of each solution onto each reference line
        # proj_len shape: (n, n_ref)
        proj_len = normalized_obj @ unit_dirs.T

        # Projection vectors: (n, n_ref, n_obj)
        proj_vec = proj_len[:, :, np.newaxis] * unit_dirs[np.newaxis, :, :]

        # Perpendicular distance
        diff = normalized_obj[:, np.newaxis, :] - proj_vec
        perp_dist = np.linalg.norm(diff, axis=2)  # (n, n_ref)

        closest_ref = np.argmin(perp_dist, axis=1)
        distances = perp_dist[np.arange(len(normalized_obj)), closest_ref]
        return closest_ref, distances

    # ------------------------------------------------------------------
    # Niching-based selection
    # ------------------------------------------------------------------

    def _niching_selection(
        self,
        fronts: list[list[int]],
        objectives: np.ndarray,
        reference_points: np.ndarray,
        n_select: int,
    ) -> list[int]:
        """Select *n_select* solutions from the critical front using niching.

        All solutions from fronts before the critical front are already
        selected.  From the critical front, solutions are chosen one at a
        time, always preferring those associated with the least-populated
        reference point.  Ties are broken by perpendicular distance.

        Args:
            fronts: Non-dominated fronts (lists of indices into *objectives*).
            objectives: (n, n_obj) objective values for the combined pop.
            reference_points: (n_ref, n_obj) reference directions.
            n_select: Total number of solutions to retain.

        Returns:
            List of selected indices into *objectives*.
        """
        # Determine ideal and nadir from the union of accepted + critical
        all_indices = [idx for front in fronts for idx in front]
        ideal = objectives[all_indices].min(axis=0)
        nadir = objectives[all_indices].max(axis=0)
        normalized = self._normalize_objectives(objectives, ideal, nadir)

        closest_ref, distances = self._associate_to_reference_points(
            normalized, reference_points
        )

        # Separate already-selected (prior fronts) from critical front
        selected: list[int] = []
        for front in fronts[:-1]:
            selected.extend(front)
            if len(selected) >= n_select:
                return selected[:n_select]

        critical_front = fronts[-1]

        # Niche count for already-selected members
        n_ref = len(reference_points)
        niche_count = np.zeros(n_ref, dtype=int)
        for idx in selected:
            niche_count[closest_ref[idx]] += 1

        remaining = set(critical_front)

        while len(selected) < n_select and remaining:
            # Reference points with minimum niche count
            min_count = niche_count.min()
            candidate_refs = np.where(niche_count == min_count)[0]

            # Pick a random reference point among those with the lowest count
            ref_idx = self.rng.choice(candidate_refs)

            # Solutions in the critical front associated with this ref point
            assoc = [idx for idx in remaining if closest_ref[idx] == ref_idx]

            if not assoc:
                # No solution associated — exclude this ref point temporarily
                niche_count[ref_idx] = n_select + 1  # large sentinel
                continue

            if niche_count[ref_idx] == 0:
                # Prefer the closest solution
                best = min(assoc, key=lambda idx: distances[idx])
            else:
                best = self.rng.choice(assoc)

            selected.append(best)
            remaining.discard(best)
            niche_count[ref_idx] += 1

        return selected

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute NSGA-III.

        Returns:
            (pareto_solutions, pareto_objectives): numpy arrays for the
            non-dominated set found after *n_gen* generations.
        """
        # --- Initialisation ---
        pop = self.initialize_population()
        objectives = self.evaluate_population(pop)

        for gen in range(self.n_gen):
            t_start = time.time()

            # --- Create offspring via SBX + polynomial mutation ---
            offspring: list[np.ndarray] = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection(pop, objectives)
                p2 = self.tournament_selection(pop, objectives)
                c1, c2 = self.sbx_crossover(
                    p1, p2, eta=self.crossover_eta
                )
                c1 = self.polynomial_mutation(c1, eta=self.mutation_eta)
                c2 = self.polynomial_mutation(c2, eta=self.mutation_eta)
                offspring.extend([c1, c2])
            offspring_arr = np.array(offspring[: self.pop_size])

            off_obj = self.evaluate_population(offspring_arr)

            # --- Combine parent + offspring (2N) ---
            combined_pop = np.vstack([pop, offspring_arr])
            combined_obj = np.vstack([objectives, off_obj])

            # --- Non-dominated sort ---
            fronts = self.fast_non_dominated_sort(combined_obj)

            # --- Fill new population front by front ---
            included_fronts: list[list[int]] = []
            count = 0
            for front in fronts:
                if count + len(front) <= self.pop_size:
                    included_fronts.append(front)
                    count += len(front)
                else:
                    # This is the critical front — use niching
                    included_fronts.append(front)
                    break

            if count < self.pop_size:
                # Niching selection on included fronts
                selected_idx = self._niching_selection(
                    included_fronts,
                    combined_obj,
                    self.reference_points,
                    self.pop_size,
                )
            else:
                selected_idx = [
                    idx for front in included_fronts for idx in front
                ]

            pop = combined_pop[selected_idx]
            objectives = combined_obj[selected_idx]

            # --- History tracking ---
            pf_fronts = self.fast_non_dominated_sort(objectives)
            pf_obj = objectives[pf_fronts[0]] if pf_fronts else objectives
            self.history['objectives'].append(pf_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                n_pf = len(pf_obj)
                print(
                    f"NSGA-III gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"ref points: {len(self.reference_points):>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
