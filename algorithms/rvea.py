"""RVEA: Reference Vector Guided Evolutionary Algorithm.

Reference:
    R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff,
    "A Reference Vector Guided Evolutionary Algorithm for Many-Objective
    Optimization,"
    IEEE Transactions on Evolutionary Computation, vol. 20, no. 5,
    pp. 773-791, 2016.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from .base import MOEABase


class RVEA(MOEABase):
    """Reference Vector Guided Evolutionary Algorithm for many-objective optimisation.

    RVEA decomposes a many-objective problem using a set of uniformly
    distributed reference vectors on the unit simplex.  Selection is driven
    by the Angle-Penalised Distance (APD), which balances convergence and
    diversity in a generation-dependent manner.  Reference vectors are
    periodically adapted to the current population's distribution so that
    computational effort focuses on the occupied region of objective space.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        alpha: float = 2.0,
        fr: float = 0.1,
        crossover_eta: float = 20,
        mutation_eta: float = 20,
        seed: int | None = None,
    ):
        """
        Args:
            problem: Problem instance with evaluate, n_var, n_obj, bounds.
            pop_size: Population size (N).
            n_gen: Number of generations.
            alpha: Controls the rate at which the angle penalty increases
                over generations.  Larger values delay the diversity
                emphasis.
            fr: Fraction of total generations that defines the reference
                vector adaptation frequency.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.alpha = alpha
        self.fr = fr
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

        # Determine number of partitions to produce ~pop_size reference vectors
        n_partitions = self._auto_partitions(self.n_obj, pop_size)
        self.reference_vectors_initial = self._generate_reference_vectors(
            n_partitions, self.n_obj
        )
        self.reference_vectors = self.reference_vectors_initial.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_partitions(n_obj: int, pop_size: int) -> int:
        """Choose *n_partitions* so that the Das-Dennis point count is
        close to *pop_size*.  C(p + n_obj - 1, n_obj - 1) ≈ pop_size."""
        from math import comb

        p = 1
        while comb(p + n_obj - 1, n_obj - 1) < pop_size:
            p += 1
        return p

    # ------------------------------------------------------------------
    # Reference vector generation (Das-Dennis + unit normalisation)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_reference_vectors(
        n_partitions: int, n_obj: int
    ) -> np.ndarray:
        """Generate uniform reference vectors on the unit simplex, then
        normalise each to unit length.

        Uses the Das-Dennis systematic approach to create points on the
        (n_obj - 1)-dimensional simplex and converts them to unit vectors
        in objective space.

        Args:
            n_partitions: Number of equal divisions along each axis.
            n_obj: Number of objectives.

        Returns:
            (n_ref, n_obj) array of unit reference vectors.
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
        vectors = np.array(points, dtype=float) / n_partitions

        # Shift zero components slightly so normalisation is well-defined
        vectors[vectors == 0.0] = 1e-6

        # Normalise to unit length
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors

    # ------------------------------------------------------------------
    # Angle-Penalised Distance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_apd(
        objectives: np.ndarray,
        ref_vectors: np.ndarray,
        gen: int,
        n_gen: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Angle-Penalised Distance for every solution.

        Each solution is first associated with the reference vector to which
        it has the smallest angle.  The APD is then:

            APD_i = (1 + P(θ_i)) · ‖f'_i‖

        where *f'* is the translated (ideal-point subtracted) objective
        vector, *θ_i* is the angle between *f'_i* and its associated
        reference vector, and

            P(θ) = n_obj · (gen / n_gen)^alpha · (θ / γ)

        with *γ* being the minimum acute angle between any pair of adjacent
        reference vectors (precomputed once).

        Args:
            objectives: (n, n_obj) translated (ideal-subtracted) objective
                values.
            ref_vectors: (n_ref, n_obj) unit reference vectors.
            gen: Current generation (0-indexed).
            n_gen: Total number of generations.
            alpha: Penalty growth rate parameter.

        Returns:
            apd: (n,) angle-penalised distance for each solution.
            assoc: (n,) index of associated reference vector per solution.
        """
        n, n_obj = objectives.shape
        n_ref = len(ref_vectors)

        # -- Normalise objective vectors to unit length --
        obj_norms = np.linalg.norm(objectives, axis=1, keepdims=True)
        obj_norms[obj_norms < 1e-14] = 1e-14
        obj_unit = objectives / obj_norms

        # -- Cosine of angles between each solution and each ref vector --
        # Shape: (n, n_ref)
        cosines = obj_unit @ ref_vectors.T
        cosines = np.clip(cosines, -1.0, 1.0)
        angles = np.arccos(cosines)  # (n, n_ref)

        # -- Associate each solution with the closest reference vector --
        assoc = np.argmin(angles, axis=1)  # (n,)
        theta = angles[np.arange(n), assoc]  # (n,)

        # -- Minimum angle between any two reference vectors (γ) --
        ref_cosines = ref_vectors @ ref_vectors.T
        np.fill_diagonal(ref_cosines, -1.0)  # ignore self
        ref_cosines = np.clip(ref_cosines, -1.0, 1.0)
        gamma = np.arccos(ref_cosines.max())  # smallest angle

        if gamma < 1e-14:
            gamma = 1e-14

        # -- Penalty function --
        penalty = n_obj * ((gen + 1) / n_gen) ** alpha * (theta / gamma)

        # -- APD --
        apd = (1.0 + penalty) * obj_norms.ravel()

        return apd, assoc

    # ------------------------------------------------------------------
    # Reference vector adaptation
    # ------------------------------------------------------------------

    def _reference_vector_adaptation(
        self,
        ref_vectors: np.ndarray,
        objectives: np.ndarray,
        gen: int,
        n_gen: int,
        fr: float,
    ) -> np.ndarray:
        """Periodically adapt reference vectors toward active population
        directions.

        At every *fr × n_gen* generations, reference vectors whose niches
        are empty are replaced by re-scaled versions of the initial
        (uniform) vectors rotated toward the currently occupied directions.

        Only performs adaptation when the generation number is an integer
        multiple of *fr × n_gen* (and > 0).

        Args:
            ref_vectors: (n_ref, n_obj) current unit reference vectors.
            objectives: (n, n_obj) current population's translated objectives.
            gen: Current generation (0-indexed).
            n_gen: Total number of generations.
            fr: Adaptation frequency as a fraction of n_gen.

        Returns:
            (n_ref, n_obj) adapted unit reference vectors.
        """
        adapt_interval = max(1, int(np.ceil(fr * n_gen)))
        if (gen + 1) % adapt_interval != 0 or gen == 0:
            return ref_vectors

        # Normalise objectives to unit vectors
        obj_norms = np.linalg.norm(objectives, axis=1, keepdims=True)
        obj_norms[obj_norms < 1e-14] = 1e-14
        obj_unit = objectives / obj_norms

        # Compute the range of each objective component across the population
        obj_max = objectives.max(axis=0)
        obj_min = objectives.min(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-14] = 1e-14

        # Scale the initial reference vectors by the objective range
        adapted = self.reference_vectors_initial * obj_range[np.newaxis, :]

        # Re-normalise to unit length
        norms = np.linalg.norm(adapted, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1e-14
        adapted = adapted / norms

        return adapted

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute RVEA.

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

            # --- Combine parent + offspring ---
            combined_pop = np.vstack([pop, offspring_arr])
            combined_obj = np.vstack([objectives, off_obj])

            # --- Translate objectives (subtract ideal point) ---
            ideal = combined_obj.min(axis=0)
            translated = combined_obj - ideal

            # --- Compute APD and associate to reference vectors ---
            apd, assoc = self._compute_apd(
                translated, self.reference_vectors,
                gen, self.n_gen, self.alpha,
            )

            # --- Elitist selection: best APD per reference vector niche ---
            n_ref = len(self.reference_vectors)
            selected_idx: list[int] = []

            for rv in range(n_ref):
                niche_mask = np.where(assoc == rv)[0]
                if len(niche_mask) == 0:
                    continue
                best_in_niche = niche_mask[np.argmin(apd[niche_mask])]
                selected_idx.append(int(best_in_niche))

            # If fewer than pop_size selected, fill with best remaining APD
            if len(selected_idx) < self.pop_size:
                remaining = set(range(len(combined_pop))) - set(selected_idx)
                remaining_list = sorted(remaining, key=lambda i: apd[i])
                needed = self.pop_size - len(selected_idx)
                selected_idx.extend(remaining_list[:needed])
            elif len(selected_idx) > self.pop_size:
                # Keep the pop_size with smallest APD among those selected
                selected_idx.sort(key=lambda i: apd[i])
                selected_idx = selected_idx[: self.pop_size]

            pop = combined_pop[selected_idx]
            objectives = combined_obj[selected_idx]

            # --- Reference vector adaptation ---
            translated_sel = objectives - objectives.min(axis=0)
            self.reference_vectors = self._reference_vector_adaptation(
                self.reference_vectors, translated_sel,
                gen, self.n_gen, self.fr,
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
                    f"RVEA     gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"front size: {n_pf:>4d}  |  "
                    f"ref vectors: {len(self.reference_vectors):>4d}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return Pareto front ---
        return self.get_pareto_front(pop, objectives)
