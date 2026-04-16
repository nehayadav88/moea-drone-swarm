"""MOPSO: Multi-Objective Particle Swarm Optimization.

Reference:
    C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga,
    "Handling Multiple Objectives with Particle Swarm Optimization,"
    IEEE Transactions on Evolutionary Computation, vol. 8, no. 3,
    pp. 256-279, 2004.
"""
from __future__ import annotations

import time

import numpy as np

from .base import MOEABase


class MOPSO(MOEABase):
    """Multi-Objective Particle Swarm Optimization with external archive.

    Maintains an external archive of non-dominated solutions with
    grid-based density estimation.  Leaders are selected from the archive
    using roulette-wheel selection on inverse grid density to encourage
    diversity.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        archive_size: int = 100,
        w: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        seed: int | None = None,
    ):
        """
        Args:
            problem: Problem instance with evaluate, n_var, n_obj, bounds.
            pop_size: Number of particles in the swarm.
            n_gen: Number of generations (iterations).
            archive_size: Maximum size of the external archive.
            w: Inertia weight for velocity update.
            c1: Cognitive acceleration coefficient (personal best).
            c2: Social acceleration coefficient (archive leader).
            seed: Random seed for reproducibility.
        """
        super().__init__(problem, pop_size, n_gen, seed)
        self.archive_size = archive_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self._n_grid_divs = 10  # grid divisions per objective for density

    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------

    def _update_archive(
        self,
        archive_pos: np.ndarray,
        archive_obj: np.ndarray,
        new_pos: np.ndarray,
        new_obj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge new solutions into the archive keeping only non-dominated.

        If the resulting archive exceeds *archive_size*, the most crowded
        members (highest grid density) are pruned.

        Args:
            archive_pos: (a, n_var) current archive positions.
            archive_obj: (a, n_obj) current archive objectives.
            new_pos: (n, n_var) candidate positions.
            new_obj: (n, n_obj) candidate objectives.

        Returns:
            Updated (archive_pos, archive_obj).
        """
        if len(archive_pos) == 0:
            combined_pos = new_pos.copy()
            combined_obj = new_obj.copy()
        else:
            combined_pos = np.vstack([archive_pos, new_pos])
            combined_obj = np.vstack([archive_obj, new_obj])

        # Keep only non-dominated solutions
        nd_mask = self._non_dominated_mask(combined_obj)
        combined_pos = combined_pos[nd_mask]
        combined_obj = combined_obj[nd_mask]

        # Prune by grid density if over capacity
        if len(combined_pos) > self.archive_size:
            density = self._grid_density(combined_obj)
            # Iteratively remove the member with highest density
            while len(combined_pos) > self.archive_size:
                worst = np.argmax(density)
                keep = np.ones(len(combined_pos), dtype=bool)
                keep[worst] = False
                combined_pos = combined_pos[keep]
                combined_obj = combined_obj[keep]
                density = self._grid_density(combined_obj)

        return combined_pos, combined_obj

    def _non_dominated_mask(self, objectives: np.ndarray) -> np.ndarray:
        """Return a boolean mask of non-dominated solutions.

        Args:
            objectives: (n, n_obj) objective values.

        Returns:
            (n,) boolean array — True for non-dominated rows.
        """
        n = len(objectives)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(i + 1, n):
                if not mask[j]:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    mask[j] = False
                elif self._dominates(objectives[j], objectives[i]):
                    mask[i] = False
                    break
        return mask

    # ------------------------------------------------------------------
    # Grid-based density estimation
    # ------------------------------------------------------------------

    def _grid_density(self, objectives: np.ndarray) -> np.ndarray:
        """Compute grid-based density for each solution in the archive.

        The objective space is partitioned into a regular grid and each
        solution's density equals the number of archive members in its
        grid cell.

        Args:
            objectives: (n, n_obj) objective values.

        Returns:
            (n,) density counts.
        """
        n = len(objectives)
        if n == 0:
            return np.array([])

        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-14] = 1e-14

        # Map each solution to a grid cell index per objective
        normalised = (objectives - obj_min) / obj_range
        cell_idx = np.clip(
            (normalised * self._n_grid_divs).astype(int),
            0,
            self._n_grid_divs - 1,
        )

        # Convert multi-dimensional cell index to a single hash
        multipliers = self._n_grid_divs ** np.arange(self.n_obj)
        cell_hash = cell_idx @ multipliers

        # Count members per cell
        unique, inverse, counts = np.unique(
            cell_hash, return_inverse=True, return_counts=True
        )
        density = counts[inverse]
        return density

    # ------------------------------------------------------------------
    # Leader selection
    # ------------------------------------------------------------------

    def _select_leader(
        self,
        archive_pos: np.ndarray,
        archive_obj: np.ndarray,
    ) -> np.ndarray:
        """Select a leader from the archive using roulette on inverse density.

        Solutions in less-crowded grid cells have higher selection
        probability, promoting diversity.

        Args:
            archive_pos: (a, n_var) archive positions.
            archive_obj: (a, n_obj) archive objectives.

        Returns:
            (n_var,) position of the selected leader.
        """
        density = self._grid_density(archive_obj)
        inv_density = 1.0 / (density + 1e-14)
        probs = inv_density / inv_density.sum()
        idx = self.rng.choice(len(archive_pos), p=probs)
        return archive_pos[idx]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate_position(
        self,
        position: np.ndarray,
        mutation_prob: float,
    ) -> np.ndarray:
        """Apply uniform random mutation to a particle position.

        Each variable is independently perturbed with probability
        *mutation_prob* by a uniform offset scaled to a fraction of the
        variable range.

        Args:
            position: (n_var,) current position.
            mutation_prob: Per-variable mutation probability.

        Returns:
            (n_var,) mutated position (clipped to bounds).
        """
        mutant = position.copy()
        for i in range(self.n_var):
            if self.rng.random() < mutation_prob:
                span = self.upper[i] - self.lower[i]
                mutant[i] += self.rng.uniform(-0.5, 0.5) * span
        return np.clip(mutant, self.lower, self.upper)

    # ------------------------------------------------------------------
    # Personal best update
    # ------------------------------------------------------------------

    def _update_personal_best(
        self,
        positions: np.ndarray,
        objectives: np.ndarray,
        pbest_pos: np.ndarray,
        pbest_obj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update personal bests.  If neither dominates, pick randomly.

        Args:
            positions: (n, n_var) current particle positions.
            objectives: (n, n_obj) current objectives.
            pbest_pos: (n, n_var) personal best positions.
            pbest_obj: (n, n_obj) personal best objectives.

        Returns:
            Updated (pbest_pos, pbest_obj).
        """
        for i in range(len(positions)):
            if self._dominates(objectives[i], pbest_obj[i]):
                pbest_pos[i] = positions[i].copy()
                pbest_obj[i] = objectives[i].copy()
            elif not self._dominates(pbest_obj[i], objectives[i]):
                # Mutually non-dominating — keep one at random
                if self.rng.random() < 0.5:
                    pbest_pos[i] = positions[i].copy()
                    pbest_obj[i] = objectives[i].copy()
        return pbest_pos, pbest_obj

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute MOPSO.

        Returns:
            (pareto_solutions, pareto_objectives): numpy arrays for the
            non-dominated archive found after *n_gen* iterations.
        """
        # --- 1. Initialise particles ---
        positions = self.initialize_population()
        velocities = self.rng.uniform(
            -(self.upper - self.lower),
            (self.upper - self.lower),
            size=(self.pop_size, self.n_var),
        )

        # --- 2. Evaluate objectives ---
        objectives = self.evaluate_population(positions)

        # --- 3. Initialise personal bests ---
        pbest_pos = positions.copy()
        pbest_obj = objectives.copy()

        # --- 4. Initialise archive with non-dominated solutions ---
        nd_mask = self._non_dominated_mask(objectives)
        archive_pos = positions[nd_mask].copy()
        archive_obj = objectives[nd_mask].copy()
        # Prune to archive_size if necessary
        if len(archive_pos) > self.archive_size:
            archive_pos, archive_obj = self._update_archive(
                np.empty((0, self.n_var)),
                np.empty((0, self.n_obj)),
                archive_pos,
                archive_obj,
            )

        for gen in range(self.n_gen):
            t_start = time.time()

            # Mutation probability decreases over generations
            mutation_prob = (1.0 - gen / self.n_gen) * (1.0 / self.n_var)

            for i in range(self.pop_size):
                # --- 5a. Select leader from archive ---
                leader = self._select_leader(archive_pos, archive_obj)

                # --- 5b. Update velocity and position ---
                r1 = self.rng.random(self.n_var)
                r2 = self.rng.random(self.n_var)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (leader - positions[i])
                )
                positions[i] = positions[i] + velocities[i]

                # --- 5c. Apply mutation ---
                positions[i] = self._mutate_position(positions[i], mutation_prob)

                # --- 5d. Clip to bounds ---
                positions[i] = np.clip(positions[i], self.lower, self.upper)

            # --- 5e. Evaluate new positions ---
            objectives = self.evaluate_population(positions)

            # --- 5f. Update personal bests ---
            pbest_pos, pbest_obj = self._update_personal_best(
                positions, objectives, pbest_pos, pbest_obj
            )

            # --- 5g. Update archive ---
            archive_pos, archive_obj = self._update_archive(
                archive_pos, archive_obj, positions, objectives
            )

            # --- 5h. History tracking ---
            self.history['objectives'].append(archive_obj.copy())

            elapsed = time.time() - t_start
            self.history['time_per_gen'].append(elapsed)

            if (gen + 1) % 10 == 0 or gen == 0:
                n_arch = len(archive_obj)
                print(
                    f"MOPSO    gen {gen + 1:>4d}/{self.n_gen}  |  "
                    f"archive size: {n_arch:>4d}  |  "
                    f"mutation prob: {mutation_prob:.4f}  |  "
                    f"time: {elapsed:.3f}s"
                )

        # --- Return archive as Pareto front ---
        return archive_pos, archive_obj
