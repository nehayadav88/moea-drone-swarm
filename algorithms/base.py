"""Base class for multi-objective evolutionary algorithms."""
import numpy as np
from abc import ABC, abstractmethod
import time


class MOEABase(ABC):
    """Abstract base class for MOEAs with consistent interface."""
    
    def __init__(self, problem, pop_size=100, n_gen=200, seed=None):
        """
        Args:
            problem: Problem instance with evaluate(x), n_var, n_obj, bounds properties
            pop_size: Population size
            n_gen: Number of generations
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.lower, self.upper = problem.bounds
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        
        # History tracking
        self.history = {
            'objectives': [],        # Best objectives per generation
            'hv': [],                # Hypervolume per generation
            'feasibility_rate': [],  # Fraction of feasible solutions per gen
            'time_per_gen': [],      # Wall clock time per generation
        }
    
    def initialize_population(self):
        """Create random initial population within bounds."""
        pop = self.rng.uniform(
            self.lower, self.upper, 
            size=(self.pop_size, self.n_var)
        )
        return pop
    
    def evaluate_population(self, pop):
        """Evaluate objectives for entire population. Returns (n, n_obj) array."""
        objectives = np.array([self.problem.evaluate(ind) for ind in pop])
        return objectives
    
    def evaluate_constraints_population(self, pop):
        """Evaluate constraints for entire population. Returns list of constraint arrays."""
        return [self.problem.evaluate_constraints(ind) for ind in pop]
    
    def fast_non_dominated_sort(self, objectives):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
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
        
        return fronts[:-1]  # Remove last empty front
    
    def _dominates(self, a, b):
        """Check if solution a dominates solution b (minimization)."""
        return np.all(a <= b) and np.any(a < b)
    
    def crowding_distance(self, objectives):
        """Compute crowding distance for a set of objectives."""
        n = len(objectives)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        
        for m in range(objectives.shape[1]):
            sorted_idx = np.argsort(objectives[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    (objectives[sorted_idx[i + 1], m] - objectives[sorted_idx[i - 1], m])
                    / obj_range
                )
        
        return distances
    
    def sbx_crossover(self, parent1, parent2, eta=20.0, prob=0.9):
        """Simulated Binary Crossover (SBX)."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if self.rng.random() > prob:
            return child1, child2
        
        for i in range(self.n_var):
            if self.rng.random() > 0.5:
                continue
            if abs(parent1[i] - parent2[i]) < 1e-14:
                continue
                
            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])
            
            beta = 1.0 + (2.0 * (y1 - self.lower[i]) / (y2 - y1 + 1e-14))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            u = self.rng.random()
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            
            beta = 1.0 + (2.0 * (self.upper[i] - y2) / (y2 - y1 + 1e-14))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            u = self.rng.random()
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            
            child1[i] = np.clip(c1, self.lower[i], self.upper[i])
            child2[i] = np.clip(c2, self.lower[i], self.upper[i])
        
        return child1, child2
    
    def polynomial_mutation(self, individual, eta=20.0, prob=None):
        """Polynomial mutation."""
        if prob is None:
            prob = 1.0 / self.n_var
        
        mutant = individual.copy()
        
        for i in range(self.n_var):
            if self.rng.random() > prob:
                continue
            
            y = mutant[i]
            yl = self.lower[i]
            yu = self.upper[i]
            
            if yu - yl < 1e-14:
                continue
            
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            
            u = self.rng.random()
            if u < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** (1.0 / (eta + 1.0))
            
            mutant[i] = np.clip(y + deltaq * (yu - yl), yl, yu)
        
        return mutant
    
    def tournament_selection(self, pop, objectives, fronts=None, crowding=None):
        """Binary tournament selection."""
        n = len(pop)
        i, j = self.rng.integers(0, n, size=2)
        
        if fronts is not None and crowding is not None:
            rank_i = self._get_rank(i, fronts)
            rank_j = self._get_rank(j, fronts)
            
            if rank_i < rank_j:
                return pop[i]
            elif rank_j < rank_i:
                return pop[j]
            else:
                return pop[i] if crowding[i] >= crowding[j] else pop[j]
        else:
            if self._dominates(objectives[i], objectives[j]):
                return pop[i]
            elif self._dominates(objectives[j], objectives[i]):
                return pop[j]
            else:
                return pop[i] if self.rng.random() < 0.5 else pop[j]
    
    def _get_rank(self, idx, fronts):
        """Get Pareto rank of individual."""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)
    
    @abstractmethod
    def run(self):
        """Execute the algorithm. Returns (pareto_front_solutions, pareto_front_objectives)."""
        pass
    
    def get_pareto_front(self, pop, objectives):
        """Extract non-dominated solutions from population."""
        fronts = self.fast_non_dominated_sort(objectives)
        if not fronts:
            return pop, objectives
        pf_idx = fronts[0]
        return pop[pf_idx], objectives[pf_idx]
