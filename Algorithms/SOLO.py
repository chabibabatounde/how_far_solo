import numpy as np
from mealpy import Optimizer
from math import gamma, sin, pi


class SOLO(Optimizer):
    """
    Single-schedule Optimizer with Leader-guided Operators
    """

    def __init__(self, epoch=100, pop_size=100, **kwargs):
        # Paramètres de l'algorithme
        self.exp_a = None
        self.exp_k = None
        # Paramètres utiles pour l'optimisation
        self.lb = None
        self.ub = None
        self.seed = None
        self.budget = None
        self.stall_limit = None
        self.stall_counter = np.zeros(pop_size, dtype=int)
        # Constructeur de la classe
        super().__init__(**kwargs)
        np.random.seed(self.seed)
        # Main parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])

        # Internal parameters (Computational variable)
        self.dimension = None
        self.lb = None
        self.ub = None
        self.stall_counter = np.zeros(self.pop_size, dtype=int)

        # --- Initialization -----------------------------------------------------

    def initialize_variables(self):
        self.dimension = len(self.problem.ub)
        self.history.pops = []
        self.lb, self.ub = np.array(self.problem.lb), np.array(self.problem.ub)
        self.stall_limit = self.pop_size * 2

    @staticmethod
    def __scale_exponential(value, max_in=100, out_values=(0.1, 0.4), k=1, up=True):
        a = k
        x = value / max_in
        base = (np.exp(a * (x ** k)) - 1) / (np.exp(a) - 1)
        if not up:
            base = 1 - base
        return out_values[0] + (out_values[1] - out_values[0]) * base

    def amend_solution(self, solution):
        x = np.asarray(solution, dtype=float).copy()
        for j in range(len(x)):
            if x[j] < self.lb[j] or x[j] > self.ub[j]:
                x[j] = self.lb[j] + np.random.rand() * (self.ub[j] - self.lb[j])
        return np.clip(x.copy(), self.lb, self.ub)

    def evolve(self, itr):
        current_best = self.pop[0]
        n = self.__scale_exponential(self.nfe_counter, self.budget, (int(self.pop_size * 0.2), self.pop_size),
                                     k=self.exp_k, up=False)
        for i in range(round(n)):
            p = self.pop[i]
            candidate = p.solution.copy()
            best = current_best.solution.copy()
            idxs = list(range(self.pop_size))
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            while r1 == r2 or r1 == i or r2 == i:
                r1, r2 = np.random.choice(idxs, 2, replace=False)

            # (1) Leader guided
            x1 = self.pop[r1].solution.copy()
            x2 = self.pop[r2].solution.copy()
            phi = self.__scale_exponential(self.nfe_counter, self.budget, (0.1, 1), k=self.exp_k - 1, up=False)
            Fi = np.random.uniform(0, phi)
            v = candidate + Fi * (best - candidate) + Fi * (x1 - x2)
            v = self.amend_solution(v)

            #  (2) Local search
            q = self.__scale_exponential(self.nfe_counter, self.budget, (0.0001, 0.1), k=self.exp_a, up=False)
            for d in range(self.problem.n_dims):
                ra = np.random.rand()
                rb = np.random.rand()
                if ra < rb:
                    control = np.random.randint(low=0, high=3)
                    if control == 0:
                        v[d] = candidate[d]
                    elif control == 1:
                        q *= (self.problem.ub[d] - self.problem.lb[d])
                        v[d] = best[d] + (np.random.rand() * 2 - 1) * q
            # Evaluation
            v = self.amend_solution(v)
            v = self.generate_agent(v)

            # Updating
            if v.target.fitness <= p.target.fitness:
                self.pop[i] = v.copy()
                self.stall_counter[i] = 0
            else:
                self.stall_counter[i] += 1
            if v.target.fitness <= current_best.target.fitness:
                current_best = v.copy()
            if self.stall_counter[i] == self.stall_limit and current_best.id != self.pop[i].id:
                self.pop[i] = self.__renew_agent(i)

    def __renew_agent(self, idx):
        mask = np.random.randint(0, 2, size=self.problem.n_dims)
        steps = self.levy_flight(self.problem.n_dims) * mask
        solution = self.pop[idx].solution.copy() + steps
        solution = self.amend_solution(solution)
        return self.generate_agent(solution)

    def levy_flight(self, n):
        beta = 1.5
        sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)
        u = np.random.normal(0, sigma_u, size=n)
        v = np.random.normal(0, 1, size=n)
        steps = 1 * u / np.abs(v) ** (1 / beta)
        return steps
