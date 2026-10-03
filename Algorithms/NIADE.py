import numpy as np
from mealpy.optimizer import Optimizer


class NIADE(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, F=0.4, CR=0.9, **kwargs):
        super().__init__(**kwargs)

        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.phi = (1 + np.sqrt(5)) / 2
        self.lb = None
        self.ub = None

    def _get_best(self, pop):
        best = pop[0]
        for agent in pop:
            if self.problem.minmax == "min":
                if agent.target.fitness < best.target.fitness:
                    best = agent
            else:
                if agent.target.fitness > best.target.fitness:
                    best = agent
        return best

    def initialize_variables(self):
        self.pop = self.generate_population(self.pop_size)
        self.g_best = self._get_best(self.pop)
        self.lb, self.ub = np.array(self.problem.lb), np.array(self.problem.ub)




    def evolve(self, epoch):
        pop_new = []

        for i in range(self.pop_size):
            xi = self.pop[i].solution.copy()

            idxs = list(range(self.pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            x1 = self.pop[r1].solution
            x2 = self.pop[r2].solution
            x3 = self.pop[r3].solution

            v = x1 + self.F * (x2 - x3)

            u = xi.copy()
            j_rand = np.random.randint(0, self.problem.n_dims)

            for j in range(self.problem.n_dims):
                if np.random.rand() <= self.CR or j == j_rand:
                    u[j] = v[j]

            w = (0.5 + np.random.rand() / 2) * self.phi
            x_new = w * u + (1 - w) * xi

            x_new = self.amend_solution(x_new)

            agent = self.generate_agent(x_new)
            pop_new.append(agent)

        # Selection
        self.pop = self.greedy_selection_population(self.pop, pop_new)

        # Best update (SAFE)
        self.g_best = self._get_best(self.pop)

        # Diversity
        if epoch % 10 == 0:
            n_restart = int(0.1 * self.pop_size)
            for i in np.random.choice(range(self.pop_size), n_restart, replace=False):
                self.pop[i] = self.generate_agent(self.problem.generate_solution())

        # Local search
        best = self.g_best.solution.copy()
        for d in range(self.problem.n_dims):
            perturb = 0.1 * (self.problem.ub[d] - self.problem.lb[d])
            new_sol = best.copy()
            new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * perturb
            new_sol = self.amend_solution(new_sol)
            new_agent = self.generate_agent(new_sol)

            if self.problem.minmax == "min":
                if new_agent.target.fitness < self.g_best.target.fitness:
                    self.g_best = new_agent
            else:
                if new_agent.target.fitness > self.g_best.target.fitness:
                    self.g_best = new_agent
