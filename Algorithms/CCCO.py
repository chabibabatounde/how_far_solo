import numpy as np
from mealpy.optimizer import Optimizer


class CCCO(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, p_coop=0.6, p_comp=0.4, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size

        # paramètres
        self.p_coop = p_coop   # probabilité de coopération
        self.p_comp = p_comp   # probabilité de compétition
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

            r = np.random.rand()

            # =========================
            # COOPERATION PHASE
            # =========================
            if r < self.p_coop:
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].solution
                best = self.g_best.solution

                r1 = np.random.rand(self.problem.n_dims)
                r2 = np.random.rand(self.problem.n_dims)

                # mouvement coopératif
                x_new = xi + r1 * (best - xi) + r2 * (xj - xi)

            # =========================
            # COMPETITION PHASE
            # =========================
            else:
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].solution

                fit_i = self.pop[i].target.fitness
                fit_j = self.pop[j].target.fitness

                # déterminer gagnant/perdant
                if self.problem.minmax == "min":
                    winner = xi if fit_i < fit_j else xj
                    loser = xi if fit_i >= fit_j else xj
                else:
                    winner = xi if fit_i > fit_j else xj
                    loser = xi if fit_i <= fit_j else xj

                # mise à jour du perdant
                r1 = np.random.rand(self.problem.n_dims)
                r2 = np.random.uniform(-1, 1, self.problem.n_dims)

                x_new = loser + r1 * (winner - loser) + r2 * (self.g_best.solution - loser)

            # =========================
            #  BOUND HANDLING
            # =========================
            x_new = self.amend_solution(x_new)

            agent = self.generate_agent(x_new)
            pop_new.append(agent)

        # =========================
        #  GREEDY SELECTION
        # =========================
        self.pop = self.greedy_selection_population(self.pop, pop_new)

        # update best
        self.g_best = self._get_best(self.pop)

        # =========================
        #  DIVERSIFICATION
        # =========================
        if epoch % 10 == 0:
            n_restart = int(0.1 * self.pop_size)
            idxs = np.random.choice(range(self.pop_size), n_restart, replace=False)

            for idx in idxs:
                self.pop[idx] = self.generate_agent(self.problem.generate_solution())

        # =========================
        #  LOCAL SEARCH (élite)
        # =========================
        best = self.g_best.solution.copy()

        for d in range(self.problem.n_dims):
            step = 0.05 * (self.problem.ub[d] - self.problem.lb[d])
            new_sol = best.copy()

            new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * step
            new_sol = self.amend_solution(new_sol)

            new_agent = self.generate_agent(new_sol)

            if self.problem.minmax == "min":
                if new_agent.target.fitness < self.g_best.target.fitness:
                    self.g_best = new_agent
            else:
                if new_agent.target.fitness > self.g_best.target.fitness:
                    self.g_best = new_agent
