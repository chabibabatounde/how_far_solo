import numpy as np
from mealpy.optimizer import Optimizer


class BBO(Optimizer):
    """
    Beaver Behavior Optimizer (BBO).

    Reference:
        Ouyang, K., Wei, D., Sha, X., Yu, J., Zhao, Y., Qiu, M., Fu, S.,
        Heidari, A. A., & Chen, H. (2026). Beaver behavior optimizer: A
        novel metaheuristic algorithm for solar PV parameter
        identification and engineering problems. Journal of Advanced
        Research, 84, 525-555. https://doi.org/10.1016/j.jare.2025.09.001
    """

    def __init__(self, epoch=1000, pop_size=100, architect_ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.architect_ratio = architect_ratio  # proportion d'architectes (25% dans le papier)

    def initialize_variables(self):
        pass  # aucun etat interne supplementaire necessaire

    def evolve(self, epoch):
        n = self.problem.n_dims
        lb, ub = np.array(self.problem.lb), np.array(self.problem.ub)
        pop_size = len(self.pop)

        # === Facteur de phase de barrage (Eq. 3) : D = sin(pi*t / (2T)) ===
        # t indexe a partir de 1 dans le code MATLAB d'origine (boucle "for t = 1:Max_iteration")
        t = epoch
        T = self.epoch
        D_factor = np.sin(np.pi * t / (2 * T))

        # Classement de la population pour identifier les architectes (top 25%)
        fitness_values = np.array([agent.target.fitness for agent in self.pop])
        sorted_idx = np.argsort(fitness_values)  # minimisation
        n_architects = max(1, round(pop_size * self.architect_ratio))
        architects_idx = set(sorted_idx[:n_architects].tolist())
        best_solution = self.pop[sorted_idx[0]].solution.copy()

        new_pop = []
        for i in range(pop_size):
            x_i = self.pop[i].solution.copy()

            if np.random.rand() < D_factor:
                # === Exploitation : maintenance du barrage (Eq. 6) ===
                k = np.random.randint(pop_size)
                while k == i:
                    k = np.random.randint(pop_size)
                for j in range(n):
                    r7, r8 = np.random.rand(), np.random.rand()
                    x_i[j] = x_i[j] + r7 * (self.pop[k].solution[j] - x_i[j]) \
                                     + r8 * (best_solution[j] - x_i[j])
            else:
                # === Exploration : collecte de materiaux (Eq. 4-5) ===
                if i in architects_idx:
                    # Architecte : apprentissage mutuel entre architectes (Eq. 4)
                    for j in range(n):
                        if np.random.rand() < 0.5:
                            k = np.random.choice(list(architects_idx))
                            x_i[j] = x_i[j] + np.random.rand() * (self.pop[k].solution[j] - x_i[j])
                else:
                    # Prospecteur : apprentissage des architectes OU exploration (Eq. 5)
                    for j in range(n):
                        if np.random.rand() < 0.5:
                            k = np.random.choice(list(architects_idx))
                            x_i[j] = x_i[j] + np.random.rand() * (self.pop[k].solution[j] - x_i[j])
                        else:
                            disturbance = (np.cos(np.pi * t / (2 * T)) * (ub[j] - lb[j])
                                           * np.random.randn() / 10.0)
                            x_i[j] = x_i[j] + disturbance

            x_i = np.clip(x_i, lb, ub)
            agent = self.generate_agent(x_i)
            if agent.target.fitness < self.pop[i].target.fitness:
                new_pop.append(agent)
            else:
                new_pop.append(self.pop[i])

        self.pop = new_pop
        current_best = min(self.pop, key=lambda a: a.target.fitness)
        if current_best.target.fitness < self.g_best.target.fitness:
            self.g_best = current_best