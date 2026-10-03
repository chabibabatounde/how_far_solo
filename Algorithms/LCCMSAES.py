import numpy as np
from mealpy.optimizer import Optimizer


class LcCMSAES(Optimizer):

    def __init__(self, epoch=1000, pop_size=None, mu=None,
                 sigma_init=None, cond_threshold=1e12, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        # pop_size (lambda) et mu : si non fournis, utilisent les valeurs
        # RECOMMANDEES PAR LE PAPIER (Table I, Spettel/Beyer/Hellwig 2019) :
        #   lambda = 4*D, mu = floor(lambda/4)
        self.pop_size_override = pop_size
        self.mu_override = mu
        self.sigma_init_override = sigma_init
        self.cond_threshold = cond_threshold  # t = 1e12 dans le papier

    def initialize_variables(self):
        n = self.problem.n_dims

        # === Table I du papier : valeurs par defaut si non fournies ===
        self.lam = self.pop_size_override if self.pop_size_override is not None else 4 * n
        self.mu_ = self.mu_override if self.mu_override is not None else max(2, self.lam // 4)
        self.pop_size = self.lam  # pour compatibilite avec la generation de population initiale de mealpy

        self.tau = 1.0 / np.sqrt(2 * n)
        self.tauC = 1 + (n * (n - 1)) / (2 * self.mu_)
        self.cov_update_period = max(1, int(min(np.floor(self.tauC), np.floor(n / 2))))

        # sigma initial = 1/sqrt(D) (Table I), PAS une fraction de la
        # diagonale du domaine -- corrige par rapport a la version precedente
        self.sigma = self.sigma_init_override if self.sigma_init_override is not None else 1.0 / np.sqrt(n)
        self.sigma_stop = 1e-6
        self.g_stop = 10000
        self.g_lag = 50 * n
        self.G_window = 10
        self.eps_abs = 1e-9
        self.eps_rel = 1e-9

        self.C = np.eye(n)
        self.sqrtC = np.eye(n)
        self._x_history = []   # pour le critere d'arret sur la difference de centroides
        self._g_bsf = 0        # derniere generation d'amelioration du best-so-far

    def _project_box(self, x, lb, ub):
        """Projection sur les bornes de boite. Remplace la projection sur
        l'orthant positif relatif a Ax=b du code original, en l'absence
        de contraintes lineaires explicites transmises par le probleme."""
        return np.clip(x, lb, ub)

    def evolve(self, epoch):
        n = self.problem.n_dims
        lb, ub = np.array(self.problem.lb), np.array(self.problem.ub)
        x = self.g_best.solution.copy()

        # === Generation des lambda descendants (boucle "for k = 1:lambda" du code original) ===
        offsprings = []
        for k in range(self.lam):
            sigma_k = self.sigma * np.exp(self.tau * np.random.randn())     # offspring.sigma
            s_k = self.sqrtC @ np.random.randn(n)                            # offspring.s
            z_k = sigma_k * s_k                                              # offspring.z (B = Identite ici)
            x_k = x + z_k                                                    # offspring.x
            x_k = self._project_box(x_k, lb, ub)                             # projection (cf. note de portee)
            agent = self.generate_agent(x_k)
            offsprings.append({"agent": agent, "sigma": sigma_k, "s": s_k})

        # === Selection des mu meilleurs, ponderation uniforme 1/mu (comme le code original) ===
        offsprings.sort(key=lambda o: o["agent"].target.fitness)
        best_mu = offsprings[:self.mu_]

        weight = 1.0 / self.mu_
        z_centroid = np.zeros(n)
        sigma_centroid = 0.0
        ss_centroid = np.zeros((n, n))
        for o in best_mu:
            z_centroid += weight * (o["agent"].solution - x)
            sigma_centroid += weight * o["sigma"]
            ss_centroid += weight * np.outer(o["s"], o["s"])

        # === Mise a jour du centre, du pas et de la covariance ===
        # sigma <- <sigma_tilde> SANS plafond artificiel (le papier n'impose
        # pas de sigma_max ; corrige par rapport a la version precedente)
        new_x = self._project_box(x + z_centroid, lb, ub)
        self.sigma = sigma_centroid
        self.C = (1 - 1.0 / self.tauC) * self.C + (1.0 / self.tauC) * ss_centroid

        # === Recalcul periodique de sqrt(C), avec regularisation du
        # conditionnement (Eq. 14/25 du papier), pas un simple clipping ===
        if epoch % self.cov_update_period == 0:
            self.sqrtC = self._compute_sqrtC_normalized(self.C, self.cond_threshold)
            self.C = self.sqrtC @ self.sqrtC.T

        new_agent = self.generate_agent(new_x)
        self.pop = [o["agent"] for o in offsprings]

        improved = new_agent.target.fitness < self.g_best.target.fitness
        if improved:
            self.g_best = new_agent
            self._g_bsf = epoch

        # === Critere d'arret sur la difference de centroides (fenetre G) ===
        self._x_history.append(new_x)
        if len(self._x_history) > self.G_window:
            self._x_history.pop(0)
        stop = False
        if self.sigma < self.sigma_stop:
            stop = True
        elif epoch - self._g_bsf >= self.g_lag:
            stop = True
        elif len(self._x_history) == self.G_window:
            diffs = [np.linalg.norm(self._x_history[i] - self._x_history[i-1])
                     for i in range(1, self.G_window)]
            rel_diffs = [abs(np.linalg.norm(self._x_history[i]) / (np.linalg.norm(self._x_history[i-1]) + 1e-300) - 1)
                         for i in range(1, self.G_window)]
            if max(diffs) < self.eps_abs or max(rel_diffs) < self.eps_rel:
                stop = True
        if stop:
            self.termination_flag = True  # a verifier selon la version de mealpy utilisee

    @staticmethod
    def _compute_sqrtC_normalized(C, t):
        """Traduction de computeSqrtCNormalized (Alg. 4 du papier) :
        racine carree symetrique de C, regularisee si le conditionnement
        depasse le seuil t, puis normalisee (determinant = 1)."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-300, None)  # securite numerique minimale
        lam1, lamN = eigvals[0], eigvals[-1]
        r = 0.0
        if lamN / lam1 > t:
            r = (np.sqrt(lamN) / t - np.sqrt(lam1)
                 + np.sqrt(max(lamN / t**2 + lamN / t - 2 * np.sqrt(lam1 * lamN) / t, 0.0)))
        M_r = eigvecs @ np.diag(np.sqrt(eigvals) + r) @ eigvecs.T
        # normalisation : determinant(M_r_normalized) = 1
        det_Mr = np.linalg.det(M_r)
        n = C.shape[0]
        if det_Mr > 0:
            M_r = M_r * (det_Mr ** (-1.0 / n))
        return M_r