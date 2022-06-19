import numpy as np
from cma_es import CMAES


class MAES(CMAES):

    def __init__(self, N, offspring_size: int, seed: int, display_result=False):
        np.random.seed(seed)
        self.display = display_result
        self.N = N
        self.offspring_size = offspring_size
        self.generation_length = int(10.0 + np.ceil(30.0 * N / self.offspring_size))
        self.mu = int(np.floor(self.offspring_size / 2.0))  # Number of best individuals
        self.w = np.full(self.mu, 1.0 / self.mu)
        self.mu_eff = 1.0 / np.sum(self.w ** 2.0)
        self.cp = (self.mu_eff / N + 4.0) / (2.0 * self.mu_eff / N + N + 4.0)
        self.cs = (self.mu_eff + 2.0) / (self.mu_eff + N + 5.0)
        self.alpha_cov = 2.0  # Can be in (0,2>
        self.c1 = self.alpha_cov / ((N + 1.3) ** 2 + self.mu_eff)
        self.cw = np.minimum(1.0 - self.c1, self.alpha_cov *
                             (self.mu_eff + 1.0 / self.mu_eff - 2.0) / (
                                         (N + 2.0) ** 2 + self.alpha_cov * self.mu_eff / 2.0))

        # mean of chi distribution with N degrees of freedom using Stirling's approximation
        self.chi_n = np.sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N ** 2))
        self.damping = 1 + self.cs + 2.0 * np.maximum(0.0, np.sqrt((self.mu_eff - 1.0) / (N + 1.0)) - 1.0)

    def calculate(self, y, sigma, fun, max_iterations: int):
        s = 0
        initial_sigma = sigma
        best_values = []
        M = np.identity(self.N)
        iteration = 0

        while True:
            solutions = []
            for i in range(self.offspring_size):
                z = np.random.randn(self.N)
                d = M @ z
                new_y = y + sigma * d
                fitness = fun(new_y)
                solutions.append((fitness, new_y, z, d))

            solutions.sort(key=lambda x: x[0], reverse=True)

            fitness_values = [solution[0] for solution in solutions]

            # Remember 'generation_length' maximum and minimum values
            if len(best_values) == self.generation_length * 2:
                best_values.pop(0)
                best_values.pop(0)

            best_values.append(np.amin(fitness_values))
            best_values.append(np.amax(fitness_values))

            zz = np.array([solution[2] for solution in solutions[:self.mu]])
            dd = np.array([solution[3] for solution in solutions[:self.mu]])

            y += sigma * self.recombination(dd)

            s = (1 - self.cs) * s + np.sqrt(self.mu_eff * self.cs * (2 - self.cs)) * self.recombination(zz)
            I = np.identity(self.N)
            M = M @ (I + (self.c1 / 2) * (np.outer(s, s.T) - I) + (self.cw / 2) *
                     (self.recombination_with_transposition(zz) - I))

            sigma *= np.exp(self.cs / self.damping * (np.linalg.norm(s) / self.chi_n - 1))

            if self.display:
                print(solutions[0][0])

            if sigma < initial_sigma * 10 ** -12 or \
                    np.amax(best_values) - np.amin(best_values) < 10 ** -12 or \
                    iteration >= max_iterations:
                return np.amin(best_values)
            iteration += 1
