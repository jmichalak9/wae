import numpy as np


class CMAES:

    def __init__(self, N, display_result=False):
        self.display = display_result
        self.N = N
        self.offspring_size = int(4.0 + np.floor(3.0 * np.log(N)))
        self.generation_length = int(10.0 + np.ceil(30.0 * N / self.offspring_size))
        self.mu = int(np.floor(self.offspring_size / 2.0))  # Number of best individuals
        self.w = np.full(self.mu, 1.0 / self.mu)
        self.mu_eff = 1.0 / np.sum(self.w ** 2.0)
        self.cp = (self.mu_eff / N + 4.0) / (2.0 * self.mu_eff / N + N + 4.0)
        self.cs = (self.mu_eff + 2.0) / (self.mu_eff + N + 5.0)
        self.alpha_cov = 2.0  # Can be in (0,2>
        self.c1 = self.alpha_cov / ((N + 1.3) ** 2 + self.mu_eff)
        self.cw = np.minimum(1.0 - self.c1, self.alpha_cov *
                             (self.mu_eff + 1.0 / self.mu_eff - 2.0) / ((N + 2.0) ** 2 + self.alpha_cov * self.mu_eff / 2.0))
        # mean of chi distribution with N degrees of freedom using Stirling's approximation
        self.chi_n = np.sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N ** 2))
        self.damping = 1 + self.cs + 2.0 * np.maximum(0.0, np.sqrt((self.mu_eff - 1.0) / (N + 1.0)) - 1.0)

    def recombination(self, vectors):
        new_vectors = []
        for i in range(self.mu):
            new_vectors.append(self.w[i] * vectors[i])

        return np.sum(new_vectors, axis=0)

    def recombination_with_transposition(self, vectors):
        assert len(vectors) == self.mu
        recombination = np.empty([self.N, self.N])
        for i in range(self.mu):
            recombination += self.w[i] * np.outer(vectors[i], vectors[i].T)

        return recombination

    def calculate(self, y, sigma, fun, max_iterations: int):
        p = 0
        s = 0
        covariance_matrix = np.identity(self.N)
        best_values = []
        initial_sigma = sigma
        iteration = 0

        while True:
            # Eigen decomposition
            tmp = (covariance_matrix + np.transpose(covariance_matrix)) / 2
            eigenvalues, B = np.linalg.eigh(tmp)
            D = np.sqrt(np.where(eigenvalues < 0, 10 ** -8, eigenvalues))

            solutions = []
            for i in range(self.offspring_size):
                z = np.random.randn(self.N)  # ~N(0, I)
                d = B.dot(np.diag(D)).dot(z)
                new_y = y + sigma * d  # ~N(m, C * sigma^2)
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
            p = (1 - self.cp) * p + np.sqrt(self.mu_eff * self.cp * (2 - self.cp)) * self.recombination(dd)

            covariance_matrix = (1 - self.c1 - self.cw) * covariance_matrix + \
                                self.c1 * np.outer(p, p.T) + \
                                self.cw * self.recombination_with_transposition(dd)

            sigma *= np.exp(self.cs / self.damping * (np.linalg.norm(s) / self.chi_n - 1))

            if self.display:
                print(solutions[0][0])

            if sigma < initial_sigma * 10 ** -12 or \
                    np.amax(best_values) - np.amin(best_values) < 10 ** -12 or \
                    iteration >= max_iterations:
                return np.amin(best_values)
            iteration += 1
