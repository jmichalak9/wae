import numpy as np
from cma_es import CMAES


class IPOPMAES(CMAES):

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
            M = M.dot((I + (self.c1 / 2) * (s * (np.transpose(s)) - I) + (self.cw / 2) *
                     (self.recombination_with_transposition(zz) - I)))

            sigma *= np.exp(self.cs / self.damping * (np.linalg.norm(s) / self.chi_n - 1))

            print(solutions[0][0])

            if sigma < initial_sigma * 10 ** -12 or \
                    np.amax(best_values) - np.amin(best_values) < 10 ** -12 or \
                    iteration >= max_iterations:
                return np.amin(best_values)
            iteration += 1
