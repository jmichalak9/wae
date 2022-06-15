from ma_es import MAES
import numpy as np


class IPOPMAES():
    max_population_size = 2 ** 5

    def calculate(self, y, sigma, fun, max_iterations: int):
        N = len(y)
        best_result = float('inf')
        best_population_size = float('inf')
        initial_population = int(4.0 + np.floor(3.0 * np.log(N)))
        population = initial_population

        while population < initial_population * self.max_population_size:
            result = MAES(N, population).calculate(y, sigma, fun, max_iterations)
            if result < best_result:
                best_result = result
                best_population_size = population

            population *= 2
        return best_result
