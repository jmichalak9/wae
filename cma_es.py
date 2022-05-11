import numpy as np
import time


def cma_es(y, sigma, fun):
    p = 0
    s = 0
    N = y.size
    initial_sigma = sigma
    offspring_size = int(4 + 3 * np.floor(np.log(N)))
    generation_length = int(10 + np.ceil(30 * N / offspring_size))
    mu = int(np.floor(offspring_size / 2))  # Number of best individuals
    w = np.full(N, 1.0 / mu)
    mu_eff = 1 / np.sum(w ** 2)
    cp = (mu_eff / N + 4) / (2 * mu_eff / N + N + 4)
    cs = (mu_eff + 2) / (mu_eff + N + 5)
    alpha_cov = 2  # Can be in (0,2>
    c1 = alpha_cov / ((N + 1.3) ** 2 + mu_eff)
    cw = np.minimum(1 - c1, alpha_cov * (mu_eff + 1 / mu_eff - 2) / ((N + 2) ** 2 + alpha_cov * mu_eff / 2))
    C = np.identity(N)
    best_values = []

    while True:
        evalues, evectors = np.linalg.eig(C)
        M = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)

        solutions = []
        for i in range(offspring_size):
            z = np.random.randn(N)
            d = M @ z
            new_y = y + sigma * d
            fitness = fun(*new_y)
            solutions.append((fitness, new_y, z, d))

        solutions.sort(key=lambda x: x[0], reverse=True)

        fitness_values = [solution[0] for solution in solutions]

        # Remember 'generation_length' maximum and minimum values
        if len(best_values) == generation_length * 2:
            best_values.pop(0)
            best_values.pop(0)

        best_values.append(np.amin(fitness_values))
        best_values.append(np.amax(fitness_values))

        zz = [solution[2] for solution in solutions[:mu]]
        dd = [solution[3] for solution in solutions[:mu]]

        y += sigma * np.sum(w * dd, axis=0)
        s = (1 - cs) * s + np.sqrt(mu_eff * cs * (2 - cs)) * np.sum(w * zz, axis=0)
        p = (1 - cp) * p + np.sqrt(mu_eff * cp * (2 - cp)) * np.sum(w * dd, axis=0)

        C = (1 - c1 - cw) * C + c1 * p * np.transpose(p) + cw * np.sum(w * dd * np.transpose(dd))

        D = np.sqrt(N)
        sigma = sigma * np.exp((1 / 2 * D) * ((np.linalg.norm(s) ** 2) / N - 1))  # Change to Chi distribution

        print([solution[0] for solution in solutions[:mu]])
        time.sleep(1)

        # First stop condition
        if sigma < initial_sigma * 10 ** -12:
            break

        # Second stop condition
        if np.amax(best_values) - np.amin(best_values) < 10 ** -12:
            break
