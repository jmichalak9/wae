import numpy as np

class CMAES:

    @staticmethod
    def calculate(y, sigma, fun, maxfun: int):
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
        covariance_matrix = np.identity(N)
        best_values = []
        maxfun_counter = 0

        while True:
            # Eigen decomposition
            tmp = (covariance_matrix + np.transpose(covariance_matrix)) / 2
            eigenvalues, B = np.linalg.eigh(tmp)
            D = np.sqrt(np.where(eigenvalues < 0, 10 ** -8, eigenvalues))

            solutions = []
            for i in range(offspring_size):
                z = np.random.randn(N)  # ~N(0, I)
                d = B.dot(np.diag(D)).dot(z)
                new_y = y + sigma * d  # ~N(m, C * sigma^2)
                fitness = fun(new_y)
                solutions.append((fitness, new_y, z, d))

            solutions.sort(key=lambda x: x[0], reverse=True)

            fitness_values = [solution[0] for solution in solutions]

            # Remember 'generation_length' maximum and minimum values
            if len(best_values) == generation_length * 2:
                best_values.pop(0)
                best_values.pop(0)

            best_values.append(np.amin(fitness_values))
            best_values.append(np.amax(fitness_values))

            zz = np.array([solution[2] for solution in solutions[:mu]])
            dd = np.array([solution[3] for solution in solutions[:mu]])

            y += sigma * np.sum(w * dd, axis=0)
            s = (1 - cs) * s + np.sqrt(mu_eff * cs * (2 - cs)) * np.sum(w * zz, axis=0)
            p = (1 - cp) * p + np.sqrt(mu_eff * cp * (2 - cp)) * np.sum(w * dd, axis=0)

            covariance_matrix = (1 - c1 - cw) * covariance_matrix +\
                                c1 * p.dot(np.transpose(p)) +\
                                cw * np.sum(w * dd.dot(np.transpose(dd)))

            # mean of chi distribution with N degrees of freedom using Stirling's approximation
            chi_n = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))
            damping = 1 + cs + 2 * np.maximum(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1)

            sigma *= np.exp(cs / damping * (np.linalg.norm(s) / chi_n - 1))

            #print([solution[0] for solution in solutions[:mu]])

            if sigma < initial_sigma * 10 ** -12 or \
                    np.amax(best_values) - np.amin(best_values) < 10 ** -12 or \
                    maxfun_counter >= maxfun:
                return np.amin(best_values)
            maxfun_counter += 1
