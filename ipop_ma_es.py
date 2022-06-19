"""This module contains implementation of IPOP-MA-ES"""
from ma_es import MAES
import numpy as np


class IPOPMAES():
  """Implementation of IPOP-MA-ES"""

  max_population_size = 2 ** 5

  def __init__(self, seed):
    self.seed = seed

  def calculate(self, y, sigma, fun, max_iterations: int):
    N = len(y)
    best_result = float('inf')
    initial_population = int(4.0 + np.floor(3.0 * np.log(N)))
    population = initial_population

    while population < initial_population * self.max_population_size:
      result = MAES(N, population, self.seed).calculate(y, sigma, fun,
                                                        max_iterations)
      if result < best_result:
        best_result = result

      population *= 2
    return best_result
