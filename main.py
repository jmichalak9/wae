"""This module is used only for local tests"""
from cma_es import CMAES
import numpy as np

if __name__ == '__main__':
  y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
  CMAES(len(y), 2137).calculate(y, 1, lambda x: -(np.prod(x)) ** 2, 500)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
