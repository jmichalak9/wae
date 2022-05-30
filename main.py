
from cma_es import CMAES
import numpy as np
import math


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmaes = CMAES()
    cmaes.calculate(np.array([10.0, 10.0, 10.0]), 5, lambda x, y, z: -(x * y * z) ** 2, 10000)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
