
from cma_es import CMAES
import numpy as np


# Press the green button in the gutter to run the script.
from ma_es import IPOPMAES

if __name__ == '__main__':
    cmaes = CMAES(5)
    cmaes.calculate(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 5, lambda x: -(np.prod(x)) ** 2, 500)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
