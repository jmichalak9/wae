
from cma_es import CMAES
import numpy as np


# Press the green button in the gutter to run the script.
from ipop_ma_es import IPOPMAES

if __name__ == '__main__':
    #IPOPMAES().calculate(np.array([2.0, 2.0, 2.0, 2.0, 2.0]), 5, lambda x: -(np.prod(x)) ** 2, 500)
    y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    CMAES(len(y), 2137).calculate(y, 1, lambda x: -(np.prod(x)) ** 2, 500)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
