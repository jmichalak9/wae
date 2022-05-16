
from cma_es import cma_es
from ma_es import ma_es
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ma_es(np.array([10.0, 10.0]), 5, lambda x, y: -(x * y) ** 2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
