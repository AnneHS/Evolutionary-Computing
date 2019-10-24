import sys

sys.path.insert(0, "evoman")

import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

plt.style.use("ggplot")

RUNS = 10



if __name__ == "__main__":

    cma = pd.read_csv('CMA_generalist_gain.csv', index_col=0, names=['cma'], header=0)
    ea = pd.read_csv('EA_generalist_gain.csv', index_col=0,  names=['ea'], header=0)
    ea['enemy'] = list(range(1,9))*10
    ea_gain = pd.DataFrame(ea.groupby(by='enemy').mean())
    cma['enemy'] = list(range(1, 9)) * 10
    cma_gain = pd.DataFrame(cma.groupby(by='enemy').mean())
    hof = pd.concat([cma_gain,ea_gain], axis=1)
    hof.boxplot()
    plt.ylabel("Maximum fitness")
    plt.savefig("boxplot.png")
    plt.show()
