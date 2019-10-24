import sys

sys.path.insert(0, "evoman")

import os


from matplotlib import pyplot as plt
import pandas as pd

plt.style.use("ggplot")

RUNS = 10



if __name__ == "__main__":

    cma = pd.read_csv('CMA_generalist_10.csv', index_col=0, names=['cma'], header=0)
    ea = pd.read_csv('EA_generalist_10.csv', index_col=0,  names=['ea'], header=0)
    hof = pd.concat([cma,ea], axis=1)
    hof.boxplot()
    plt.ylabel("Maximum fitness")
    plt.savefig("boxplot.png")
    plt.show()
