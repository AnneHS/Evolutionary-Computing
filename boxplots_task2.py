import sys

sys.path.insert(0, "evoman")

import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use("ggplot")

RUNS = 10



if __name__ == "__main__":

    cma = pd.read_csv('CMA_generalist_gain.csv', index_col=0, names=['cma'], header=0)
    ea = pd.read_csv('EA_generalist_gain.csv', index_col=0,  names=['ea'], header=0)



    ea['enemy'] = list(range(1,9))*10
    ea['run'] = list(np.repeat(range(1,11), 8))
    cma['enemy'] = list(range(1, 9)) * 10
    cma['run'] = list(np.repeat(range(1, 11), 8))

    ea['method'] = list(['ea']*80)
    cma['method'] = list(['cma']*80)
    ea.rename(columns={'ea': 'value'}, inplace=True)
    cma.rename(columns={'cma': 'value'}, inplace=True)

    both = pd.concat([cma, ea])

    ea_gain = ea.pivot(index='run', columns='enemy', values='ea')
    cma_gain = cma.pivot(index='run', columns='enemy', values='cma')

    ax = sns.boxplot(x="enemy", y="value", hue='method', data=both)
    ax.set_ylabel("Gain")

