import sys
sys.path.insert(0, "evoman")
import os
import numpy as np
import pandas as pd
import seaborn as sns

RUNS = 10

if __name__ == "__main__":

    cma = pd.read_csv('CMA_generalist_gain.csv', index_col=0, names=['cma'])
    ea = pd.read_csv('EA_generalist_gain.csv', index_col=0,  names=['ea'])
    
    ea['method'] = list(['SPO-EA'] * 10)
    cma['method'] = list(['CMA-ES'] * 10)
    ea.rename(columns={'ea': 'value'}, inplace=True)
    cma.rename(columns={'cma': 'value'}, inplace=True)
    both = pd.concat([cma, ea])

    ax = sns.boxplot(x="method", y="value", data=both)
    ax.set_ylabel("Gain")

