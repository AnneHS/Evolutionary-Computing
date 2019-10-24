import sys

sys.path.insert(0, 'evoman')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

stats = []

EA = pd.read_csv("./GENERALIST/EA/EA_generalist.csv")
DEAP = pd.read_csv("./GENERALIST/CMA/DEAP_generalist.csv")
both = pd.concat([EA,DEAP])
both["method"] = np.array([["EA"]*30, ["DEAP"]*30]).flatten()
stats.append(both)

fig, ax = plt.subplots(2, 1, sharex="col", figsize=(8,12))
#fig.subplots_adjust(hspace=0.1, wspace=0.1)
#ax = ax.flatten()
ax[1].set(xlabel = "Generation")
ax[0].set(ylabel='Fitness')
ax[1].set(ylabel='Life')

i=0
b = len(EA)
ax[0].plot(stats[i].index[:b], stats[i].fit_avg[:b], label="average fitness_SPO-EA")
ax[0].plot(stats[i].index[b:], stats[i].fit_avg[b:], label="average fitness_DEAP")

ax[0].plot(stats[i].index[:b], stats[i].fit_max[:b], label="max fitness_SPO-EA")
ax[0].plot(stats[i].index[b:], stats[i].fit_max[b:], label="max fitness_DEAP")

ax[1].plot(stats[i].index[:b], stats[i].player_life[:b], label="enemy_life_SPO-EA")
ax[1].plot(stats[i].index[b:], stats[i].player_life[b:], label="enemy_life_DEAP")

ax[1].plot(stats[i].index[:b], stats[i].enemy_life[:b], label="player_life_SPO-EA")
ax[1].plot(stats[i].index[b:], stats[i].enemy_life[b:], label="player_life_DEAP")



# CHOSE FOR 95% CONFIDENCE ERROR BARS (2 STDS FROM MEAN ON EACH SIDE)
ax[0].fill_between(stats[i].index[:b], stats[i].fit_avg[:b] + 2 * stats[i].fit_avg_std[:b], stats[i].fit_avg[:b] - 2 * stats[i].fit_avg_std[:b], alpha=0.1)
ax[0].fill_between(stats[i].index[b:], stats[i].fit_avg[b:] + 2 * stats[i].fit_avg_std[b:], stats[i].fit_avg[b:] - 2 * stats[i].fit_avg_std[b:], alpha=0.1)

ax[0].fill_between(stats[i].index[:b], stats[i].fit_max[:b] + 2 * stats[i].fit_max_std[:b], stats[i].fit_max[:b] - 2 * stats[i].fit_max_std[:b], alpha=0.1)
ax[0].fill_between(stats[i].index[b:], stats[i].fit_max[b:] + 2 * stats[i].fit_max_std[b:], stats[i].fit_max[b:] - 2 * stats[i].fit_max_std[b:], alpha=0.1)

ax[1].fill_between(stats[i].index[:b], stats[i].player_life[:b] + 2 * stats[i].player_life_std[:b], stats[i].player_life[:b] - 2 * stats[i].player_life_std[:b], alpha=0.1)
ax[1].fill_between(stats[i].index[b:], stats[i].player_life[b:] + 2 * stats[i].player_life_std[b:], stats[i].player_life[b:] - 2 * stats[i].player_life_std[b:], alpha=0.1)

ax[1].fill_between(stats[i].index[:b], stats[i].enemy_life[:b] + 2 * stats[i].enemy_life_std[:b], stats[i].enemy_life[:b] - 2 * stats[i].enemy_life_std[:b], alpha=0.1)
ax[1].fill_between(stats[i].index[b:], stats[i].enemy_life[b:] + 2 * stats[i].enemy_life_std[b:], stats[i].enemy_life[b:] - 2 * stats[i].enemy_life_std[b:], alpha=0.1)


ax[0].legend(loc=4)
ax[1].legend(loc=1)

plt.show()
