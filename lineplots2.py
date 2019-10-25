import sys

sys.path.insert(0, 'evoman')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#plt.style.use('ggplot')

stats = []

EA = pd.read_csv("./GENERALIST/EA/EA_generalist.csv")
DEAP = pd.read_csv("./GENERALIST/CMA/DEAP_generalist.csv")
both = pd.concat([EA,DEAP])
EA["method"] = np.array(["SPO-EA"]*30).flatten()
EA = EA.loc[:22]
DEAP["method"] = np.array(["DEAP"]*30).flatten()

stats.append(both)

fig, ax = plt.subplots(2, 1, sharex="col", figsize=(8,12))

ax[1].set(xlabel="Generation SPO-EA")
ax[0].set(ylabel='Fitness')
ax[1].set(ylabel='Life')
ax2 = ax[0].twiny()
ax3 = ax[1].twiny()

ax2.set_xlabel('Generation CMA-ES')


ax[0].plot(EA.index, EA.fit_avg, label="mean fitness SPO-EA")
ax2.plot(DEAP.index, DEAP.fit_avg, label="mean fitness CMA-ES", color='g')

ax[0].plot(EA.index, EA.fit_max, label="max fitness SPO-EA")
ax2.plot(DEAP.index, DEAP.fit_max, label="max fitness CMA-ES", color='r')

ax[1].plot(EA.index, EA.player_life, label="enemy life SPO-EA")
ax3.plot(DEAP.index, DEAP.player_life, label="enemy life CMA-ES", color='g')

ax[1].plot(EA.index, EA.enemy_life, label="player_life SPO-EA")
ax3.plot(DEAP.index, DEAP.enemy_life, label="player life CMA-ES", color='r')



# CHOSE FOR 95% CONFIDENCE ERROR BARS (2 STDS FROM MEAN ON EACH SIDE)
ax[0].fill_between(EA.index, EA.fit_avg + 2 * EA.fit_avg_std, EA.fit_avg - 2 * EA.fit_avg_std, alpha=0.1)
ax2.fill_between(DEAP.index, DEAP.fit_avg + 2 * DEAP.fit_avg_std, DEAP.fit_avg - 2 * DEAP.fit_avg_std, alpha=0.1, color='g')

ax[0].fill_between(EA.index, EA.fit_max + 2 * EA.fit_max_std, EA.fit_max - 2 * EA.fit_max_std, alpha=0.1)
ax2.fill_between(DEAP.index, DEAP.fit_max + 2 * DEAP.fit_max_std, DEAP.fit_max - 2 * DEAP.fit_max_std, alpha=0.1,  color='r')

ax[1].fill_between(EA.index, EA.player_life + 2 * EA.player_life_std, EA.player_life - 2 * EA.player_life_std, alpha=0.1)
ax3.fill_between(DEAP.index, DEAP.player_life + 2 * DEAP.player_life_std, DEAP.player_life - 2 * DEAP.player_life_std, alpha=0.1,  color='g')

ax[1].fill_between(EA.index, EA.enemy_life + 2 * EA.enemy_life_std, EA.enemy_life - 2 * EA.enemy_life_std, alpha=0.1)
ax3.fill_between(DEAP.index, DEAP.enemy_life + 2 * DEAP.enemy_life_std, DEAP.enemy_life - 2 * DEAP.enemy_life_std, alpha=0.1, color='r')

ax[0].set_xticks(ticks=[])

ax[0].legend(loc=4)
ax[1].legend(loc=1)
ax2.legend(loc=8)
ax3.legend(loc=9)
fig.tight_layout()
plt.show()
