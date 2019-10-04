from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

enemies = [3, 7, 8]
stats = []

for enemy in enemies:
    EA = pd.read_csv("EA_specialist_enemy_{}.csv".format(enemy))
    DEAP = pd.read_csv("DEAP_specialist_enemy_{}.csv".format(enemy))
    both = pd.concat([EA,DEAP])
    both["method"] = np.array([["EA"]*20, ["DEAP"]*20]).flatten()
    stats.append(both)



fig, ax = plt.subplots(2, len(enemies), sharey = "row", sharex="col", figsize=(12,4))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
#ax = ax.flatten()
ax[1,1].set(xlabel = "Generation")
fig.text(0.07, 0.70, "Fitness" ,va="center", rotation="vertical", fontsize=14)
fig.text(0.07, 0.30, "Energy Points" ,va="center", rotation="vertical", fontsize=14)

for i, enemy in enumerate(enemies):
    ax[0, i].plot(stats[i].index[:20], stats[i].fit_avg[:20], label="SPO-EA avg fitness")
    ax[0, i].plot(stats[i].index[20:], stats[i].fit_avg[20:], label="CMA-ES avg fitness")

    ax[0, i].plot(stats[i].index[:20], stats[i].fit_max[:20], label="SPO-EA max fitness")
    ax[0, i].plot(stats[i].index[20:], stats[i].fit_max[20:], label="CMA-ES max fitness")

    ax[1, i].plot(stats[i].index[:20], stats[i].player_life[:20], label="SPO-EA player life")
    ax[1, i].plot(stats[i].index[20:], stats[i].player_life[20:], label="CMA-ES player life")

    ax[1, i].plot(stats[i].index[:20], stats[i].enemy_life[:20], label="SPO-EA enemy life")
    ax[1, i].plot(stats[i].index[20:], stats[i].enemy_life[20:], label="CMA-ES enemy life")



    # CHOSE FOR STANDARD DEVIATIONS
    ax[0, i].fill_between(stats[i].index[:20], stats[i].fit_avg[:20] +  stats[i].fit_avg_std[:20], stats[i].fit_avg[:20] - stats[i].fit_avg_std[:20], alpha=0.1)
    ax[0, i].fill_between(stats[i].index[20:], stats[i].fit_avg[20:] +  stats[i].fit_avg_std[20:], stats[i].fit_avg[20:] - stats[i].fit_avg_std[20:], alpha=0.1)

    ax[0, i].fill_between(stats[i].index[:20], stats[i].fit_max[:20] +  stats[i].fit_max_std[:20], stats[i].fit_max[:20] - stats[i].fit_max_std[:20], alpha=0.1)
    ax[0, i].fill_between(stats[i].index[20:], stats[i].fit_max[20:] +  stats[i].fit_max_std[20:], stats[i].fit_max[20:] - stats[i].fit_max_std[20:], alpha=0.1)

    ax[1, i].fill_between(stats[i].index[:20], stats[i].player_life[:20] +  stats[i].player_life_std[:20], stats[i].player_life[:20] - stats[i].player_life_std[:20], alpha=0.1)
    ax[1, i].fill_between(stats[i].index[20:], stats[i].player_life[20:] +  stats[i].player_life_std[20:], stats[i].player_life[20:] - stats[i].player_life_std[20:], alpha=0.1)

    ax[1, i].fill_between(stats[i].index[:20], stats[i].enemy_life[:20] +  stats[i].enemy_life_std[:20], stats[i].enemy_life[:20] - stats[i].enemy_life_std[:20], alpha=0.1)
    ax[1, i].fill_between(stats[i].index[20:], stats[i].enemy_life[20:] +  stats[i].enemy_life_std[20:], stats[i].enemy_life[20:] - stats[i].enemy_life_std[20:], alpha=0.1)

    ax[0, i].set_title("Enemy {}".format(enemy))

ax[0, 2].legend(loc=4)
ax[1, 2].legend(loc=1)

plt.show()
