from matplotlib import pyplot as plt
import pandas as pd
plt.style.use('ggplot')


enemies = [3,7,8]
stats = []
for enemy in enemies:
    stats.append(pd.read_csv("DEAP_specialist_enemy_{}.csv".format(enemy)))


fig, ax = plt.subplots(2, len(enemies), sharey = "row", sharex="col", figsize=(12,4))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
#ax = ax.flatten()
ax[1,1].set(xlabel = "Generation")
fig.text(0.07, 0.5, "Fitness" ,va="center", rotation="vertical")



for i,enemy in enumerate(enemies):
    ax[0,i].plot(stats[i].index, stats[i].fit_avg, label="average fitness")
    ax[0,i].plot(stats[i].index, stats[i].fit_max, label="max fitness")
    ax[1,i].plot(stats[i].index, stats[i].player_life, label="player_life")
    ax[1,i].plot(stats[i].index, stats[i].enemy_life, label="enemy_life")


    #CHOSE FOR 95% CONFIDENCE ERROR BARS (2 STDS FROM MEAN ON EACH SIDE)
    ax[0,i].fill_between(stats[i].index, stats[i].fit_avg + 2 * stats[i].fit_avg_std, stats[i].fit_avg - 2 * stats[i].fit_avg_std, alpha=0.1)
    ax[0,i].fill_between(stats[i].index, stats[i].fit_max + 2 * stats[i].fit_max_std, stats[i].fit_max - 2 * stats[i].fit_max_std, alpha=0.1)
    ax[1,i].fill_between(stats[i].index, stats[i].player_life + 2 * stats[i].player_life_std, stats[i].player_life - 2 * stats[i].player_life_std, alpha=0.1)
    ax[1,i].fill_between(stats[i].index, stats[i].enemy_life + 2 * stats[i].enemy_life_std, stats[i].enemy_life - 2 * stats[i].enemy_life_std, alpha=0.1)


    ax[0,i].set_title("Enemy {}".format(enemy))
ax[0,2].legend(loc=4)
ax[1,2].legend(loc=4)


plt.show()