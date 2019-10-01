from matplotlib import pyplot as plt
import pandas as pd
plt.style.use('ggplot')


enemies = [3,7,8]
stats = []
for enemy in enemies:
    stats.append(pd.read_csv(f"DEAP_specialist_enemy_{enemy}.csv"))


fig, ax = plt.subplots(1, 3, sharey = "all", figsize=(12,4))
fig.subplots_adjust(hspace=0, wspace=0.05)
ax = ax.flatten()
ax[0].set(ylabel = "Fitness")
ax[1].set(xlabel = "Generation")


for i,enemy in enumerate(enemies):
    ax[i].plot(stats[i].index, stats[i].fit_avg, label="average fitness")
    ax[i].plot(stats[i].index, stats[i].fit_max, label="max fitness")

    #CHOSE FOR 95% CONFIDENCE ERROR BARS (2 STDS FROM MEAN ON EACH SIDE)
    ax[i].fill_between(stats[i].index, stats[i].fit_avg + 2 * stats[i].fit_avg_std, stats[i].fit_avg - 2 * stats[i].fit_avg_std, alpha=0.1)
    ax[i].fill_between(stats[i].index, stats[i].fit_max + 2 * stats[i].fit_max_std, stats[i].fit_max - 2 * stats[i].fit_max_std, alpha=0.1)
    ax[i].set_title(f"Enemy {enemy}")
plt.legend(loc=4)
plt.show()