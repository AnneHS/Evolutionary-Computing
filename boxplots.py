import sys
sys.path.insert(0, "evoman")

import os
# avoid print statements for SPOT
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
# do not display pygame window
os.environ["SDL_VIDEODRIVER"] = "dummy"

from demo_controller import player_controller
from evoman.environment import Environment
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
plt.style.use("ggplot")


RUNS = 10
experiment_name = "halloffame"

def fitness(single_weight_matrix):
    f,e,p,t = env.play(pcont=single_weight_matrix)
    return f

if __name__ == "__main__":

    methods = ["DEAP","EA"]
    enemies = [3,7,8]
    best_fit = np.zeros((10,6))

    for method_idx, method in enumerate(methods):
        for enemy_idx, enemy in enumerate(enemies):
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy],
                              playermode="ai",
                              player_controller=player_controller(),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              logs="off")

            for run in range(RUNS):
                best_contr = np.loadtxt("./{}/best_{}/run_{}.txt".format(method, enemy, run+1))
                fit = float(fitness(best_contr))
                if method_idx == 0:
                    best_fit[run][method_idx+enemy_idx] = fit
                elif method_idx == 1:
                    best_fit[run][2 + method_idx + enemy_idx] = fit

    hof = pd.DataFrame(best_fit, columns=["DEAP e3", "DEAP e7", "DEAP e8", "EA e3", "EA e7", "EA e8"])
    hof.to_csv("hall_of_fame.csv")
    
    hof.boxplot()
    plt.ylabel("Maximum fitness")
    plt.savefig("boxplot.png")
    plt.show()
