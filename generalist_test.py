import sys

sys.path.insert(0, "evoman")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
import os

plt.style.use("ggplot")

# do not display pygame window
os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "test_generalist"


def fitness(single_weight_matrix):
    f, p, e, t = env.play(pcont=single_weight_matrix)
    return p, e


if __name__ == "__main__":

    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    lifes = np.empty((10, 8), dtype=("float64", (2,)))
    log = []

    for run in range(1, 11):
        ctr = np.loadtxt(f"./GENERALIST/EA/best/run_{run}.txt")

        gain = 0
        for counter, enemy in enumerate(enemies):
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy],
                              # multiplemode='yes',
                              playermode="ai",
                              player_controller=player_controller(),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              logs="off")

            player_life = 0
            enemy_life = 0
            for i in range(5):
                p, e = fitness(ctr)
                player_life += float(p)
                enemy_life += float(e)

            gain += (player_life - enemy_life) / 5.0
            lifes[run-1, counter] = [player_life / 5, enemy_life / 5]

        log.append(gain)

    top = pd.DataFrame(lifes[np.argmax(log), :])
    top.to_csv("EA_generalist_life.csv")

    top10 = pd.DataFrame(log)
    top10.to_csv("EA_generalist_gain.csv")
