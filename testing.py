import sys
sys.path.insert(0, 'evoman')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
plt.style.use('ggplot')

experiment_name = "test_generalist"


def fitness(single_weight_matrix):
    f,p,e,t = env.play(pcont=single_weight_matrix)
    return p,e


if __name__ == "__main__":

    enemies = [1,2,3,4,5,6,7,8]
    log = []

    for run in range(1, 11):
        ctr = np.loadtxt(f'./GENERALIST/EA/run_{run}.txt')

        gain = 0
        for enemy in enemies:
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy],
                              # multiplemode='yes',
                              playermode="ai",
                              player_controller=player_controller(),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              logs="off")

            gain_enemy = 0
            for i in range(5):
                p, e = fitness(ctr)
                g = float(p) - float(e)
                gain_enemy += g
            gain_enemy /= 5

            gain += gain_enemy
            print(gain)
            log.append(gain)
            top10 = pd.DataFrame(log)
            top10.to_csv('EA_generalist_gain.csv')



