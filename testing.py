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
    f,e,p,t = env.play(pcont=single_weight_matrix)
    return f


if __name__ == "__main__":

    enemies = [1,2,3,4,5,6,7,8]
    log = []

    for run in range(1, 11):
        ctr = np.loadtxt(f'./GENERALIST/CMA/run_{run}.txt')

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          multiplemode='yes',
                          playermode="ai",
                          player_controller=player_controller(),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          logs="off")

        fit = 0
        for i in range(5):
            fit += float(fitness(ctr))
            fit /= 5

        log.append(fit)
        top10 = pd.DataFrame(log)
        top10.to_csv('CMA_generalist_10.csv')



