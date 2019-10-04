import sys
sys.path.insert(0, 'evoman')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
plt.style.use('ggplot')

RUNS = 10
experiment_name = "halloffame"

def fitness(single_weight_matrix):
    f,e,p,t = env.play(pcont=single_weight_matrix)
    return f

if __name__ == "__main__":
    methods = ["DEAP"]
    enemies = [3,7]
    best_fit = np.zeros((10,6))
    print(best_fit)

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

            for run in range(0,RUNS):
                best_contr = np.loadtxt("./{}/best/run_{}_{}.txt".format(method,enemy,run+1))
                fit = float(fitness(best_contr))
                best_fit[run][method_idx+enemy_idx] = fit


    #STORE IN PANDAS WIDE DF
    hof = pd.DataFrame(best_fit, columns=["DEAP_3", "DEAP_7", "DEAP_8", "EA_3", "EA_7" ,"EA_8" ])

    hof.boxplot()
    hof.to_csv("hall_of_fame.csv")
    #BOXPLOTS
