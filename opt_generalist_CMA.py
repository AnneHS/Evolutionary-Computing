##########################################################
# SCRIPT USED TO RUN THE BASELINE ALGORITHM              #
##########################################################

import sys

sys.path.insert(0, "evoman")

from deap import algorithms, base, cma, creator, tools
from demo_controller import player_controller
from environment import Environment
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import pandas as pd

plt.style.use('ggplot')

#os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "DEAP"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

ENEMY = [7,8]
env = Environment(experiment_name=experiment_name,
                  enemies=ENEMY,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(10),
                  enemymode="static",
                  level=2,
                  speed="fastest")

run_mode = "train"

IND_SIZE = (env.get_num_sensors() + 1) * 10 + (10 + 1) * 5
LAMBDA = 100
MU = 15
NGEN = 20
NUM_RUNS = 10
COUNTER = 0


def evalFitness(single_weight_matrix):
    global life_stats, COUNTER
    f, p, e, t = env.play(pcont=np.array(single_weight_matrix))
    COUNTER += len(ENEMY)
    print(COUNTER)
    life_stats.append([p, e])
    return f,


def main():
    # CMA-ES CONFIGURATION
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    strategy = cma.Strategy(centroid=[np.random.uniform(-1, 1) for _ in range(IND_SIZE)], sigma=5.0,
                            lambda_=LAMBDA, mu=MU)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evalFitness)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)
    return pop, logbook, hof


if run_mode == "train":

    stats = pd.DataFrame()
    life_stats = []

    for i in range(1, NUM_RUNS + 1):
        ini = time.time()
        pop, logbook, hof = main()
        fim = time.time()  # prints total execution time for experiment
        print("Run {} of {} finished!".format(i, ENEMY))
        print('\nExecution time: ' + str(round((fim - ini) / 60.0)) + ' minutes \n')

        # logging stats
        gen, fit_avg, fit_max = logbook.select("gen", "avg", "max")
        stats = stats.append(list(zip(gen, fit_avg, fit_max)))

        # write best controllers disk
        if not os.path.exists(experiment_name + '/best'):
            os.makedirs(experiment_name + '/best')
        np.savetxt(experiment_name + '/best/run_' + str(i) + '.txt', hof[0])

    # data transformation and aggregation
    life = pd.DataFrame(life_stats)
    life_agg = life.groupby(life.index // LAMBDA).mean()
    life_agg.columns = ["player_life", "enemy_life"]
    stats.columns = ["gen", "fit_avg", "fit_max"]
    stats.reset_index(drop=True, inplace=True)
    avg_stats = pd.concat([stats, life_agg], axis=1)

    means = avg_stats.groupby("gen").agg(np.mean)
    stdvs = avg_stats.groupby("gen").agg(np.std)
    stdvs.columns = ["fit_avg_std", "fit_max_std", "player_life_std", "enemy_life_std"]
    avgs = pd.concat([means, stdvs], axis=1)

    # write to disk
    avgs.to_csv("DEAP_specialist_enemy_{}.csv".format(ENEMY))
