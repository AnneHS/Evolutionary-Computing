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

experiment_name = "DEAP"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


ENEMY = 3
env = Environment(experiment_name=experiment_name,
                  enemies=[ENEMY],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

run_mode = "train"


IND_SIZE = (env.get_num_sensors()+1)*10+(10+1)*5
LAMBDA = 100
MU = 15
NGEN = 20
NUM_RUNS = 10


def evalFitness(single_weight_matrix):
    f,e,p,t = env.play(pcont=np.array(single_weight_matrix))
    return f,

def main():

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evalFitness)

    strategy = cma.Strategy(centroid=[np.random.uniform(-1, 1) for _ in range(IND_SIZE)], sigma=5.0,
                            lambda_=LAMBDA, mu=MU)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)
    return pop, logbook


if run_mode == "train":

    stats = pd.DataFrame()

    for i in range(1, NUM_RUNS+1):
        ini = time.time()
        pop, logbook = main()
        fim = time.time() # prints total execution time for experiment
        print(f"Run {i} of {NUM_RUNS} finished!")
        print('\nExecution time: ' + str(round((fim - ini) / 60.0)) + ' minutes \n')

        gen, fit_avg, fit_max = logbook.select("gen", "avg", "max")
        print("\nfit_avg\n")
        print(fit_avg)
        print("\nfit_max\n")
        print(fit_max)

        stats = stats.append(list(zip(gen, fit_avg, fit_max)))

    stats.columns = ["gen", "fit_avg", "fit_max"]
    means = stats.groupby("gen").agg(np.mean)
    stdvs = stats.groupby("gen").agg(np.std)
    stdvs.columns = ["fit_avg_std", "fit_max_std"]
    avg_stats = pd.concat([means,stdvs], axis=1)

    # CHANGE THIS ACCORDING TO ENEMY SELECTION
    avg_stats.to_csv(f"DEAP_specialist_enemy_{ENEMY}_LOG.csv")

    fig, ax = plt.subplots(1)
    ax.plot(avg_stats.index, avg_stats.fit_avg, label="average fitness")
    ax.plot(avg_stats.index, avg_stats.fit_max, label="max fitness")

    #CHOSE FOR 95% CONFIDENCE ERROR BARS (2 STDS FROM MEAN ON EACH SIDE)
    ax.fill_between(avg_stats.index, avg_stats.fit_avg + 2 * avg_stats.fit_avg_std, avg_stats.fit_avg - 2 * avg_stats.fit_avg_std, alpha=0.1)
    ax.fill_between(avg_stats.index, avg_stats.fit_max + 2 * avg_stats.fit_max_std, avg_stats.fit_max - 2 * avg_stats.fit_max_std, alpha=0.1)
    ax.legend()
    plt.xlabel("Generation")
    plt.savefig(f"DEAP_specialist_enemy_{ENEMY}.png")
    plt.show()




"""
TO DO:
- general stats and plotting framework
- generic? (use logbook DEAP or log VU)
"""