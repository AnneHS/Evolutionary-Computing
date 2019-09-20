import sys
sys.path.insert(0, "evoman")

from deap import algorithms, base, cma, creator, tools
from demo_controller import player_controller
from environment import Environment
from matplotlib import pyplot as plt
import numpy as np
import os
import time


experiment_name = "DEAP"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

run_mode = "train"


IND_SIZE = (env.get_num_sensors()+1)*10+(10+1)*5
LAMBDA = 100
MU = 15
NGEN = 100


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

    ini = time.time()
    pop, logbook = main()
    fim = time.time() # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60.0)) + ' minutes \n')

    gen, fit_avg, fit_max = logbook.select("gen", "avg", "max")
    print("\nfit_avg\n")
    print(fit_avg)
    print("\nfit_max\n")
    print(fit_max)

    plt.plot(gen, fit_avg, "b", label="average fitness")
    plt.plot(gen, fit_max, "g", label="maximum fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()