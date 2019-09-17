import sys
sys.path.insert(0, "evoman")

import array
from deap import algorithms, base, creator, tools
from demo_controller import player_controller
from environment import Environment
import numpy
import os
import random
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
MIN_WEIGHT = -1
MAX_WEIGHT = 1
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
# CHANGE TOURNSIZE?????
TOURNSIZE = 3


# lower bound on strategy values to avoid pure exploitation
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def evalFitness(single_weight_matrix):
    f,e,p,t = env.play(pcont=numpy.array(single_weight_matrix))
    return f,

def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def main():

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     IND_SIZE, MIN_WEIGHT, MAX_WEIGHT, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalFitness)
    # whole arithmetic recombination on both the individual and the strategy
    toolbox.register("mate", tools.cxESBlend, alpha=0.0)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

    # toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    MU, LAMBDA = 15, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.5, mutpb=0.5, ngen=100, stats=stats, halloffame=hof)
    return pop, logbook, hof


if run_mode == "train":

    ini = time.time()
    pop, logbook, hof = main()
    fim = time.time() # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60.0)) + ' minutes \n')