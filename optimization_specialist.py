###############################################################################
# EvoMan Genetic Algorithm 			                                          #
#                                                                             #
# Author: Lennart Frahm      			                                      #
# lennart.frahm@web.de        				                                  #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import glob, os
from random import randint

# set up experiment folder
experiment_name = '1209_1252'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

# genetic algorithm params

run_mode = 'train' # train or test

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
upper_limit = 1
lower_limit = -1

N = 100
gens = 30
mutation_rate = 0.05
k = 2
number_of_offspring = np.random.randint(1,4)


def documentation(pop, pop_fitness, i):
    # Documentation
    best = np.argmax(pop_fitness)
    std = np.std(pop_fitness)
    mean = np.mean(pop_fitness)

    # saves results
    file_aux = open(experiment_name + '/results.txt', 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(pop_fitness[best], 6))
          + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.write('\n' + str(i) + ' ' + str(round(pop_fitness[best], 6))
                   + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name + '/gen.txt', 'w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name + '/best.txt', pop[best])

    # saves simulation state
    solutions = [pop, pop_fitness]
    env.update_solutions(solutions)
    env.save_state()


# Calculates the fitness of a single weight matrix
def fitness(single_weight_matrix):
    f,e,p,t = env.play(pcont=single_weight_matrix)
    return f


# calculates the fitness for every subject in the population
def pop_evaluation(pop):
    return np.array(list(map(lambda y: fitness(y), pop)))


# selects individual with highest fitness out of <k> competitors
def tournament_selection(pop, pop_fitness):
    highest_fitness = 0
    winner_index = 0
    for _ in range(k):
        selection = randint(0,N-1)
        if (highest_fitness == 0) or (pop_fitness[selection] > highest_fitness):
            highest_fitness = pop_fitness[selection]
            winner_index = selection
    return pop[winner_index]


# mutates proportion of variables
# uses <random resetting> to determine new value
def mutate(single_weight_matrix):
    for i in range(n_vars):
        if np.random.uniform(0,1) <= mutation_rate:
            single_weight_matrix[i] = np.random.uniform(-1,1)


# reproduction function
# chooses 2 parents through <tournament selection>
def crossover(pop, pop_fitness):
    total_offspring = np.zeros((0,n_vars))

    for _ in range(0,N,2):
        parent_1 = tournament_selection(pop, pop_fitness)
        parent_2 = tournament_selection(pop, pop_fitness)

        offspring = np.zeros((number_of_offspring,n_vars))
        for i in range(1, number_of_offspring):
            proportion = np.random.uniform(0,1)
            offspring[i] = parent_1*proportion + parent_2*(1-proportion)
            mutate(offspring[i])
        total_offspring = np.vstack((total_offspring,offspring))
    return total_offspring


def merge_pops(pop, pop_fitness, offspring, offspring_fitness):
    return np.r_[np.c_[pop, pop_fitness], np.c_[offspring,offspring_fitness]]


# reduce pop(now parents + offspring) to N
# currently uses fitness based selection
def survivor_selection(pop, pop_fitness):
    offspring = crossover(pop, pop_fitness)
    offspring_fitness = pop_evaluation(offspring)
    total_pop = merge_pops(pop, pop_fitness, offspring, offspring_fitness)
    ranked_pop = np.argsort(-total_pop[:,-1])[0:N]
    return total_pop[ranked_pop]


def initialization():
    if not os.path.exists(experiment_name + '/evoman_solstate'):
        print('\nNEW EVOLUTION\n')
        pop = np.random.uniform(lower_limit, upper_limit, (N, n_vars))
        pop_fitness = pop_evaluation(pop)
        best = np.argmax(pop_fitness)
        mean = np.mean(pop_fitness)
        std = np.std(pop_fitness)
        ini_g = 0
        solutions = [pop, pop_fitness]
        env.update_solutions(solutions)
    else:
        print('\nCONTINUING EVOLUTION\n')

        env.load_state()
        pop = env.solutions[0]
        pop_fitness = env.solutions[1]

        best = np.argmax(pop_fitness)
        mean = np.mean(pop_fitness)
        std = np.std(pop_fitness)

        # finds last generation number
        file_aux = open(experiment_name + '/gen.txt', 'r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(ini_g) + ' ' + str(round(pop_fitness[best], 6))
          + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.write('\n' + str(ini_g) + ' ' + str(round(pop_fitness[best], 6))
                   + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()
    return pop, pop_fitness, ini_g


def evolution():
    ini = time.time()  # sets time marker
    pop, pop_fitness, ini_g = initialization()
    for i in range(ini_g + 1, gens):
        merged = survivor_selection(pop, pop_fitness)
        pop = merged[:,:-1]
        pop_fitness = merged[:,-1]
        documentation(pop, pop_fitness, i)
    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')


evolution()
env.state_to_log() # checks environment state
