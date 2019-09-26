###############################################################################
# EvoMan Genetic Algorithm 			                                          #
#                                                                             #
# Author: Lennart Frahm      			                                      #
# lennart.frahm@web.de        				                                 #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
from random import randint

# set up experiment folder
experiment_name = 'rank_based_survivor_selection'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(10),
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
N = 40
gens = 20
mutation_rate = 0.1
K = 2 # how many individuals are drawn for tournament
number_of_offspring = 2


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
    for _ in range(K):
        selection = randint(0,N-1)
        if (highest_fitness == 0) or (pop_fitness[selection] > highest_fitness):
            highest_fitness = pop_fitness[selection]
            winner_index = selection
    return pop[winner_index]


def limit(x):
    if x > upper_limit:
        x = upper_limit
    if x < lower_limit:
        x = lower_limit
    return x


# mutates proportion of variables
def mutate(single_weight_matrix):
    for i in range(n_vars):
        if np.random.uniform(0,1) <= mutation_rate:
            single_weight_matrix[i] = limit(single_weight_matrix[i] + np.random.normal(0,1))


def incest_check(parent1, parent2):
    return np.allclose(parent1, parent2)


# reproduction function
# chooses 2 parents through <tournament selection>
def crossover(pop, pop_fitness):
    total_offspring = np.zeros((0, n_vars))

    for _ in range(0,N,2):
        parent_1 = tournament_selection(pop, pop_fitness)
        parent_2 = tournament_selection(pop, pop_fitness)
        while incest_check(parent_1, parent_2):
            parent_2 = tournament_selection(pop, pop_fitness)
        offspring = np.zeros((number_of_offspring, n_vars))
        for i in range(number_of_offspring):
            proportion = np.random.uniform(0,1)
            offspring[i] = parent_1*proportion + parent_2*(1-proportion)
            mutate(offspring[i])
        total_offspring = np.vstack((total_offspring,offspring))
    return total_offspring


def merge_pops(pop, pop_fitness, offspring, offspring_fitness):
    return np.r_[pop, offspring], np.r_[pop_fitness, offspring_fitness]


# reduce pop(now parents + offspring) to N
# currently uses fitness based selection
def reproduction(pop, pop_fitness):
    offspring = crossover(pop, pop_fitness)
    offspring_fitness = pop_evaluation(offspring)
    new_pop, new_pop_fitness = merge_pops(pop, pop_fitness, offspring, offspring_fitness)
    return new_pop, new_pop_fitness


def norm(single_fitness_value, pop_fitness):
    min = np.min(pop_fitness)
    max = np.max(pop_fitness)
    x_norm = (single_fitness_value - min) / (max - min)
    return x_norm


def survival_selection(whole_pop, whole_pop_fit, selection_mechanism="rank"):
    if selection_mechanism == "fitness":
        whole_pop_fit_norm = np.array(list(map(lambda y: norm(y, whole_pop_fit), whole_pop_fit)))
        probs = whole_pop_fit_norm/whole_pop_fit_norm.sum()
    if selection_mechanism == "rank":
        temp = whole_pop_fit.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(whole_pop_fit))
        probs = ranks/ranks.sum()
    chosen = np.random.choice(whole_pop.shape[0], N, p=probs, replace=False)
    pop = whole_pop[chosen]
    fit_pop = whole_pop_fit[chosen]
    return pop, fit_pop


def initialization():
    if not os.path.exists(experiment_name + '/evoman_solstate'):
        print('\nNEW EVOLUTION\n')
        pop = np.random.uniform(lower_limit, upper_limit, (N, n_vars))
        pop_fitness = pop_evaluation(pop)
        ini_g = 0
        solutions = [pop, pop_fitness]
        env.update_solutions(solutions)
    else:
        print('\nCONTINUING EVOLUTION\n')

        env.load_state()
        pop = env.solutions[0]
        pop_fitness = env.solutions[1]

        # finds last generation number
        file_aux = open(experiment_name + '/gen.txt', 'r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    best = np.argmax(pop_fitness)
    return pop, pop_fitness, ini_g


"""def doomsday(pop, pop_fit):
    # documentation
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\ndoomsday')
    file_aux.close()

    # kill half the population, replace with random new solutions
    random_indices = np.random.choice(N, int(N/2)) # Potentially could be replaced by np.argsort(-pop_fitness)[0:(N/2)]
    survivors = pop[random_indices]
    survivors_fitness = pop_fit[random_indices]
    new_pop = np.random.uniform(lower_limit, upper_limit, (int(N/2), n_vars))
    new_pop_fit = pop_evaluation(new_pop)
    total_pop = np.r_[survivors, new_pop]
    total_pop_fitness = np.r_[survivors_fitness, new_pop_fit]
    return total_pop, total_pop_fitness
"""

def diversity_check(pop, pop_fitness):
    values, count = np.unique(pop_fitness, return_counts=True)
    print(count)
    most_frequent_index = count.argmax()
    most_frequent_fitness = values[most_frequent_index]
    if count.max() > N/3:
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write('\nDiversifying!')
        file_aux.close()
        freq_ind = np.where(pop_fitness == most_frequent_fitness)
        to_delete = np.random.choice(freq_ind[0], int(len(freq_ind[0])/2), replace=False)
        for i in to_delete:
            pop[i] = np.random.uniform(lower_limit, upper_limit, (1, n_vars))
            pop_fitness[i] = pop_evaluation(pop[to_delete])

def evolution():
    ini = time.time()  # sets time marker
    pop, pop_fitness, ini_g = initialization()
    for i in range(ini_g + 1, gens):
        new_pop, new_pop_fitness = reproduction(pop, pop_fitness)
        pop, pop_fitness = survival_selection(new_pop, new_pop_fitness)
        diversity_check(pop, pop_fitness)
        documentation(pop, pop_fitness, i)


    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')


if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    fitness(np.array(bsol))
    sys.exit(0)
if run_mode =="train":
    evolution()
env.state_to_log() # checks environment state








