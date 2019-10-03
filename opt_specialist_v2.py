import sys
sys.path.insert(0, "evoman")

import os
import pandas as pd
# avoid print statements for SPOT
#os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
# do not display pygame window
#os.environ["SDL_VIDEODRIVER"] = "dummy"

from demo_controller import player_controller
from evoman.environment import Environment
import numpy as np
# import time


experiment_name = "EA"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

ENEMY = 3
env = Environment(experiment_name=experiment_name,
                  enemies=[ENEMY],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  # avoid print statements for SPOT
                  logs="off")


IND_SIZE = (env.get_num_sensors()+1)*10+(10+1)*5
RUN_MODE = "train"
NGEN = 5
NRUN = 3

UPPER_LIMIT = 1.0
LOWER_LIMIT = -1.0

#m = 10
#m = eval(m.split()[0])
MU = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
# tournament size
K = 2
DOOMSDAY_RATIO = 0.5


# ??????????
# implements diversity control
def doomsday(pop, pop_fitness, gen):
    values, count = np.unique(pop_fitness, return_counts=True)
    most_frequent_index = np.argmax(count)
    most_frequent_fitness = values[most_frequent_index]
    # if over DOOMSDAY_RATIO of the population is identical, ...
    if (np.max(count) > MU * DOOMSDAY_RATIO) and (gen <= NGEN - 5):
        freq_ind = np.where(pop_fitness == most_frequent_fitness)
        # ... delete half of the identical individuals...
        to_delete = np.random.choice(freq_ind[0], int(len(freq_ind[0]) * 0.5), replace=False)
        # ... and replace them with random ones
        for i in to_delete:
            pop[i] = np.random.uniform(LOWER_LIMIT, UPPER_LIMIT, (1, IND_SIZE))
            pop_fitness[i] = fitness(pop[i])


# calculates the fitness of a single weight matrix
def fitness(single_weight_matrix):
    global life_stats
    f,e,p,t = env.play(pcont=single_weight_matrix)
    life_stats.append([p, e])
    return f


# checks for incest
def incest_check(parent_1, parent_2):
    return np.allclose(parent_1, parent_2)


# sets limit on weights
def limit(x):
   return np.clip(x,LOWER_LIMIT,UPPER_LIMIT)



# merges the populations
def merge_pops(pop, pop_fitness, offspring, offspring_fitness):
    return np.r_[pop, offspring], np.r_[pop_fitness, offspring_fitness]


# implements mutation
def mutation(single_weight_matrix):
    for i in range(IND_SIZE):
        if np.random.uniform(0, 1) <= MUTATION_RATE:
            single_weight_matrix[i] = limit(single_weight_matrix[i] + np.random.normal(0, 1))


# normalizes a single fitness value
def norm(single_fitness_value, pop_fitness):
    # ??????????
    x_norm = (single_fitness_value - np.min(pop_fitness)) / float(np.max(pop_fitness) - np.min(pop_fitness))
    return x_norm


# calculates the fitness for every individual in the population
def pop_evaluation(pop):
    return np.array(list(map(lambda y: fitness(y), pop)))


# ??????????
# implements reproduction
def reproduction(pop, pop_fitness):
    total_offspring = np.zeros((0, IND_SIZE))
    for _ in range(0, MU, 2):
        # parent selection through tournament selection
        parent_1 = tournament_selection(pop, pop_fitness)
        parent_2 = tournament_selection(pop, pop_fitness)
        while incest_check(parent_1, parent_2):
            parent_2 = tournament_selection(pop, pop_fitness)
        offspring = np.zeros((2, IND_SIZE))
        # crossover
        if np.random.uniform(0, 1) <= CROSSOVER_RATE:
            # whole arithmetic crossover
            proportion = np.random.uniform(0, 1)
            offspring[0] = parent_1 * proportion + parent_2 * (1 - proportion)
            offspring[1] = parent_2 * proportion + parent_1 * (1 - proportion)
        else:
            offspring[0] = parent_1
            offspring[1] = parent_2
        # mutation
        mutation(offspring[0])
        mutation(offspring[1])
        total_offspring = np.vstack((total_offspring, offspring))
    total_offspring_fitness = pop_evaluation(total_offspring)
    new_pop, new_pop_fitness = merge_pops(pop, pop_fitness, total_offspring, total_offspring_fitness)
    return new_pop, new_pop_fitness


# implements survivor selection
def survivor_selection(whole_pop, whole_pop_fit, selection_mechanism="rank"):
    if selection_mechanism == "fitness":
        whole_pop_fit_norm = np.array(list(map(lambda y: norm(y, whole_pop_fit), whole_pop_fit)))
        probs = whole_pop_fit_norm / float(sum(whole_pop_fit_norm))
    elif selection_mechanism == "rank":
        temp = np.argsort(whole_pop_fit)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(whole_pop_fit))
        probs = ranks / float(sum(ranks))
    chosen = np.random.choice(whole_pop.shape[0], MU, p=probs, replace=False)
    pop = whole_pop[chosen]
    fit_pop = whole_pop_fit[chosen]
    return pop, fit_pop


# selects individual with highest fitness out of <k> competitors
def tournament_selection(pop, pop_fitness):
    highest_fitness = 0
    winner_index = 0
    for k in range(K):
        # ??????????
        selection = np.random.randint(0, MU)
        if (highest_fitness == 0) or (pop_fitness[selection] > highest_fitness):
            highest_fitness = pop_fitness[selection]
            winner_index = selection
        if k == MU:
            break
    return pop[winner_index]



if RUN_MODE == "train":

    stats = pd.DataFrame()
    life_stats = []

    for run in range(1, NRUN+1):

        fit_avg = []
        fit_max = []
        gen = []
        current_best_contr = None
        current_best_fit = 0

        # ini = time.time()
        # initialize population
        pop = np.random.uniform(LOWER_LIMIT, UPPER_LIMIT, (MU, IND_SIZE))
        pop_fitness = pop_evaluation(pop)
        for i in range(1, NGEN+1):
            new_pop, new_pop_fitness = reproduction(pop, pop_fitness)
            pop, pop_fitness = survivor_selection(new_pop, new_pop_fitness)
            doomsday(pop, pop_fitness, i)
            if np.max(pop_fitness) > current_best_fit:
                current_best_fit = np.max(pop_fitness)  # best solution in generation
                current_best_contr = pop[np.argmax(pop_fitness)]

            gen.append(i)
            fit_avg.append(pop_fitness.mean())
            fit_max.append(pop_fitness.max())


        # fim = time.time()  # prints total execution time for experiment
        # print('\nExecution time: ' + str(round((fim - ini) / 60.0)) + ' minutes \n')

        # required print statement for SPOT
        #print(-np.mean(pop_fitness))
        stats = stats.append(list(zip(gen, fit_avg, fit_max)))

        if not os.path.exists(experiment_name + '/best'):
            os.makedirs(experiment_name + '/best')
        np.savetxt(experiment_name + '/best/run_' + str(run) + '.txt', current_best_contr)

    life = pd.DataFrame(life_stats)
    life_agg = life.groupby(life.index // MU).mean()
    life_agg.columns = ["player_life", "enemy_life"]
    stats.columns = ["gen", "fit_avg", "fit_max"]
    stats.reset_index(drop=True, inplace=True)
    avg_stats = pd.concat([stats, life_agg], axis=1)

    means = avg_stats.groupby("gen").agg(np.mean)
    stdvs = avg_stats.groupby("gen").agg(np.std)
    stdvs.columns = ["fit_avg_std", "fit_max_std", "player_life_std", "enemy_life_std"]
    avgs = pd.concat([means, stdvs], axis=1)

    # CHANGE THIS ACCORDING TO ENEMY SELECTION
    avgs.to_csv("EA_specialist_enemy_{}.csv".format(ENEMY))