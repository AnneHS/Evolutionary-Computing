##############################################################
#  THIS SCRIPT IS USED TO RUN THE ALREADY TUNED EA ALGORITHM #
# > IT TRACKS ALL STATS AND SAVES THEM TO CSV                #
##############################################################


import sys
sys.path.insert(0, "evoman")
import os
import pandas as pd
from demo_controller import player_controller
from evoman.environment import Environment
import numpy as np

# do not display pygame window
os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "EA"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# SPECIFY ENEMY/ENEMIES
ENEMY = [7,8]

env = Environment(experiment_name=experiment_name,
                  enemies=ENEMY,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  # avoid print statements for SPOT
                  logs="off")


IND_SIZE = (env.get_num_sensors()+1)*10+(10+1)*5
NGEN = 20
NRUN = 10

UPPER_LIMIT = 1.0
LOWER_LIMIT = -1.0

COUNTER = 0

#m = 10
#m = eval(m.split()[0])
MU = 100
MUTATION_RATE = 0.0206
CROSSOVER_RATE = 0.9347
# tournament size
K = 100
DOOMSDAY_RATIO = 0.8209


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
    global life_stats, COUNTER
    f,e,p,t = env.play(pcont=single_weight_matrix)
    COUNTER += len(ENEMY)
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
    x_norm = (single_fitness_value - np.min(pop_fitness)) / float(np.max(pop_fitness) - np.min(pop_fitness))
    return x_norm


# calculates the fitness for every individual in the population
def pop_evaluation(pop):
    return np.array(list(map(lambda y: fitness(y), pop)))


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
        selection = np.random.randint(0, MU)
        if (highest_fitness == 0) or (pop_fitness[selection] > highest_fitness):
            highest_fitness = pop_fitness[selection]
            winner_index = selection
        if k == MU:
            break
    return pop[winner_index]

if __name__ == "__main__":

    # initialize logging PER RUN
    stats = pd.DataFrame()
    life_stats = []

    for run in range(1, NRUN+1):
        print("Starting run {}...".format(run))

        # initialize logging PER GENERATION
        fit_avg = []
        fit_max = []
        gen = []
        current_best_contr = None
        current_best_fit = 0

        # initialize population
        pop = np.random.uniform(LOWER_LIMIT, UPPER_LIMIT, (MU, IND_SIZE))
        pop_fitness = pop_evaluation(pop)

        for i in range(1, NGEN+1):
            print("Starting generation {}...".format(i))

            new_pop, new_pop_fitness = reproduction(pop, pop_fitness)
            pop, pop_fitness = survivor_selection(new_pop, new_pop_fitness)
            doomsday(pop, pop_fitness, i)

            # check if new max fitness has been reached >>> store
            if np.max(pop_fitness) > current_best_fit:
                current_best_fit = np.max(pop_fitness)  # best solution in generation
                current_best_contr = pop[np.argmax(pop_fitness)]

            # log stats PER GEN
            gen.append(i)
            fit_avg.append(pop_fitness.mean())
            fit_max.append(pop_fitness.max())

        # log stats PER RUN
        stats = stats.append(list(zip(gen, fit_avg, fit_max)))

        # save best controller of the current run
        if not os.path.exists(experiment_name + '/best'):
            os.makedirs(experiment_name + '/best')
        np.savetxt(experiment_name + '/best/run_' + str(run) + '.txt', current_best_contr)


    # data reformatting and aggregation
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

    # write to disk
    avgs.to_csv("EA_specialist_enemy_{}.csv".format(ENEMY))