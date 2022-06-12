"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

from obj import *

from sklearn.metrics.cluster import adjusted_rand_score

import random
import numpy as np

def compute_ari(arguments):
    x, y = arguments
    return adjusted_rand_score(x, y)


def binary_tournament(pop, f1, f2):
    pop_size = len(pop)
    i, j = np.random.randint(pop_size), np.random.randint(pop_size)
    if (f2[i] <= f2[j]) and (f1[i] <= f1[j]) and (f2[i] < f2[j]) or (f1[i] < f1[j]):
        return pop[i]
    else:
        return pop[j]


def cross(p1, p2):
    p1 = list(p1)
    p2 = list(p2)
    offspring = p1.copy()
    for i in range(len(p1)):
        if np.random.randint(0,2) == 1:
            offspring[i] = p2[i]
    return offspring


def geneUpdatedValue(ind, heuristics):
    index = np.random.randint(0,len(heuristics))
    return heuristics[index]


def mutation(ind):
    mutated = ind.copy()
    for i in range(len(ind)):
        mutProb = 1/len(ind)
        pm = random.uniform(0, 1)
        if pm <= mutProb:
            mutated[i] = geneUpdatedValue(i, ind)
    return mutated


def select_and_recombine(temp_pop_arguments):
    pop, function1_values, function2_values = temp_pop_arguments
    offspring_in_temp_pop = 1
    while offspring_in_temp_pop == 1:
        p1 = binary_tournament(pop, function1_values, function2_values)
        p2 = binary_tournament(pop, function1_values, function2_values)
        offspring = cross(p1, p2)
        mutated = mutation(offspring)
        rand_indexes_offspring_temp_pop = [compute_ari([ind, mutated]) for ind in pop]
        offspring_in_temp_pop = max(rand_indexes_offspring_temp_pop)
    return mutated


def temporal_population_generator(pop, function1_values, function2_values, features, pool):
    temp_pop_arguments = [[pop, function1_values, function2_values] for _ in range(len(pop))]
    temp_pop = list(pool.map(select_and_recombine, temp_pop_arguments))
    return pop + temp_pop

