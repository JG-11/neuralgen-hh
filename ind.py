"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""


import numpy as np


def takesecond(elem):
    return elem[1]


def getRandomPath(args):
    heuristics, N = args
    chromosome = []
    for i in range(N):
        index = np.random.randint(0,len(heuristics))
        chromosome.append(heuristics[index])
    return chromosome


def initial_pop(N, pop_size, max_gens):
    print('Creating Firefighter Problem Instances. Optimization Algorithm: NSGA-2')
    heuristics = ['LDEG', 'GDEG']
    parallel_pop_args = [[heuristics, N] for i in range(pop_size)]
    pop = [getRandomPath(arg) for arg in parallel_pop_args]
    print('Population size: {}. Generations: {}'.format(len(pop), max_gens))
    print("Pop List")
    print(pop)
    print(len(pop))
    print(len(pop[0]))
    print("Pop List")
    return pop