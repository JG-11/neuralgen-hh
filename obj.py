"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

import random

# args = initial_pop(N, pop_size, max_gens)

def getFitness(args):
    chromosome = args[0]
    f1 = random.uniform(0, 1)
    f2 = random.uniform(0, 1)
    features = []
    #print("Chromosome:", chromosome)
    for i in range(len(chromosome)):
        state = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        features.append(state)
    return -f1, -f2, features

"""
1. Population: combinaciones de acciones dentro de cada generación
2. Generation: cantidad de corridas de cada población
3. Chromosome: acciones que se realizan en cada generación
"""

