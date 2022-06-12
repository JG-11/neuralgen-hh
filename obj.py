"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

from global_vars import *

# args = [objective_arguments]
def getFitness(args):
    print(F1)
    chromosome = args[0]
    features = []
    for i in range(len(chromosome)):
        state = []
        for i in range(len(FFP_FEATURES)):
            state.append(0)
        features.append(state)
    return -F1, -F2, features

"""
1. Population: combinaciones de acciones dentro de cada generación
2. Generation: cantidad de corridas de cada población
3. Chromosome: acciones que se realizan en cada generación
"""

