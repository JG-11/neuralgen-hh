"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

import random
import glob

from ffp import FFP, GeneticAux

def get_number_of_firefighters(filename):
    file = open(filename, "r")
    text = file.read()    
    tokens = text.split()
    
    int(tokens.pop(0))
    int(tokens.pop(0))
    int(tokens.pop(0))
    int(tokens.pop(0))

    firefighters = int(tokens.pop(0))

    return firefighters

def getFitness(args):
    chromosome = args[0]
    ffp_features = args[1]

    random_folder = random.choice(["GBRL", "BBGRL"])
    random_file = random.choice(glob.glob(f"./instances/{random_folder}/*.in"))
    problem = FFP(random_file)
    genetic_aux = GeneticAux(chromosome)

    firefighters = get_number_of_firefighters(random_file)

    state_features = problem.whole_state

    burning_nodes = problem.solve(genetic_aux, firefighters, False)
    burning_edges = problem.getFeature("BURNING_EDGES")

    fill = len(chromosome) - len(state_features)
    for _ in range(fill):
        state_features[-1].append(["Z"])
    
    return -burning_nodes, -burning_edges, state_features[0]

