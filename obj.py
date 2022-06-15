"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

import random
import glob

from ffp import FFP

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

    firefighters = get_number_of_firefighters(random_file)

    number_heuristics = len(chromosome)
    i = 0
    state_features = []
    nodes_in_danger = float('inf')
    while number_heuristics > 0 and nodes_in_danger > 0:
        state = []
        for i in range(len(ffp_features)):
            state.append(problem.getFeature(ffp_features[i]))
        state_features.append(state)

        burning_nodes = problem.solve(chromosome[i], firefighters, False)
        
        number_heuristics -= 1
        i += 1

        nodes_in_danger = problem.getFeature("NODES_IN_DANGER")

        burning_edges = problem.getFeature("BURNING_EDGES")
    
    fill = len(chromosome) - len(state_features)
    for i in range(fill):
        state_features.append(["Z"])

    return -burning_nodes, -burning_edges, state_features

