"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""


from gen import *

from obj import *

if __name__ == "__main__":
    chromosome = [
        "LDEG",
        "GDEG",
        "GDEG",
        "GDEG",
        "LDEG",
        "LDEG",
        "GDEG",
    ]

    args = [chromosome, []]
    burning_nodes, burning_edges, state_features = getFitness(args)

    print("Burning nodes:", burning_nodes)
    print("Burning edges:", burning_edges)
    print("State features:", str(state_features))

    #runNSGA2(N=5, pop_size=10, max_gens=10, runs=10)