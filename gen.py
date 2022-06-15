"""
Authors:
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
"""

from oper import *
from obj import *
from ind import *


import pandas as pd
import numpy as np
import math
import time
import multiprocessing
import os

def index_of(a, lst):
    for i in range(len(lst)):
        if lst[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(0, len(values1))]
    rank = [0 for _ in range(0, len(values1))]
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front


def crowding_distance(arguments):
    values1, values2, front = arguments
    distance = [0 for _ in range(len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance


def DataFrame_preparation(N, fronts, solutions, pop_size, neg_function1, neg_function2, runtime, max_gens, pop, last_features):
    d = dict()
    d['Chromosome size'] = [N]*solutions
    d['Population size'] = [pop_size]*solutions
    d['Max. gens'] = [max_gens]*solutions
    d['No. objectives'] = [2]*solutions
    d['Obj. 1 name'] = ['Time']*solutions
    d['Objective 1'] = neg_function1
    d['Obj. 2 name'] = ['Nodes on fire']*solutions
    d['Objective 2'] = neg_function2
    d['Time'] = [runtime]*solutions
    for i in range(len(pop[0])):
        d['F{}'.format(i+1)] = list()
    for i in range(len(pop[0])):
        d['H{}'.format(i+1)] = list()
    for solution in fronts[0]:
        for gene in range(len(pop[solution])):
            d['H{}'.format(gene + 1)].append(str(pop[solution][gene])) 
            d['F{}'.format(gene + 1)].append(last_features[solution][gene])
    out = pd.DataFrame(d)
    return out


def result_export(pop_size, max_gens, run, runtime, out):
    if not os.path.exists('out/{}_{}'.format(pop_size, max_gens)):
        os.makedirs('out/{}_{}'.format(pop_size, max_gens))
    out.to_csv('out/{}_{}/solution-{}_{}-{}.csv'.format(pop_size, max_gens, pop_size, max_gens, run))
    print('CPU Runtime: {}.'.format(time.strftime('%H:%M:%S', time.gmtime(runtime))))


def process_end_metrics(function1_values, function2_values, fronts, start):
    neg_function1 = [-1 * function1_values[solution] for solution in fronts[0]]
    neg_function2 = [-1 * function2_values[solution] for solution in fronts[0]]
    runtime = time.time() - start
    return neg_function1, neg_function2, runtime


def crowding_distance_cut(temp_fronts, temp_crowding_distance_values, pop_size):
    new_pop = []
    for i in range(len(temp_fronts)):
        front_indeces = [index_of(temp_fronts[i][j], temp_fronts[i]) for j in range(len(temp_fronts[i]))]
        sorted_front_indeces = sort_by_values(front_indeces, temp_crowding_distance_values[i])
        sorted_front = [temp_fronts[i][sorted_front_indeces[j]] for j in range(len(temp_fronts[i]))]
        sorted_front.reverse()
        for value in sorted_front:
            if value not in new_pop:
                new_pop.append(value)
            if len(new_pop) == pop_size:
                break
        if len(new_pop) == pop_size:
            break
    return new_pop


def temporal_population_metrics(temp_pop, pop_size, features, function1_values, function2_values, gen, start, pool, ffp_features):
    objective_arguments = [[temp_pop[i], ffp_features] for i in range(pop_size, 2*pop_size)]
    fitness_values = list(pool.map(getFitness, objective_arguments))
    parallel_f1_values = [x[0] for x in fitness_values]
    parallel_f2_values = [x[1] for x in fitness_values]
    parallel_features = [x[2] for x in fitness_values]
    temp_function1_values = function1_values + parallel_f1_values
    temp_function2_values = function2_values + parallel_f2_values
    temp_features = features + parallel_features
    temp_fronts = fast_non_dominated_sort(temp_function1_values, temp_function2_values)
    crowding_distance_arguments = []
    for i in range(len(temp_fronts)):
        crowding_distance_arguments.append([temp_function1_values, temp_function2_values, temp_fronts[i]])  
    temp_crowding_distance_values = list(pool.map(crowding_distance, crowding_distance_arguments))
    temp_runtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print('Generation {}. Pareto Front Size: {}. Elapsed Time: {}'.format(gen, len(temp_fronts[0]), temp_runtime))
    return temp_fronts, temp_crowding_distance_values, temp_function1_values, temp_function2_values, temp_features


def evolutionary_process(features, max_gens, pop, pop_size, function1_values, function2_values, pool, start, ffp_features):
    gen = 1
    while gen <= max_gens:
        temp_pop = temporal_population_generator(pop, function1_values, function2_values, features, pool)
        temp_fronts, temp_crowding_distance_values, temp_function1_values, temp_function2_values, temp_features = temporal_population_metrics(temp_pop, pop_size, features, function1_values, function2_values, gen, start, pool, ffp_features)
        new_pop = crowding_distance_cut(temp_fronts, temp_crowding_distance_values, pop_size)
        pop = [temp_pop[i] for i in new_pop]
        function1_values = [temp_function1_values[i] for i in new_pop]
        function2_values = [temp_function2_values[i] for i in new_pop]
        features = [temp_features[i] for i in new_pop]
        gen = gen + 1
    return temp_function1_values, temp_function2_values, temp_features, temp_fronts, temp_pop


def runNSGA2(N=10, pop_size=100, max_gens=10, runs=10, features=[]):
    results_list = []

    for run in range(1, runs+1):
        print('================================== RUN {} ================================='.format(run))
        start = time.time()
        pool = multiprocessing.Pool()
        
        init_pop = initial_pop(N, pop_size, max_gens)
        objective_arguments = [[init_pop[i], features] for i in range(pop_size)]
        
        fitness_values = list(pool.map(getFitness, objective_arguments))
        init_f1 = [x[0] for x in fitness_values]
        init_f2 = [x[1] for x in fitness_values]
        init_features = [x[2] for x in fitness_values]

        last_f1_values, last_f2_values, last_features, last_fronts, last_temp_pop = evolutionary_process(init_features, max_gens, init_pop, pop_size, init_f1, init_f2, pool, start, features)
        neg_function1, neg_function2, runtime = process_end_metrics(last_f1_values, last_f2_values, last_fronts, start)
        out = DataFrame_preparation(N, last_fronts, len(last_fronts[0]), pop_size, neg_function1, neg_function2, runtime, max_gens, last_temp_pop, last_features)
        result_export(pop_size, max_gens, run, runtime, out)
        
        results_list.append((last_fronts, last_temp_pop, last_features))
    return results_list