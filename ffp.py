"""
Authors:
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

import random
import math

from gen import *

from ai_models import *

from os.path import exists

from keras.models import load_model

import joblib

NN_MODEL_PATH = 'neuralgen.h5'
DT_MODEL_PATH = 'decision_tree_model_joblib.pickle'
NOTHING_HEURISTIC = 'Nothing'

# Provides the methods to create and solve the firefighter problem
class FFP:
  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")    
    text = file.read()    
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))  
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1      
    self.graph = []    
    for i in range(self.n):
      self.graph.append([0] * self.n);
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1

    self.iteration = None
    self.whole_state = []
    self.state_features = []

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))    
    self.iteration = 0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):        
        heuristic = method
        if (isinstance(method, HyperHeuristic)):
          self.iteration += 1
          heuristic = method.nextHeuristic(self)
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected   
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(self.iteration) + ": A firefighter protects node " + str(node))            
      # It spreads the fire among the unprotected nodes
      spreading = False 
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1): 
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(self.iteration) + ": Fire spreads to node " + str(j))     
      self.iteration += 1
      ffp_features = ["EDGE_DENSITY", "AVG_DEGREE", "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
      self.state_features.append([self.getFeature(ffp_features[0]), self.getFeature(ffp_features[1]), self.getFeature(ffp_features[2]), self.getFeature(ffp_features[3]), self.getFeature(ffp_features[4])])
      self.whole_state.append(self.state_features)
      if (debug):
        print("---------------")
    if (debug):    
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def __nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i        
        break
    value = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])              
              break
        elif (heuristic == "GDEG"):        
          value = sum(self.graph[i])   
        elif (heuristic == NOTHING_HEURISTIC):
          value = best    
        else:
          print("=====================")
          print("Critical error at FFP.__nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")          
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)      
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph) 
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph) 
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))    
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:      
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")          
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
class HyperHeuristic:

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  def __init__(self, features, heuristics):
    if (features):
      self.features = features.copy()
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of features cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
    if (heuristics):
      self.heuristics = heuristics.copy()
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of heuristics cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    print("=====================")
    print("Critical error at HyperHeuristic.nextHeuristic.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

  # Returns the string representation of this hyper-heuristic 
  def __str__(self):
    print("=====================")
    print("Critical error at HyperHeuristic.__str__.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires
class DummyHyperHeuristic(HyperHeuristic):

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  #   nbRules = The number of rules to be contained in this hyper-heuristic  
  def __init__(self, features, heuristics, nbRules, seed):
    super().__init__(features, heuristics)
    random.seed(seed)
    self.conditions = [] # Initial state of the features
    self.actions = [] # Sequence of heuristics to apply
    for i in range(nbRules):
      self.conditions.append([0] * len(features))
      for j in range(len(features)):
        self.conditions[i][j] = random.random()
      self.actions.append(heuristics[random.randint(0, len(heuristics) - 1)])
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    minDistance = float("inf")
    index = -1
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    print("\t State:" + str(state))
    print("\t Conditions:" + str(self.conditions))
    for i in range(len(self.conditions)):
      distance = self.__distance(self.conditions[i], state)      
      if (distance < minDistance):
        minDistance = distance
        index = i
    heuristic = self.actions[index]
    print("\t Actions:" + str(self.actions))
    print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
    return heuristic

  # Returns the string representation of this dummy hyper-heuristic
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
    for i in range(len(self.conditions)):      
      text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"      
    return text

  # Returns the Euclidian distance between two vectors
  def __distance(self, vectorA, vectorB):
    distance = 0
    for i in range(len(vectorA)):
      distance += (vectorA[i] - vectorB[i]) ** 2
    distance = math.sqrt(distance)
    return distance

class GeneticHyperHeuristic(HyperHeuristic):
  def __init__(self, features, heuristics, chromosomes_size, pop_size, max_gens, runs, model_name="DT"):
    super().__init__(features, heuristics)

    self.model_name = model_name

    trained_model = (exists(NN_MODEL_PATH) or exists(DT_MODEL_PATH))
    if not trained_model:
      input = runNSGA2(chromosomes_size, pop_size, max_gens, runs, features)
      output = []
      
      self.parse_classes = {}
      for i, value in enumerate(heuristics):
        self.parse_classes[value] = i

      print("\t Classification classes:" + str(self.parse_classes))

      for i in range(len(input)):
        partial_result = input[i]

        last_fronts = partial_result[0]
        last_temp_pop = partial_result[1]
        last_features = partial_result[2]

        for solution in last_fronts[0]:
          for gene in range(len(last_temp_pop[solution])):
            new_row = last_features[solution][gene]

            if new_row == ['Z']:
              continue
            
            new_row.append(self.parse_classes[str(last_temp_pop[solution][gene])])
            output.append(new_row)

      self.conditions = []
      self.actions = []
      rows = len(output)
      for i in range(rows):
        self.conditions.append(output[i][:-1])
        self.actions.append(output[i][-1])
      
      if model_name == "NN":
        self.model = train_neural_network(self.conditions, self.actions, 1, len(features))
      else:
        self.model = train_decision_tree(self.conditions, self.actions)
    else:
      self.conditions = "Already trained"
      self.actions = "Already trained"
      
      self.parse_classes = {
        "LDEG": 0,
        "GDEG": 1,
      }

      if model_name == "NN":
        self.model = load_model(NN_MODEL_PATH)
      else:
        self.model = joblib.load(DT_MODEL_PATH)
        
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    print("\t State:" + str(state))

    if(self.model_name == "NN"):
      predictions = self.model.predict(np.array(state).reshape(1, -1))
      predicted_class = int(predictions[np.argmax(predictions[0])][0])
    else:
      prediction = self.model.predict(np.array(state).reshape(1, -1))
      predicted_class = prediction[0]

    keys = list(self.parse_classes.keys())
    heuristic = keys[predicted_class]
    print("Selected heuristic:" + heuristic)
    
    return heuristic

  # Returns the string representation of this dummy hyper-heuristic
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
    if isinstance(self.conditions, str):
      text += "\t" + self.conditions + "\n" " => " + self.actions + "\n"
    else:
      for i in range(len(self.conditions)):      
        text += "\t" + str(self.conditions[i]) + " => " + str(self.actions[i]) + "\n"
    return text

class GeneticAux(HyperHeuristic):
  def __init__(self, chromosome):
    #super().__init__(features, heuristics)
    self.chromosome = chromosome
  
  def nextHeuristic(self, problem):
    if problem.iteration >= len(self.chromosome):
      return NOTHING_HEURISTIC

    return self.chromosome[problem.iteration]

# Tests
# =====================
if __name__ == '__main__':
  features = ["EDGE_DENSITY", "AVG_DEGREE", "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
  heuristics = ["LDEG", "GDEG"]
  firefighters = 1

  instances = []
  ldeg = []
  gdeg = []
  dummy = []
  custom = []

  model_name = "DT"

  for filename in glob.glob("./instances/GBRL/*"):
    problem = FFP(filename)

    #custom_hh = GeneticHyperHeuristic(features, heuristics, chromosomes_size=25, pop_size=50, max_gens=50, runs=20)
    custom_hh = GeneticHyperHeuristic(features, heuristics, chromosomes_size=50, pop_size=20, max_gens=20, runs=10, model_name=model_name)
    #custom_hh = GeneticHyperHeuristic(features, heuristics, chromosomes_size=30, pop_size=10, max_gens=10, runs=2, model_name=model_name)
    print(custom_hh)
    custom_hh_res = problem.solve(custom_hh, firefighters, False)
    print("Custom HyperHeuristic = " + str(custom_hh_res))

    seed = random.randint(0, 1000)
    print('Seed:', seed)
    hh = DummyHyperHeuristic(features, heuristics, 1, seed)
    print(hh)

    dummy_hh_res = problem.solve(hh, 1, False)
    print("Dummy HH = " + str(dummy_hh_res))

    ldeg_res = problem.solve("LDEG", 1, False)
    print("LDEG = " + str(ldeg_res))

    gdeg_res = problem.solve("GDEG", 1, False)
    print("GDEG = " + str(gdeg_res))

    instances.append(filename.split("/")[-1])
    ldeg.append(ldeg_res)
    gdeg.append(gdeg_res)
    dummy.append(dummy_hh_res)
    custom.append(custom_hh_res)

  for filename in glob.glob("./instances/BBGRL/*"):
    problem = FFP(filename)

    custom_hh = GeneticHyperHeuristic(features, heuristics, chromosomes_size=50, pop_size=20, max_gens=20, runs=10, model_name=model_name)
    print(custom_hh)
    custom_hh_res = problem.solve(custom_hh, firefighters, False)
    print("Custom HyperHeuristic = " + str(custom_hh_res))

    seed = random.randint(0, 1000)
    print('Seed:', seed)
    hh = DummyHyperHeuristic(features, heuristics, 1, seed)
    print(hh)

    dummy_hh_res = problem.solve(hh, 1, False)
    print("Dummy HH = " + str(dummy_hh_res))

    ldeg_res = problem.solve("LDEG", 1, False)
    print("LDEG = " + str(ldeg_res))

    gdeg_res = problem.solve("GDEG", 1, False)
    print("GDEG = " + str(gdeg_res))

    instances.append(filename.split("/")[-1])
    ldeg.append(ldeg_res)
    gdeg.append(gdeg_res)
    dummy.append(dummy_hh_res)
    custom.append(custom_hh_res)
  
  df = {
    "Instance": instances,
    "LDEG": ldeg,
    "GDEG": gdeg,
    "Dummy-HyperHeuristic": dummy,
    "Genetic-HyperHeuristic": custom,
  }
  df = pd.DataFrame(df)
  df.to_csv("results.csv", index=False)