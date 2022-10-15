import networkx as nx
import math
from scipy.stats import poisson
import random
import matplotlib.pyplot as plt
import numpy as np
import json
import os
#from netgraph import Graph
from collections import defaultdict
class TransitionStructure():
    """Wrapper class for Networkx multidigraph structure
    """
    def __init__(self):
        self.G = nx.MultiDiGraph()
    
    def add_node(self, val, currString = ""):
        """Class Method to add a node to the graph

        Args:
            val (string): name of the node
        """
        self.G.add_node(val, currString = currString)
    
    def add_edge(self, From, To, Label):
        """Class Method to add a edge to the graph

        Args:
            From (string): name of starting node
            To (end): name of ending node
        """
        self.G.add_edge(From, To, label=Label)

    def get_nodes(self):
        return list(self.G.nodes)

    def remove_node(self, node):
        """Class Method to remove a node from the graph

        Args:
            node (string): name of the node to be removed
        """
        self.G.remove_node(node)
    

    def remove_edge(self, From, To):
        """Class Method to remove a edge from the graph

        Args:
            edge (string): name of the edge to be removed
        """
        self.G.remove_edge(From, To)

    def num_nodes(self):
        """Class Method to count the number of nodes

        Returns:
            integer: number of nodes
        """
        return self.G.number_of_nodes()
    
    def num_edges(self):
        """Class Method to count the number of edges

        Returns:
            Integer: number of edges
        """
        return self.G.number_of_edges()
    
    def adjacency_list(self):
        """Class Method to return the adjacency list representation of the graph

        Returns:
            List of Lists: Adjacency List Representation of the graph
        """
        return self.G.adj
    
    def graph_info(self):
        """Class Method to return summary of the Graph

        Returns:
            String: Summary of the Graph
        """
        return nx.info(self.G) 

    def get_edge_info(self, From, To): # TODO: Get Edge info
        data = self.G.get_edge_data(From, To, default='DNE')
        #print(f"From: {From}")
        #print(f'To: {To}')
        #print(f"Data: {data}")
        if data == "DNE":
            return "DNE"
        
        return data
         

    def get_graph(self):
        """Class Method to return the Graph itself

        Returns:
            Object: MultiDiGraph
        """
        return self.G

    def draw_graph(self, filename='DFAs/TransitionStructure.png'):
        pos = nx.spring_layout(self.G)

        nx.draw(self.G, pos, with_labels=True, connectionstyle='arc3, rad = .2')
        edge_labels = dict([((u,v,),d['label'])
              for u,v,d in self.G.edges(data=True)])
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, label_pos=.3, font_size=7)
        plt.savefig(filename)
        plt.close()
    
    
    def relabel_nodes(self, mapping):
        self.G = nx.relabel_nodes(self.G, mapping)
        #print(list(self.G)) # check to make sure that we properly included data



class RandomTransitionStructure: #TODO: self.n, self.alphabet
    def __init__(self, n, alphabet):
        self.n = n # n is the number of states prior to removal of one state
        self.alphabet = alphabet
        self.k = len(alphabet)
        self.row = self.computeRow()
        self.end_states = {}
        self.TransitionStructure = None

    def generateRandomTransitionStructure(self, trial=0):
        row = self.computeRow() # part 1
        #print(row)
        if self.k < 2:
            return 'Error: k must be greater than 2'
        
        while (True):
            #print("in here")
            lambdaArray = [0] * self.n
            
            while sum(lambdaArray) != self.k * self.n + 1:
                lambdaArray = poisson.rvs(mu=row, size=self.n + 1, loc=1)
            
            partition = self.generatePartition(lambdaArray)
            for i in range(len(partition)):
                partition[i] = sorted(partition[i])
            
            partition = sorted(partition, key=lambda x: x[0])

            if (self.checkDyke(partition)):
                #print(f"chosen partition: {partition}")
                break
        
        TransitionStruct, order = self.partition_to_transition_structure(partition, self.alphabet, self.n) 
        self.TransitionStructure = self.compute_transition_structure(TransitionStruct, order, trial)
        
        
        


    def compute_transition_structure(self, TransitionStruct, order, trial=0):
        #print(f"order: {order}")
        vertexToRemove = order.pop(0) # removes first elem. Also check if this is even correct w/ turbo.
        #print(TransitionStruct.get_nodes())
        TransitionStruct.remove_node(vertexToRemove[0])
        determineEndStates = np.random.randint(2, size=self.n) # if 1, the index of that location is an end state, else 0
        #print(determineEndStates)


        for i, val in enumerate(determineEndStates):
            if val == 1:
                self.end_states[i + 1] = 'end_state'
            else:
                self.end_states[i + 1] = 'not_end_state'
        
        

        mapping = {}
        for i, k in enumerate(self.end_states.keys()):
            mapping[order[i][0]] = k
        
        #print(f"mapping: {mapping}")
        TransitionStruct.relabel_nodes(mapping)
        #self.n, self.alphabet

        
        # up to here
        return TransitionStruct

     
    def partition_to_transition_structure(self, kDyckPartition, alphabet, numStates):
        #print(kDyckPartition)
        stack = [] # empty stack
        TransStruct = TransitionStructure()
        initialState = 1
        TransStruct.add_node(initialState)
        order = [[1, 1]] # keeps track of order that Nodes get added into graph.
        newState = 2
        k = len(alphabet)
        n = numStates
        alphabet.sort(reverse=True)
        for a in alphabet: # should sort the list in reverse lexographical order
            stack.append((initialState, a))
        
        countPop = 0
        for i in range(2, k * (n + 1) + 2): 
            
            p, a = stack.pop()
            #print(stack)
            countPop += 1
            q = 1
            for j, partition in enumerate(kDyckPartition):
                if i in partition:
                    q = j + 1
        
            if q == newState:
                TransStruct.add_node(q) # create new state q
                order.append([q, newState]) # append the q, and new state
                newState += 1
                for b in alphabet:
                    stack.append((q, b))
                    
            #print(f"Edge getting added: {p} to {q} labeled {a}")
            TransStruct.add_edge(p,q,a) 

    
        return TransStruct, order # order stores node and state of node


    def checkDyke(self, partition):
        for j in range(len(partition)):
            if (min(partition[j]) > self.k * j + 1):
                return False
            
        return True

    def generatePartition(self, lambdaArray):
        #print(f"Lambda Array: {lambdaArray}")
        partition = []
        startVal = 1
        endingVal = 1
        
        for lambdaVal in lambdaArray:
            endingVal += lambdaVal
            currPart = []
            for i in range(startVal, endingVal):
                currPart.append(i)
            partition.append(currPart)
            startVal = endingVal
        #print(f"Partition: {partition}")
        bijection = self.generateRandomBijection()

        for part in partition:
            for index, elem in enumerate(part):
                part[index] = bijection[elem]  
        #print(f"bijected Partition: {partition}")
        for i in range(2, self.k):
            partition[0].append(i)

        #print(f"Add i and bijected Partition: {partition}")
        j = random.sample(range(0, self.n + 1), 1)[0]
        #print(f"Partition chosen: {j}")
        partition[j].append(self.k * (self.n + 1) + 1)

        return partition

    def generateRandomBijection(self):
        randomMapping = dict()
        randomCodomain = []
        randomCodomain.append(1)

        randomSample = random.sample(range(self.k + 1, self.k * (self.n + 1) + 1), self.n * self.k)
        for elem in randomSample:
            randomCodomain.append(elem)
        random.shuffle(randomCodomain)
        for i in range(self.k * self.n + 1):
            randomMapping[i + 1] = randomCodomain[i]
        
        return randomMapping

    def computeRow(self):
        ratio = self.k - ((self.k - 1) / (self.n + 1)) # f(k - (k - 1)/(n + 1))
        row = self.approximator(ratio)
        return row

    def approximator(self, ratio):
        z = -1 * ratio * math.exp(-1 * ratio) # -x * e^-x
        
        # sum from 1 to 100 of (-n)^(n - 1) / n! * z^n
        approximation = 0
        for n in range(1, 101):
            approximation += (((-1 * n)**(n - 1)) / (math.factorial(n))) *  z ** n

        approximation += ratio 
        return approximation


    def traverseTransitionStructure(self, edge_probability, end_state_probability):
        if self.TransitionStructure == None:
            print("Create a Transition Structure first!")
            return
    
    
    def get_transition_structure(self):
        if self.TransitionStructure == None:
            print("Create a Transition Structure First")
            return None
        
        return self.TransitionStructure


    #TODO: Some DFAs might not print any strings.
    # this is by design, and needs to be an edge case that gets hand
    def BFSString(self, m, trial=None, flag=False): # change to m -> breaking condition is that if I have a string of 2m + 1 -> infinite language
        
        if self.TransitionStructure == None:
            print("Create a Transition Structure First")
            return None
        
        print(self.end_states)
        A = nx.to_dict_of_lists(self.TransitionStructure.get_graph())
        
        strings = [] # list that will contain all possible strings up to a given length
        if all(value == 'not_end_state' for value in self.end_states.values()): # edge case when no end states
            return strings, [], False # might change to -1
        
        #print(A)
        start = 0 # index of starting node in the graph
        # self.checkCycle(self.TransitionStructure.get_graph())


        queue = [] # create a queue for BFS 
        

        state = list(A.keys())[start]
        #print(f"STATE: {state}")
        if self.end_states[state] == 'end_state': # appends the empty string if start state is also end state
            strings.append("")
        
        queue.append((state, ""))
        flag = False
        


        #currString = ""
        #visited = []
        # if and only if we can get into a cycle that contains at least 1 end state in it (good question to figure out probability)
        strlen = 0
        #print(f"end_states: {self.end_states}")

        # ISSUES: 
        # 1: No way to get to end state
        # Approach: 

        # HW: print out strings of size 11, if i have lengths of 6-11, i have infinite. PROVE THIS!
        all_strings = []
        while strlen < 2*m + 1:
            if len(queue) == 0:
                break
            
            state, currString = queue.pop(0)
            #

            for i, vertex in enumerate(A[state]):
                # if edge exists
                
                if self.TransitionStructure.get_edge_info(state, vertex) != 'DNE':
                    data = self.TransitionStructure.get_edge_info(state, vertex)
                    # print(f"STATE: {state}. VERTEX: {vertex}. data {data}")
                    for key in data.keys(): 
                        character = data[key]['label']

                        if self.end_states[vertex] == 'end_state':
                            strings.append(currString + character)
                            #print(currString + character)
                            #print(strings)
                        
                                                
                        strlen = len(currString + character)
                        #print(f"strlen: {strlen}")
                        #print(f"strings: {len(strings)}")
                        queue.append((vertex, currString + character))

        strings = list(set(strings))
        #print(strings)
        #print(strlen)

        isInfinite = False
        filtered = list(filter(lambda string: m < len(string) <= 2*m, strings))
        #print(filtered)
        if len(filtered) > 0:
            print("Infinite")
            isInfinite = True
        
        if not flag:
            DFAfp = ""
            filepath = ""
            if isInfinite:
                DFAfp = 'InfDFA/Final_Transition_Structure_' + str(self.n) + '_' + str(len(self.alphabet)) + '_' + str(trial) + '.png'
                filepath = 'InfDFA/Final_Transition_Structure_' + str(self.n) + '_' + str(len(self.alphabet)) + '_' + str(trial) + '.json' #TODO: LOOK HERE
            else:
                DFAfp = 'FinDFA/Final_Transition_Structure_' + str(self.n) + '_' + str(len(self.alphabet)) + '_' + str(trial) + '.png'
                filepath = 'FinDFA/Final_Transition_Structure_' + str(self.n) + '_' + str(len(self.alphabet)) + '_' + str(trial) + '.json' #TODO: LOOK HERE
        
            self.TransitionStructure.draw_graph(DFAfp)
        # File path for json file
            A = nx.to_dict_of_lists(self.TransitionStructure.get_graph())
            info = {
            "adj_List" : str(A),
            "end_states" : str(self.end_states),
            "nodes" : str(self.TransitionStructure.get_nodes()),
            "edges" : str(self.TransitionStructure.get_graph().edges(data='label')),
            "imgFP" : DFAfp,
            "isInfinite" : str(isInfinite)

            }
            with open(filepath, "w") as outfile:
                json.dump(info, outfile)
        elif flag == True:

            print('in here')
            test_img_fp = 'result.png'
            test_json_fp = 'result.json'
            info = {
            "adj_List" : str(A),
            "end_states" : str(self.end_states),
            "nodes" : str(self.TransitionStructure.get_nodes()),
            "edges" : str(self.TransitionStructure.get_graph().edges(data='label')),
            "imgFP" : DFAfp,
            "isInfinite" : str(isInfinite)

            }
            self.TransitionStructure.draw_graph(test_img_fp)
            with open(test_json_fp, "w") as outfile:
                json.dump(info, outfile)
            
        self.TransitionStructure.draw_graph('result.png')
        print(A)
        print(self.TransitionStructure.get_graph().edges(data='label'))
        return strings, filtered, isInfinite

    def debug(self):
        print(f'Nodes: {self.TransitionStructure.get_nodes()}')
        print(f"Edges: {self.TransitionStructure.get_graph().edges(data='label')}")
        
        

def main():
    # graphical library limitation
    tester = RandomTransitionStructure(3,['a', 'b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']) # clarify this with Turbo on Friday
    tester.generateRandomTransitionStructure()
    #tester.debug()
    strings, filtered, isInfinite = tester.BFSString(m=1, flag = True)
    print(strings)
    print(isInfinite)
if __name__ == "__main__":
    main()