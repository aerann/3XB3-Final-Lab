import min_heap2 as min_heap 
import random
import timeit 
import numpy as np
import math
from matplotlib import pyplot as plt
# from scipy.stats import linregress

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([]) 
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf") 
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min() #Originally gonna be the starting node
        current_node = current_element.value
        dist[current_node] = current_element.key #key is originally infinity for every node except starting node. after it represents distance from the starting node 
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def bellman_ford(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total

def create_random_complete_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i,j,random.randint(1,upper))
    return G


#Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d


#Positive edge graph for mystery function 
myGraph = DirectedWeightedGraph()
for i in range(4):
    myGraph.add_node(i)


myGraph.add_edge(0, 2, 7)
myGraph.add_edge(2, 0, 7)

myGraph.add_edge(0, 3, 100)
myGraph.add_edge(3, 0, 100)

myGraph.add_edge(1, 3, 1)
myGraph.add_edge(3, 1, 1)

myGraph.add_edge(1, 2, 8)
myGraph.add_edge(2, 1, 8)

myGraph.add_edge(0, 1, 5)
myGraph.add_edge(1, 0, 5)

#Negative Graph - Testing for Mystery Graph
myGraph2 = DirectedWeightedGraph()
for i in range(3):
    myGraph2.add_node(i)

myGraph2.add_edge(0, 1, 5)
myGraph2.add_edge(0, 2, 6)
myGraph2.add_edge(2, 1, -3)

#Negative cycle graph - Testing for Mystery Graph
myGraph3 = DirectedWeightedGraph()
for i in range(3):
    myGraph3.add_node(i)

myGraph3.add_edge(0, 2, 6)
myGraph3.add_edge(2, 1, -15)
myGraph3.add_edge(1, 0, 5)


#Log log graph, computing the runtime of the mystery function as the number of nodes in a graph increases
def mystery_algorithm_test(mystery):
    xvalues = []
    yvalues = [] 
    for num_nodes in range(1, 100): 
        time = 0
        for _ in range(10): #average of 10 trials 
            g = create_random_complete_graph(num_nodes, 25)
            start = timeit.default_timer()
            mystery(g) 
            end = timeit.default_timer()
            time += end - start
        xvalues.append(num_nodes)
        yvalues.append(time / 10) 
    return xvalues, yvalues 

#Plotting runtime curve of mystery function
x1, y1 = mystery_algorithm_test(mystery)
x_log, y_log = np.log(x1), np.log(y1)
plt.title("Number of Nodes vs Runtime")
plt.xlabel("Number of Nodes")
plt.ylabel("Runtime [s]")
plt.loglog(x1, y1, color='r', label = "Mystery Algorithm")  

#Plotting the line of best fit
# slope, intercept, r_value, p_value, std_err = linregress(x_log, y_log)
# fit_line = np.exp(intercept) * x1**slope 
# plt.loglog(x1, fit_line, linestyle='--', color = 'b', label=f'Fit (Slope - {slope: .2f})')

# plt.legend()

# plt.show()
