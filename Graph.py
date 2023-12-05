from abc import ABC, abstractmethod 
from math import radians, sin, cos, sqrt, atan2
import csv
import min_heap2 as min_heap 
import random
import timeit 
import numpy as np
# from A_Star import *

class Graph(ABC):
    @abstractmethod
    def __init__(self, n):
        pass

    @abstractmethod
    def get_adj_nodes(self, n):
        pass

    @abstractmethod
    def add_node(self):
        pass

    @abstractmethod
    def add_edge(self, node1, node2):
        pass

    @abstractmethod
    def get_num_of_nodes(self):
        pass
    
class WeightedGraph(Graph):
    def __init__(self):
        self.adj = {}
        self.weights = {}
        
    def get_adj_nodes(self, n):
        return self.adj[n]

    def add_node(self, n):
        self.adj[n] = []

    def add_edge(self, node1, node2, w):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)
        self.weights[(node1, node2)] = w 

    def get_num_of_nodes(self):
        return len(self.adj)

    def w(self, node1, node2):
        present = False
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                present = True
        if present == True:
            return self.weights[(node1, node2)]

def buildGraph():
    graphStructure = {}
    stationsFile = 'london_stations.csv'
    connectionsFile = 'london_connections.csv'
    
    with open(stationsFile, 'r') as stations:
        stations_reader = csv.reader(stations)
        header = next(stations_reader) 
        idIdx = header.index('id')
        latIdx = header.index('latitude')
        lonIdx = header.index('longitude')

        for row in stations_reader:
            id = row[idIdx]
            lat = row[latIdx]
            lon = row[lonIdx]
            graphStructure[id] = {'latitude': lat, 'longitude': lon, 'neighbors': {}}
            
    G = WeightedGraph()
    print(len(graphStructure))
    for station in list(graphStructure.keys()):
        G.add_node(station)

    with open(connectionsFile, 'r') as connections:
        connections_reader = csv.reader(connections)
        header = next(connections_reader)
        station1Idx = header.index('station1')
        station2Idx = header.index('station2')

        for row in connections_reader:
            station1 = row[station1Idx]
            station2 = row[station2Idx]
            distance = calculate_distance(graphStructure[station1], graphStructure[station2])
            # graphStructure[station1]['neighbors'][station2] = distance
            # graphStructure[station2]['neighbors'][station1] = distance 
            G.add_edge(station1, station2, distance)
    
    return G

def calculate_distance(station1, station2):
    x1, y1 = radians(float(station1['latitude'])), radians(float(station1['longitude']))
    x2, y2 = radians(float(station2['latitude'])), radians(float(station2['longitude']))
    dlat = x2 - x1
    dlon = y2 - y1
    a = sin(dlat / 2) ** 2 + cos(x1) * cos(x2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    earth_radius = 6371.0
    distance = earth_radius * c
    return distance
    
def dijkstra(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([]) 
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        print(node)
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf") 
    print(Q)
    print(source)
    print(0)
    Q.decrease_key(source, 0)
    print('do we get here')

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
                print('here!')
    return dist, pred

def main():
    G = buildGraph()
    # print(G.adj)
    #h = buildH(G, s) # placeholder heuristic
    #pathed = a_star(G, 0, 1, h)
    dpathed = dijkstra(G, 1)
    print(dpathed[0], dpathed[1])

main()