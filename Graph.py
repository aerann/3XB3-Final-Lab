from abc import ABC, abstractmethod 
from math import radians, sin, cos, sqrt, atan2
import csv

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
    def __init__(self, n):
        self.adj = {}
        self.weights = {}
        for i in range(n):
            self.adj[i] = []
        
    def get_adj_nodes(self, n):
        return self.adj[n]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = 5 # calculate euclidian distance here

    def get_num_of_nodes(self):
        return len(self.adj)
    
    def w(self, node1, node2):
        present = True
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                present = True
            else: 
                present = False
        if present:
            return self.weights[(node1, node2)]

class HeuristicGraph(Graph):
    heuristic = {}

    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []
        
    def get_adj_nodes(self, n):
        return self.adj[n]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def get_num_of_nodes(self):
        return len(self.adj)
    
    def get_heuristic(self):
        return heuristic

def main():
    graph = {}
    stationsFile = 'london_stations.csv'
    connectionsFile = 'london_connections.csv'
    
    with open(stationsFile, 'r') as stations:
        stations_reader = csv.reader(stations)
        next(stations_reader)  # Skip header
        for row in stations_reader:
            station_id, latitude, longitude = map(int, row)
            graph[station_id] = {'latitude': latitude, 'longitude': longitude, 'neighbors': {}}

    # Read connections data from csv
    with open(connectionsFile, 'r') as connections:
        connections_reader = csv.reader(connections)
        next(connections_reader)  # Skip header
        for row in connections_reader:
            station1, station2 = map(int, row)
            distance = self.calculate_distance(graph[station1], graph[station2])
            graph[station1]['neighbors'][station2] = distance
            graph[station2]['neighbors'][station1] = distance


def calculate_distance(self, station1, station2):
    x1, y1 = radians(station1['latitude']), radians(station1['longitude'])
    x2, y2 = radians(station2['latitude']), radians(station2['longitude'])
    dlat = x2 - x1
    dlon = y2 - y1
    a = sin(dlat / 2)  ** 2 + cos(x1) * cos(x2) * sin(dlon / 2) **  2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    earth_radius = 6371.0
    distance = earth_radius * c
    return distance

    
    


