from abc import ABC, abstractmethod 
from math import radians, sin, cos, sqrt, atan2
import csv
import min_heap2 as min_heap 
import timeit
import random
    
class WeightedGraph():
    def __init__(self):
        self.adj = {}
        self.weights = {}
        
    def get_adj_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, start, end, w):
        if start not in self.adj[end]:
            self.adj[start].append(end)
            self.adj[end].append(start)
        self.weights[(start, end)] = w 
        self.weights[(end, start)] = w 

    def get_num_of_nodes(self):
        return len(self.adj)

    def w(self, node1, node2):
        present = False
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                present = True
        if present == True:
            return self.weights[(node1, node2)]

def parse(): # can prob make stationphys global or smth in refactor
    stationPhys = {}
    stationsFile = 'london_stations.csv'
    connectionsFile = 'london_connections.csv'
    
    with open(stationsFile, 'r') as stations:
        stations_reader = csv.reader(stations)
        header = next(stations_reader) 
        idIdx = header.index('id')
        latIdx = header.index('latitude')
        lonIdx = header.index('longitude')

        for row in stations_reader:
            id = int(row[idIdx])
            lat = float(row[latIdx])
            lon = float(row[lonIdx])
            stationPhys[id] = {'latitude': lat, 'longitude': lon, 'neighbours': {}}

    with open(connectionsFile, 'r') as connections:
        connections_reader = csv.reader(connections)
        header = next(connections_reader)
        station1Idx = header.index('station1')
        station2Idx = header.index('station2')
        lineIdx = header.index('line')

        for row in connections_reader:
            station1 = int(row[station1Idx])
            station2 = int(row[station2Idx])
            line = int(row[lineIdx])
            distance = calculate_distance(stationPhys[station1], stationPhys[station2])
            stationPhys[station1]['neighbours'][station2] = [line, distance]
            stationPhys[station2]['neighbours'][station1] = [line, distance]
    return stationPhys

def calculate_distance(station1, station2):
    x1 = radians(station1['latitude'])
    y1 = station1['longitude']
    x2 = radians(station2['latitude'])
    y2 = station2['longitude']
    dlat = radians(x2 - x1)
    dlon = radians(y2 - y1)
    a = sin(dlat / 2.0) ** 2 + cos(x1) * cos(x2) * sin(dlon / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    earthRad = 6371000.0
    distance = earthRad * c
    return distance
    
def dijkstra2(G, source, d):
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
    return dist, pred

def heuristic(G, target, stations): # Example heuristic - assigns every node heuristic value of 5: result should be same as dijkstra's 
    hMap = {node: 0 for node in G.adj}
    for node in hMap:
        if node != target:
            distance = calculate_distance(stations[node], stations[target])
            hMap[node] = distance
    return hMap

def optimize(dist, neighbours, h):
    nodes = []
    distance = {}
    for i in neighbours:
        distance[i] = dist[i] + h[i]
    sortedDist = sorted(distance.items(), key=lambda distance: distance[1])
    for node in sortedDist:
        nodes.append(node[0])
    return nodes

def a_star2(G, s, d, h):
    pred = {}
    dist = {}
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(s, 0)

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        neighbours = G.adj[current_node]
        if current_node == d:
           break
        optimizedAdj = optimize(dist, neighbours, h)
        for neighbour in optimizedAdj:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist, pred

def buildGraph(stations):
    G = WeightedGraph()
    for station in list(stations.keys()):
        G.add_node(station)
    for station in G.adj:
        for connection in stations[station]['neighbours']:
            G.add_edge(station, connection, stations[station]['neighbours'][connection][1])
    return G
    
def main():
    time1 = 0
    time2 = 0
    # experiment 
    for _ in range(100): 
        stations = parse()
        G = buildGraph(stations)

        s = 1
        e = random.choice(list(stations.keys()))
        h = heuristic(G, e, stations) 

        start1 = timeit.default_timer()
        apathed = a_star2(G, s, e, h)
        end1 = timeit.default_timer()
        time1 += end1 - start1

        start2 = timeit.default_timer()
        dpathed = dijkstra2(G, s, e)
        end2 = timeit.default_timer()
        time2 += end2 - start2

    print('a star', time1)
    print('dijkstra', time2)

    for _ in range(100): 
        stations = parse()
        G = buildGraph(stations)

        s = 11
        e = 163
        h = heuristic(G, e, stations) 

        start1 = timeit.default_timer()
        apathed = a_star2(G, s, e, h)
        end1 = timeit.default_timer()
        time1 += end1 - start1

        start2 = timeit.default_timer()
        dpathed = dijkstra2(G, s, e)
        end2 = timeit.default_timer()
        time2 += end2 - start2

    print('a star', time1)
    print('dijkstra', time2)

main()