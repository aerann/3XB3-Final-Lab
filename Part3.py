from math import radians, sin, cos, sqrt, atan2
import csv
import min_heap2 as min_heap 
import timeit
import random
import numpy as np
from matplotlib import pyplot as plt
    
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
    return dist, pred
    
def dijkstra_modified(G, source, d):
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
        if current_node == d:  #modificaiton for same line
            break 
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

def a_star(G, s, d, h):
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
    
#EXPERIMENT SUITE 2 

#Experiment 1: All pairs
def all_pairs_graph(): 
    pair = 0 
    a_xvalues = []
    a_yvalues = []
    d_xvalues = []
    d_yvalues = []

    stations = parse()
    G = buildGraph(stations)

    for source in list(stations.keys()):
        time1 = 0
        time2 = 0

        for destination in list(stations.keys()):
            if source != destination: 
                start1 = timeit.default_timer()
                h = heuristic(G, destination, stations)  
                a_star(G, source, destination, h)
                end1 = timeit.default_timer()
                time1 += end1 - start1

        a_xvalues.append(pair)
        a_yvalues.append(time1)

        start2 = timeit.default_timer()
        dijkstra(G, source)
        end2 = timeit.default_timer()
        time2 += end2 - start2
        d_xvalues.append(pair)
        d_yvalues.append(time2)
        pair += 1

    return a_xvalues, a_yvalues, d_xvalues, d_yvalues


x1, y1, x2, y2 = all_pairs_graph()
xa, ya, xd, yd = np.array(x1), np.array(y1), np.array(x2), np.array(y2)

plt.figure(1)
plt.title("All Pairs Comparison")
plt.xlabel("Pair")
plt.ylabel("Runtime [s]")
plt.plot(xa, ya, color='r', label = "A* algorithm")
plt.plot(xd, yd, color='b', label = "Djikstras Algorithm")
plt.legend()

plt.show()


#Experiment 2: Same line (line 1), not directly connected 
def same_line_indirect():
    a_xvalues = []
    a_yvalues = []
    d_xvalues = []
    d_yvalues = []

    stations = parse()
    G = buildGraph(stations)

    line1_stations = []

    #get all stations on line 1
    for station in list(stations.keys()): 
        station_neighbours = stations[station]['neighbours']
        for neighbour in station_neighbours: 
            if stations[station]['neighbours'][neighbour][0] == 1:
                line1_stations.append(station)
                line1_stations.append(neighbour)

    pair = 0 

    for source in line1_stations: 
        for dest in line1_stations:
            if source != dest:
                time1 = 0 
                time2 = 0  
                h = heuristic(G, dest, stations)  
                start1 = timeit.default_timer()
                a_star(G, source, dest, h) 
                end1 = timeit.default_timer()
                time1 += end1 - start1

                a_xvalues.append(pair)
                a_yvalues.append(time1) 

                start2 = timeit.default_timer()
                dijkstra_modified(G, source, dest)
                end2 = timeit.default_timer()
                time2 += end2 - start2

                d_yvalues.append(time2)
                d_xvalues.append(pair)
                pair += 1

    return a_xvalues, a_yvalues, d_xvalues, d_yvalues

x1_sli, y1_sli, x2_sli, y2_sli = same_line_indirect()
xsli, ysli, xd_sli, yd_sli = np.array(x1_sli), np.array(y1_sli), np.array(x2_sli), np.array(y2_sli)

plt.figure(2)
plt.title("Shortest Path on Same Line Comparison")
plt.xlabel("Trial")
plt.ylabel("Runtime [s]")
plt.plot(xsli, ysli, color='r', label = "A* algorithm")
plt.plot(xd_sli, yd_sli, color='b', label = "Djikstras Algorithm")
plt.legend()
plt.show()


#Find path of stations from start to end by djikstra
def find_path(g, start, end): 
    path = [] 
    path.append(end)
    pred = end 
    target = end
    while pred != start:
        pred = dijkstra(g, start)[1][target]
        target = pred 
        path.insert(0, pred)
    return path 

#Count number of transfers
def count_transfers(station_list, path): 
    transfer_path = set()
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        prev = station_list[p1]['neighbours'][p2][0] #get the line connected by p1 and p2 
        transfer_path.add(prev)
    return len(transfer_path), list(transfer_path)

#Experiment 3: Stations are not on the same line, need to do several transfers
def transfers(): 
    a_xvalues = []
    a_yvalues = []
    d_xvalues = []
    d_yvalues = []

    stations = parse()
    G = buildGraph(stations)

    line1_stations = []
    notline1_stations = []

    #get all stations on line 1
    for station in list(stations.keys()): 
        station_neighbours = stations[station]['neighbours']
        for neighbour in station_neighbours: 
            if stations[station]['neighbours'][neighbour][0] == 1:
                line1_stations.append(station)
                line1_stations.append(neighbour)

    #get stations that are not on line 1 
    for station in list(stations.keys()): 
        onLineOne = False
        station_neighbours = stations[station]['neighbours']
        for neighbour in station_neighbours: 
            if stations[station]['neighbours'][neighbour][0] == 1:
                onLineOne = True 
                break 
        if not onLineOne: 
            notline1_stations.append(station)

    numTransfers = 0
    target_transfer = 1 
    while numTransfers < 5: 
            #until we get a number of transfer value where the runtime hasn't been evaluated yet
            time1 = 0 
            time2 = 0 
            for _ in range(10): 
                while numTransfers != target_transfer:
                    notline1_station = random.choice(notline1_stations)
                    line1_station = random.choice(line1_stations)
                    path = find_path(G, line1_station, notline1_station)
                    numTransfers = count_transfers(stations, path)[0]

                h = heuristic(G, notline1_station, stations) 
                start1 = timeit.default_timer()
                a_star(G, line1_station, notline1_station, h) #have to transfer, on diff lines 
                end1 = timeit.default_timer()
                time1 += end1 - start1 

                start2 = timeit.default_timer()
                dijkstra(G, line1_station) #have to transfer, on diff lines 
                end2 = timeit.default_timer()
                time2 += end2 - start2

            a_xvalues.append(numTransfers)
            a_yvalues.append(time1/10)

            d_xvalues.append(numTransfers)
            d_yvalues.append(time2/10)

            target_transfer += 1

    return a_xvalues, a_yvalues, d_xvalues, d_yvalues

# x1_t, y1_t, x2_t, y2_t = transfers()
# xt, yt, xd_t, yd_t = np.array(x1_t), np.array(y1_t), np.array(x2_t), np.array(y2_t)

# plt.figure(3)
# plt.title("Paths with multiple transfers comparison")
# plt.xlabel("Number of Transfers")
# plt.ylabel("Runtime [s]")
# plt.plot(xt, yt, color='r', label = "A* algorithm")
# plt.plot(xd_t, yd_t, color='b', label = "Djikstras Algorithm")
# plt.legend()
# plt.show()

def are_lines_adjacent(stations, line_a, line_b):
    # Collect stations for each line
    line_a_stations = []
    line_b_stations = []

    #get all stations on line 1
    for station in list(stations.keys()): 
        station_neighbours = stations[station]['neighbours']
        for neighbour in station_neighbours: 
            if stations[station]['neighbours'][neighbour][0] == line_a:
                line_a_stations.append(station)
                line_a_stations.append(neighbour)

    for station in list(stations.keys()): 
        station_neighbours = stations[station]['neighbours']
        for neighbour in station_neighbours: 
            if stations[station]['neighbours'][neighbour][0] == line_b:
                line_b_stations.append(station)
                line_b_stations.append(neighbour)

    # Check for adjacency
    # Both lines contain at least one of the same station
    for station in line_a_stations:
        for station2 in line_b_stations: 
            if station == station2: 
                return True  # Found a connection between the two lines
    return False

#Experiment 4: Stations are on adjacent lines
def adjacent_lines(): 
    a_xvalues = []
    a_yvalues = []
    d_xvalues = []
    d_yvalues = []

    stations = parse()
    pair = 0 
    G = buildGraph(stations)
    for station in list(stations.keys()):
        if pair > 15: 
            break
        for station2 in list(stations.keys()):
            if station != station2:
                if pair > 15: 
                    break
                path = find_path(G, station, station2)
                lines = count_transfers(stations, path)[1]
                is_adjacent = are_lines_adjacent(stations, lines[0], lines[-1])
                if is_adjacent: 
                    h = heuristic(G, station2, stations) 
                    start1 = timeit.default_timer()
                    a_star(G, station, station2, h) 
                    end1 = timeit.default_timer()
                    time1 = end1 - start1 

                    start2 = timeit.default_timer()
                    dijkstra(G, station) 
                    end2 = timeit.default_timer()
                    time2 = end2 - start2

                    a_xvalues.append(pair)
                    a_yvalues.append(time1)

                    d_xvalues.append(pair)
                    d_yvalues.append(time2)
                    pair += 1
        
    return a_xvalues, a_yvalues, d_xvalues, d_yvalues


# x1_adj, y1_adj, x2_adj, y2_adj = adjacent_lines()
# xadj, yadj, xd_adj, yd_adj = np.array(x1_adj), np.array(y1_adj), np.array(x2_adj), np.array(y2_adj)

# plt.figure(4)
# plt.title("Adjacent Line Test")
# plt.xlabel("Trial")
# plt.ylabel("Runtime [s]")
# plt.plot(xadj, yadj, color='r', label = "A* algorithm")
# plt.plot(xd_adj, yd_adj, color='b', label = "Djikstras Algorithm")
# plt.legend()
# plt.show()