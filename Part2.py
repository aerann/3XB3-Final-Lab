from final_project_part1 import *
from random import *

def buildH(G, s): # Example heuristic - assigns every node heuristic value of 5: result should be same as dijkstra's 
    hMap = {node: 0 for node in G.adj}
    for node in hMap:
        # Heuristic function to give certain nodes precedence
        current = hMap.get(node) 
        hMap[node] = current + 5
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

def dijTarget(G, source, target):
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

def main():
    G = create_random_complete_graph(5, 10)
    print(G.adj)
    s = randint(0, G.number_of_nodes())
    h = buildH(G, s) # placeholder heuristic
    apathed = a_star(G, 0, 1, h)
    dpathed = dijTarget(G, 0, 1)
    print(apathed[1])
    print(dpathed[1])
main()