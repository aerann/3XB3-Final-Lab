from part1 import *
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
        node_path = optimize(dist, neighbours, h)
        for neighbour in node_path:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist, pred

def main():
    G = create_random_complete_graph(5, 10)
    print(G.adj)
    s = randint(0, G.number_of_nodes())
    h = buildH(G, s) # placeholder heuristic
    apathed = a_star(G, 0, 1, h)
    dpathed = dijkstra(G, 0)
    print(apathed[0], apathed[1])
main()