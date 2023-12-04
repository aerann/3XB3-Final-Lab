from final_project_part1 import *

def dijkstra_approx(G, source, k): 
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relaxed = {} #Dictionary of the nodes and the number of times each node has been relaxed 

    Q = min_heap.MinHeap([]) 
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf") 
        relaxed[node] = 0 
    Q.decrease_key(source, 0)

    while not Q.is_empty():
        current_element = Q.extract_min() #Originally gonna be the starting node, after it picks the node with the smallest key 
        current_node = current_element.value
        dist[current_node] = current_element.key #key is originally infinity for every node except starting node. after it represents distance from the starting node 
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and relaxed[neighbour] < k:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
                relaxed[neighbour] += 1
    return dist


#Testing
# myGraph = create_random_complete_graph(20, 100)

# myGraph = DirectedWeightedGraph()
# for i in range(4):
#     myGraph.add_node(i)

# myGraph.add_edge(0, 1, 5)
# myGraph.add_edge(1, 0, 5)

# myGraph.add_edge(0, 2, 7)
# myGraph.add_edge(2, 0, 7)

# myGraph.add_edge(0, 3, 100)
# myGraph.add_edge(3, 0, 100)

# myGraph.add_edge(1, 3, 1)
# myGraph.add_edge(3, 1, 1)

# myGraph.add_edge(1, 2, 8)
# myGraph.add_edge(2, 1, 8)

# myGraph = create_random_complete_graph(20, 100)
# print('graph', myGraph.adj)
# # print('graph after', dijkstra_approx(myGraph, 0, 5))

# relax = 18
# sup = dijkstra_approx(myGraph, 0, relax)
# print("graph after relaxing", relax, "times:", sup)
# sum = 0 

# for i in range(len(sup)): 
#     sum += sup[i] 

# print("sum", sum)