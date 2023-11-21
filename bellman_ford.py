from final_project_part1 import *

def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relaxed = {} #Dictionary of the nodes and the number of times each node has been relaxed 
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
        relaxed[node] = 0 
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()): 
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and relaxed[neighbour] < k:
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
                    relaxed[neighbour] += 1
    return dist

#Testing
# myGraph = DirectedWeightedGraph()
# g = DirectedWeightedGraph() 
# for i in range(5):
#     g.add_node(i)

#:)
# g.add_edge(0, 1, -1)
# g.add_edge(0, 2, 4)
# g.add_edge(1, 2, 3)
# g.add_edge(3, 1, 1)
# g.add_edge(3, 2, 5)
# g.add_edge(1, 3, 2)
# g.add_edge(1, 4, 2)
# g.add_edge(4, 3, -3)

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

# print('graph', myGraph.adj)
# print('graph after', bellman_ford(myGraph, 0, 2))

# print('graph', g.adj)
# print('graph after', bellman_ford_approx(g, 0, 2))







