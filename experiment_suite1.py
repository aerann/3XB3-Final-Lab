from dijkstra import * 
from bellman_ford import * 
from final_project_part1 import * 
from matplotlib import pyplot as plt
import numpy as np
import timeit
from random import randrange 
#number of nodes (X) 
#number of edges (dense vs sparse) 
#k from 0 to v-1, compare which graph gets a closer result to shortest path? 

def create_custom_edge_graph(num_nodes, num_edges, upper):
    g = DirectedWeightedGraph()

    for i in range(num_nodes):
        g.add_node(i)

    edges_added = 0
    j = 1

    for i in range(num_nodes):
        while edges_added < num_edges:
            if i != j and not g.are_connected(i, j):
                g.add_edge(i, j, random.randint(1, upper))
                print("edges added", edges_added)
                edges_added += 1 
            j += 1 
    return g

    # while edges_added < num_edges:
    #     n1 = randrange(i)
    #     n2 = randrange(i)

    #     # Ensure n1 and n2 are distinct and the edge doesn't already exist
    #     if n1 != n2 and not g.are_connected(n1, n2):
    #         g.add_edge(n1, n2, random.randint(1,upper))
    #         edges_added += 1

    # return g
    # for i in range(num_nodes):
    #     for j in range(n):
    #         if i != j:
    #             G.add_edge(i,j,random.randint(1,upper))
    # return G


#First experiment: Seeing effect of increased number of nodes, with max weight of 25 and k = N - 1
def number_of_nodes_test(shortest_path_alg):
    xvalues = []
    yvalues = [] 
    for num_nodes in range(1, 30): 
        time = 0
        for _ in range(10): #average of 10 trials 
            g = create_random_complete_graph(num_nodes, 25)
            start = timeit.default_timer()
            shortest_path_alg(g, 0, g.number_of_nodes() - 1) 
            end = timeit.default_timer()
            time += end - start
        xvalues.append(num_nodes)
        yvalues.append(time / 10)
    return xvalues, yvalues 

#Second Experiment: Seeing the effect of number of edges (sparse vs dense graphs), with a fixed number of nodes 
def number_of_edges_test(shortest_path_alg):
    xvalues = []
    yvalues = [] 
    num_nodes = 30 
    for num_edges in range(num_nodes - 1, num_nodes*num_nodes): #sparse (edges = V) to dense (edges = V^2)
        time = 0
        print("edges to be added: ", num_edges)
        for _ in range(1): #average of 10 trials 
            g = create_custom_edge_graph(num_nodes, num_edges, 25)
            start = timeit.default_timer()
            shortest_path_alg(g, 0, g.number_of_nodes() - 1) #N-1 Relaxations 
            end = timeit.default_timer()
            time += end - start
        xvalues.append(num_nodes)
        yvalues.append(time / 10)
    return xvalues, yvalues 

#First experiment: 
# x1, y1 = number_of_nodes_test(dijkstra_approx)
# x2, y2 = number_of_nodes_test(bellman_ford_approx)
# xd, yd = np.array(x1), np.array(y1)
# xb, yb = np.array(x2), np.array(y2)

# plt.figure(1)
# plt.title("Number of Nodes vs Runtime")
# plt.xlabel("Number of Nodes")
# plt.ylabel("Runtime [s]")
# plt.plot(xd, yd, color='r', label = "Dijkstra's Algorithm")
# plt.plot(xb, yb, color='b', label = "Bellman Ford Algorithm")
# plt.legend()

# plt.show()

#Second Experiment: 
x1_2, y1_2 = number_of_edges_test(dijkstra_approx)
x2_2, y2_2 = number_of_edges_test(bellman_ford_approx)
xd_2, yd_2 = np.array(x1_2), np.array(y1_2)
xb_2, yb_2 = np.array(x2_2), np.array(y2_2)

# plt.figure(2)
plt.title("Number of Edges vs Runtime")
plt.xlabel("Number of Edges")
plt.ylabel("Runtime [s]")
plt.plot(xd_2, yd_2, color='r', label = "Dijkstra's Algorithm")
plt.plot(xb_2, yb_2, color='b', label = "Bellman Ford Algorithm")
plt.legend()

plt.show()
