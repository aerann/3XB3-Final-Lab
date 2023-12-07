
from Part1 import * 
from matplotlib import pyplot as plt
import numpy as np
import timeit
from random import randrange 

def create_custom_edge_graph(num_nodes, max_edges, upper):
    g = DirectedWeightedGraph()

    for i in range(num_nodes):
        g.add_node(i)

    edges_added = 0 

    for i in range(num_nodes):
        j = 0 
        while j < num_nodes and edges_added < max_edges:
            if i != j:
                g.add_edge(i,j, random.randint(1,upper))
                edges_added += 1 
            j += 1 
    return g


#First experiment: Seeing effect of increased number of nodes between Djikstras and Bellman Ford 
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
    v = 30
    for num_edges in range(v, v*v): #sparse (edges = V) to dense (edges = V^2)
        time = 0
        for _ in range(10): #average of 10 trials 
            g = create_custom_edge_graph(v, num_edges, 10)
            start = timeit.default_timer()
            shortest_path_alg(g, 0, g.number_of_nodes() - 1) #graph, source, N-1 Relaxations 
            end = timeit.default_timer()
            time += end - start
        xvalues.append(num_edges)
        yvalues.append(time / 10)
    return xvalues, yvalues 


def distance_test(shortest_path_alg_approx, shortest_path_alg):
    xvalues = []
    yvalues = [] 
    v = 30
 
    for k in range(0, 10): 
        totDiff = 0  
        for _ in range(10):
            g = create_random_complete_graph(v, 100)
            dist_approx = shortest_path_alg_approx(g, 0, k)
            dist = shortest_path_alg(g, 0)
            totDiff += abs(total_dist(dist_approx) - total_dist(dist))
        xvalues.append(k)
        yvalues.append(totDiff/10)
    return xvalues, yvalues 


#First experiment: 
x1, y1 = number_of_nodes_test(dijkstra_approx)
x2, y2 = number_of_nodes_test(bellman_ford_approx)
xd, yd = np.array(x1), np.array(y1)
xb, yb = np.array(x2), np.array(y2)

plt.figure(1)
plt.title("Number of Nodes vs Runtime")
plt.xlabel("Number of Nodes")
plt.ylabel("Runtime [s]")
plt.plot(xd, yd, color='r', label = "Dijkstra's Algorithm")
plt.plot(xb, yb, color='b', label = "Bellman Ford Algorithm")
plt.legend()

plt.show()

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

# Third Experiment (Djikstra's)
x1_3, y1_3 = distance_test(dijkstra_approx, dijkstra)
xd_3, yd_3 = np.array(x1_3), np.array(y1_3)

plt.figure(3)
plt.title("Number of Relaxations vs Difference of Distance")
plt.xlabel("Number of Relaxations")
plt.ylabel("Difference of Distance")
plt.plot(xd_3, yd_3, color='r', label = "Dijkstra's Algorithm")
plt.legend()

plt.show()


#Fourth Experiment (Bellman Ford)
x1_4, y1_4 = distance_test(dijkstra_approx, dijkstra)
xb_4, yb_4 = np.array(x1_4), np.array(y1_4)

plt.figure(4)
plt.title("Number of Relaxations vs Difference of Distance")
plt.xlabel("Number of Relaxations")
plt.ylabel("Difference of Distance")
plt.plot(xb_4, yb_4, color='b', label = "Bellman Ford Algorithm")
plt.legend()

plt.show()