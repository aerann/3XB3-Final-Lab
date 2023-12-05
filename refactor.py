from abc import ABC, abstractmethod 
import min_heap2 as min_heap 

#Graph interface
class Graph(ABC):
    @abstractmethod
    def get_adj_notes(node: int) -> list[int]:
        pass

    @abstractmethod
    def add_node(node: int):
        pass

    @abstractmethod
    def add_edge(start: int, end: int):
        pass

    @abstractmethod 
    def get_num_of_nodes() -> int:
        pass

class WeightedGraph(Graph): #implements from graph
    def __init__(self, node):
        self.adj = {}
        self.weights = {}
        for i in range(node):
            self.adj[i] = []
        
    def get_adj_nodes(self, node: int) -> list[int]:
        return self.adj[node]

    def add_node(self, node: int):
        self.adj[node] = []

    def add_edge(self, start: int, end: int):
        if end not in self.adj[start]:
            self.adj[start].append(end)
        self.weights[(start, end)] = 5 # calculate euclidian distance here

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
            return float(self.weights[(node1, node2)])

#SPAlgorithm Interface
class SPAlgorithm(ABC):
    @abstractmethod
    def calc_sp(graph: Graph, source: int, dest: int) -> float:
        pass

class dijkstra(SPAlgorithm):
    def calc_sp(graph: Graph, source: int, dest: int) -> float:
        pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {} #Distance dictionary
        Q = min_heap.MinHeap([]) 
        nodes = list(graph.adj.keys())

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
            for neighbour in graph.adj[current_node]:
                if dist[current_node] + graph.w(current_node, neighbour) < dist[neighbour]:
                    Q.decrease_key(neighbour, dist[current_node] + graph.w(current_node, neighbour))
                    dist[neighbour] = dist[current_node] + graph.w(current_node, neighbour)
                    pred[neighbour] = current_node

        return float(dist[dest]) 


class bellman_ford(SPAlgorithm):
    def calc_sp(graph: Graph, source: int, dest: int) -> float:
        pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {} #Distance dictionary
        nodes = list(graph.adj.keys())

        #Initialize distances
        for node in nodes:
            dist[node] = float("inf")
        dist[source] = 0

        #Meat of the algorithm
        for _ in range(graph.number_of_nodes()):
            for node in nodes:
                for neighbour in graph.adj[node]:
                    if dist[neighbour] > dist[node] + graph.w(node, neighbour):
                        dist[neighbour] = dist[node] + graph.w(node, neighbour)
                        pred[neighbour] = node

        return float(dist[dest])
    
class A_star:
    def buildH(G): # Example heuristic - assigns every node heuristic value of 5: result should be same as dijkstra's 
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


class Adapter(SPAlgorithm):
    def __init__(self, graph):
        self.a_star_class = A_star()
        self.graph = graph
        # self.source = source
        # self.dest = dest

    def calc_sp(self, source, dest):
        dist, pred = self.a_star_class.a_star(self.graph, source, dest)
        return float(dist[dest])

        
class ShortestPathFinder:
    #init function to have composite relationship 
    def __init__(self): 
       self._Graph = Graph()
       self._SPAlgorithm = SPAlgorithm() 

    def calc_short_path(self, source: int, dest: int) -> float:
        self._SPAlgorithm.calc_sp(source, dest)

    #Property decorator to achieve setters
    @property
    def set_graph(self, graph: Graph):
        self._Graph = graph

    @property 
    def set_algorithm(self, algo: SPAlgorithm):
        self._SPAlgorithm = algo         
