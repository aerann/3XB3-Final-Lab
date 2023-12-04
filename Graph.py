from abc import ABC, abstractmethod 

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

    def add_node(self):
        self.adj[len(self.adj)] = []

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


    


        

    
    


