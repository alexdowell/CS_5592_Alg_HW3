# Importing necessary libraries
import math
import networkx as nx
import matplotlib.pyplot as plt

# Class definition for a graph
class Graph:
    def __init__(self, n, m, order, k):
        """
        Initializes a graph object for a homogeneous amalgamated star S(n, m).

        Args:
            n (int): The number of arms.
            m (int): The order of each arm.
            order (int): The total number of vertices.
            k (int): The maximum value a vertex label can take.
        """
        # Initializing graph parameters
        self.n = n
        self.m = m
        self.k = k
        self.order = order
        # Initializing data structures to represent the graph
        self.adj_list = {i: [] for i in range(self.order)}  # Adjacency list representation of the graph
        self.edge_weights = {}  # Dictionary to store edge weights
        self.vertex_labels = {i: None for i in range(self.k)}  # Dictionary to store vertex labels

    
    def add_edge(self, u, v, weight):
        """
        Adds an edge between two nodes with a given weight.

        Args:
            u (int): One end of the edge.
            v (int): The other end of the edge.
            weight (int): The weight to be assigned to the edge.
        """
        # Adding edge to the adjacency list
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        # Storing edge weight in both directions
        self.edge_weights[(u, v)] = weight
        self.edge_weights[(v, u)] = weight

    def vertex_k_labeling(self):
        """
        Assigns labels to vertices according to the provided rules, respecting the maximum label 'k'.
        """
        n_ceil = math.ceil(self.n / 4)
        self.vertex_labels[0] = 1 # Central vertex always labeled as 1
        k = self.k  # Maximum allowable label
        m = self.m
# Case 1
        if self.n % 4 in {0, 2, 3}:
            # Labeling internal vertices
            for i in range(1, self.n + 1):
                vertex = i
                if 1 <= i <= n_ceil + 1:
                    self.vertex_labels[vertex] = 3 * i - 2
                elif n_ceil + 1 <= i <= self.n:
                    self.vertex_labels[vertex] = 2 * n_ceil + i

            # Labeling external vertices
            vertex +=1
            for i in range(1, n_ceil + 1):
                for j in range(1, m + 1):  # Adjusted range for dynamic m
                    label = j+1
                    self.vertex_labels[vertex] = label
                    vertex += 1

            for i in range(n_ceil + 1, self.n + 1):
                for j in range(1, m + 1):  # Adjusted range for dynamic m
                    label = self.n + i + j - 1 - (2 * n_ceil)
                    self.vertex_labels[vertex] = label
                    vertex += 1

        # Case 2: n = 1 (mod 4)
        elif self.n % 4 == 1:
            # Labeling internal vertices
            for i in range(1, self.n + 1):
                if 1 <= i <= n_ceil:
                    self.vertex_labels[i] = 3 * i - 2
                else:
                    self.vertex_labels[i] = 2 * n_ceil + i - 1

            # Labeling external vertices ensuring uniqueness
            vertex = self.n + 1
            for i in range(1, self.n + 1):
                for j in range(1, self.m + 1):
                    if i <= n_ceil:
                        proposed_label = j + 1
                    elif i == n_ceil + 1 and j == 1:
                        proposed_label = 2
                    elif i == n_ceil + 1 and j == 2:
                        proposed_label = self.n - n_ceil + 3
                    else:
                        proposed_label = self.n + i + j - 2 * n_ceil

                    # Ensure the label does not exceed k
                    self.vertex_labels[vertex] = min(proposed_label, k)
                    vertex += 1

        # Verify all labels are assigned
        if None in self.vertex_labels.values():
            missing_labels = [vertex for vertex, label in self.vertex_labels.items() if label is None]
            raise ValueError(f"Missing labels for vertices: {missing_labels}")

        return self.vertex_labels
            
    def calculate_edge_weights(self):
        """
        Calculates edge weights based on vertex labels and adjacency list.
        """
        for vertex, neighbors in self.adj_list.items():
            for neighbor in neighbors:
                # Calculate edge weight by summing up the labels of the two vertices
                weight = self.vertex_labels[vertex] + self.vertex_labels[neighbor]
                self.edge_weights[(vertex, neighbor)] = weight
                self.edge_weights[(neighbor, vertex)] = weight
        
        return self.edge_weights
    
    def get_adj_list(self):
        """
        Returns the adjacency list of the graph.
        """
        return self.adj_list
    def recalculate_arm_labeling(self, arm_number, m):
        """
        Recalculates the labeling for the specified arm if an edge weight exists.

        Args:
            arm_number (int): The arm number.
            m (int): The order of the arm.
        """
        for j in range(1, m + 1):  # Adjusted range for dynamic m
            self.vertex_labels[arm_number + (j - 1) * self.n] = (j - 1) % m + 1
def create_and_label_graph(n, m):
    order = math.ceil(m * n + 1)  # Total number of vertices including the central node
    k = math.ceil(order / 2)  # Maximum value a vertex label can take
    graph = Graph(n, m, order, k)

    # Connect each arm node to the central node (0)
    outer_verts = n

    # Adding edges for the star graph
    for i in range(1, n + 1):
        graph.add_edge(0, i, 0)  # Connect central vertex to inner vertices
        for j in range(1, m):
            outer_verts += 1
            graph.add_edge(i, outer_verts, 0) # Connect inner vertices to their external vertices

    # Label the vertices
    graph.vertex_k_labeling()

    # Before calculating edge weights, check that all vertices have a label assigned
    missing_labels = [v for v, label in graph.vertex_labels.items() if label is None]
    if missing_labels:
        raise ValueError(f"Missing labels for vertices: {missing_labels}")

    # Calculate edge weights
    graph.calculate_edge_weights()

    return graph



def print_graph_info(vertex_labels, adj_list, edge_weights, filename='graph_output.txt'):
    with open(filename, 'w') as file:
        file.write("===== Vertex Labels =====\n")
        for vertex, label in vertex_labels.items():
            file.write(f"Vertex: {vertex}, Label: {label}\n")
        
        file.write("\n===== Adjacency List =====\n")
        for vertex, neighbors in adj_list.items():
            neighbors_str = ', '.join(map(str, neighbors))  # Convert list of neighbors to string
            file.write(f"Vertex: {vertex}, Neighbors: [{neighbors_str}]\n")
        
        file.write("\n===== Edge Weights =====\n")
        for edge, weight in edge_weights.items():
            file.write(f"Edge: {edge}, Weight: {weight}\n")
def visualize_graph(graph, vertex_labels, edge_weights):
    G = nx.Graph()
    
    # Ensure all nodes are added, even if they don't have connected edges yet.
    G.add_nodes_from(range(graph.order))
    
    # Adding edges based on the adjacency list
    for vertex, neighbors in graph.get_adj_list().items():
        for neighbor in neighbors:
            if (vertex, neighbor) not in edge_weights:  # Ensure there's an edge weight before adding
                continue
            G.add_edge(vertex, neighbor, weight=edge_weights[(vertex, neighbor)])

    pos = nx.spring_layout(G)  # Compute the positions for all nodes.

    # Ensuring all vertex labels are present for drawing
    labels = {node: str(vertex_labels[node]) for node in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
    nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=500)
    edge_labels = {(u, v): edge_weights[(u, v)] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.show()
def available_memory():
    return psutil.virtual_memory().available

def test_limits(initial_n, initial_m, base_increment, memory_limit_ratio=0.8, timeout_seconds=300):
    n, m = initial_n, initial_m
    increment = base_increment
    max_n, max_m = n, m
    memory_limit = available_memory() * memory_limit_ratio
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout_seconds:
            print("Testing timeout reached. Returning the last found values.")
            break

        try:
            graph = Graph(n, m)
            graph.build_graph()

            # Estimate current memory usage
            current_memory_usage = psutil.Process().memory_info().rss

            if current_memory_usage > memory_limit:
                if increment > 1:
                    # If the limit is reached, step back and reduce increment
                    n -= increment
                    m -= increment
                    increment = 1  # Fine-tune with smallest possible increment
                    continue
                break  # Break if already at smallest increment

            max_n, max_m = n, m
            n += increment
            m += increment  # Increment both n and m

        except MemoryError:
            break

    return max_n, max_m
    
    
# Main function

def main():
    n = int(input("Enter the number of arms (n): "))
    m = int(input("Enter the order of each arm (m): "))  # m represents the order of each arm
    
    graph = create_and_label_graph(n, m)
    print_graph_info(graph.vertex_labels, graph.get_adj_list(), graph.edge_weights, 'graph_info_output.txt')
    visualize_graph(graph, graph.vertex_labels, graph.edge_weights)

if __name__ == "__main__":
    main()
