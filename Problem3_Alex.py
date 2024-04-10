import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time 
import math 

class Graph:
    def __init__(self, n):
        """
        Initializes a graph object with a given number of nodes.

        Args:
            n (int): The number of branch nodes.
        """
        self.n = n
        # The total number of nodes is n branch nodes + 1 center node + 2n leaf nodes
        self.order = 1 + n + 2 * n
        self.adj_list = {i: [] for i in range(self.order)}
        self.edge_labels = {}  # Dictionary to store edge labels

    def add_edge(self, u, v, label):
        """
        Adds an edge between two nodes with a given label.

        Args:
            u (int): One end of the edge.
            v (int): The other end of the edge.
            label (int): The label to be assigned to the edge.
        """
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        self.edge_labels[(u, v)] = label  # Storing the edge label
        self.edge_labels[(v, u)] = label  # Storing the edge label for the opposite direction


def assign_labels(graph):
    """
    Assigns vertex and edge labels to the graph using edge irregular k-labeling scheme.

    Edge Irregular k-labeling:
    assigns labels to both vertices and edges such that the maximum label on any edge incident to a vertex is the label of that vertex. 

    Args:
        graph (Graph): The graph object to which labels will be assigned.

    Returns:
        tuple: A tuple containing dictionaries of vertex labels and unique edge labels.
    """
    vertex_labels = {}  # Dictionary to store vertex labels
    edge_labels = {}  # Dictionary to store edge labels
    # Calculate k based on the number of edges and vertices using the ceiling function
    number_of_edges = 3 * graph.n + graph.n  # Correct formula for the number of edges in the modified star graph
    k = math.ceil(number_of_edges * math.log2(graph.order))  # Use math.ceil to round up
    print("k value (upper bound): ", k)

    # Greedy labeling starting from central vertex
    vertex_labels[0] = 1 # min(graph.n + 1, k)  # Labeling the center vertex
    
    # Output vertex labels
    inner_verts = 11
    for vert in range(1, graph.n + 1):
        vertex_labels[vert] = min(k, inner_verts)
        inner_verts += 4
 
    for i in range(1, graph.n+1):
        branch_val = vertex_labels[i]
        for j in range(1, 4, 2):
            vert = vert + 1
            vertex_labels[vert] = min(k, branch_val + j)
            
    # After labeling, find the maximum vertex label
    max_vertex_label = max(vertex_labels.values())
    print("Maximum vertex label value: ", max_vertex_label)



    # Assigning edge labels
    for vertex, neighbors in graph.adj_list.items():
        for neighbor in neighbors:
            edge_label = vertex_labels[vertex] + vertex_labels[neighbor]
            edge_labels[(vertex, neighbor)] = edge_label
            edge_labels[(neighbor, vertex)] = edge_label

    return vertex_labels, edge_labels

def verify_unique_edge_values(edge_labels):
    """
    Verifies if all edge values are unique and prints the maximum edge weight.

    Args:
        edge_labels (dict): A dictionary of edge labels.

    Returns:
        bool: True if all edge values are unique, False otherwise.
    """
    edge_values = list(edge_labels.values())
    unique_values = len(edge_values)/2 == len(set(edge_values)) + 1
    max_edge_value = max(edge_values)
    print(f"All edge values are unique: {unique_values}")
    print(f"Maximum edge weight value: {max_edge_value}")

def main():
    start_time = time.time()  # Start timing

    # Number of branch nodes
    n = int(input("Enter the value of n: "))

    # Initialize graph
    graph = Graph(n)

    # Adding edges for the modified star graph
    for i in range(1, n + 1):
        graph.add_edge(0, i, 0)  # Connect branch nodes to center
        leaf1 = n + 2 * (i - 1) + 1
        leaf2 = n + 2 * (i - 1) + 2
        graph.add_edge(i, leaf1, 0)  # Add leaf nodes
        graph.add_edge(i, leaf2, 0)
        # Connect each branch node to the next and previous one
        next_branch = i % n + 1
        graph.add_edge(i, next_branch, 0)

    # Assign labels to vertices and edges
    vertex_labels, edge_labels = assign_labels(graph)
    
    # printing the graph labeling compute time 
    end_time = time.time()  # End timing
    print(f"Computation time (before plotting): {end_time - start_time} seconds")
    
    
    # verifying edge weight uniqueness (for debugging)
    verify_unique_edge_values(edge_labels)
    
    # printing edge and vertex labels (for debugging)
    print("Vertex Labels:")
    for vertex, label in vertex_labels.items():
        print(f"Vertex {vertex}: Label {label}")
    print("Edge Labels:")
    i = 1
    for edge, label in edge_labels.items():
        if i == 1:
            print(f"Edge {edge}: Label {label}")
            i = 0
        else:
            i = 1
    
    # Visualize graph using networkx library
    plt.figure(figsize=(12, 12))  # Increase the figure size
    G = nx.Graph()
    for vertex, neighbors in graph.adj_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor, label=edge_labels[(vertex, neighbor)])
    
    # Use a circular layout for the branch nodes
    center_pos = {0: (0, 0)}  # Center node at origin
    branch_pos = nx.circular_layout(range(1, n + 1), center=(0, 0))  # Circular layout for branch nodes
    
   # Calculate the circular positions for the branch nodes
    radius = 1  # Radius for branch nodes
    branch_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    branch_pos = {i: (radius * np.cos(ang), radius * np.sin(ang)) for i, ang in enumerate(branch_angles, 1)}

    # Calculate the positions for the leaf nodes
    leaf_angles = np.linspace(0, 2 * np.pi, 2*n, endpoint=False)
    radius_leaf = 1.5  # Radius for leaf nodes
    leaf_pos = {}
    for i in range(n):
        leaf_pos[n + 2 * i + 1] = (radius_leaf * np.cos(leaf_angles[2 * i]), radius_leaf * np.sin(leaf_angles[2 * i]))
        leaf_pos[n + 2 * i + 2] = (radius_leaf * np.cos(leaf_angles[2 * i + 1]), radius_leaf * np.sin(leaf_angles[2 * i + 1]))

    # Merge all positions together
    pos = {**center_pos, **branch_pos, **leaf_pos}

    nx.draw(G, pos, labels = vertex_labels, node_color='lightblue', edge_color='gray', width=2, linewidths=1, node_size=700, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title('Snowflake Graph with Edge Irregular K-Labeling')
    plt.axis('off')  # Turn off the axis
    plt.show()


if __name__ == "__main__":
    main()
