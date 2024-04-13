import math
import networkx as nx
import matplotlib.pyplot as plt
import time
import psutil
import time
import threading


class Graph:
    def __init__(self, n, m, k, order, d):
        """
        Initializes an instance of the class.

        Parameters:
        - n (int): The value of n.
        - m (int): The value of m.
        - k (int): The value of k.
        - order (int): The value of order.
        - d (int): The value of d.

        Returns:
        None
        """
        # Initializing data structures to represent the graph
        self.n = n
        self.m = m
        self.k = k
        self.order = order
        self.d = d
        self.adj_list = {i: [] for i in range(order)}  # Adjacency list representation of the graph
        # self.edge_weights = {}  # Dictionary to store edge weights ## Uncomment for edge weights set list approach ##
        self.vertex_labels = {i: None for i in range(order)}  # Dictionary to store vertex labels
        
        ## Uncomment for edge weights set list min approach ##
        self.edge_weights_set = set(range(2, self.k*2 + 1)) # Set of possible edge weights
        self.edge_weights = {}

    def add_edge(self, u, v, weight):

        # Adding edge to the adjacency list
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

        # Storing edge weight in both directions
        self.edge_weights[(u, v)] = weight
        self.edge_weights[(v, u)] = weight

    def get_adj_list(self):
        return self.adj_list
    

    def find_min_weight(self):

        write_to_output_file("edge_set.txt", "List of available edge weights: ", self.edge_weights_set)
        min_weight = min(self.edge_weights_set)
        return min_weight
    
    def build_graph(self):
        
        outer_verts = self.n
        # Constructing the star graph with n branchs and m leaves per branch by adding edges 
        for i in range(1, self.n + 1):
            self.add_edge(0, i, 0)  # Connect central vertex to banches
            for j in range(1, self.m):
                outer_verts += 1
                self.add_edge(i, outer_verts, 0) # Connect branch to their external leaves     
                
    def vertex_k_labeling(self):
        """
        Greedy labeling starting from the central vertex.

        Returns:
        dict: A dictionary of vertex labels.
        """
        d = self.d
        self.vertex_labels[0] = 1  # Label for central vertex
        leaf_verts = self.n              # Counter for leaf vertices
        
        current_label = 0           # Initialize current_label to 0
        # Labeling of internal/arm vertices of the graph
        for branch in range(1, self.n + 1):
            # Hardcode the first branch vertex label to 1
            if branch == 1:
                self.vertex_labels[branch] = 1 
                weight = self.vertex_labels[0] + self.vertex_labels[branch]       # Calculate edge weight
                
                if weight in self.edge_weights_set:
                    self.edge_weights_set.remove(weight)
                    
                # Assign edge weight to the edge from center vertex to the branch in both directions **
                self.edge_weights[(0, branch)] = weight
                self.edge_weights[(branch, 0)] = weight
                
                # Labeling the rest of the branch vertices from branch 2 to n (inclusive)
            else:
                current_label = current_label + d                                # From branch 2 to n (inclusive), the label is incremented by d, where d = k/(n-1), k is the maximum label value (ceiling), and n is the number of branch vertices.
                self.vertex_labels[branch] = math.floor(current_label)          # Assign the floored value of current_label to the branch vertex 
                weight = self.vertex_labels[0] + self.vertex_labels[branch]    # Calculate edge weight
                self.edge_weights[(branch, 0)] = weight
                self.edge_weights[(0, branch)] = weight
                
                if weight in self.edge_weights_set:
                    self.edge_weights_set.remove(weight)
                
        # Labeling of leaf vertices of the graph
        
        for branch in range(1, self.n + 1):                                         # Iterate through each branch vertex
            for leaf in range(1, self.m):                                           # Iterate through each leaf vertex of the branch
                leaf_weight = self.find_min_weight()                          # Find the minimum weight that is not already assigned to an edge
                leaf_verts += 1                                                # Increment the leaf vertex counter by 1 to get the next leaf vertex number.
                self.vertex_labels[leaf_verts] =  leaf_weight - self.vertex_labels[branch] 
                self.edge_weights[(leaf_verts, branch)] = leaf_weight
                self.edge_weights[(branch, leaf_verts)] = leaf_weight
                self.edge_weights_set.remove(leaf_weight)
            
        return self.vertex_labels

    def visualize_graph(self):
        """
        Visualizes the graph using NetworkX and Matplotlib.

        Returns:
        None
        """

        if self.n * self.m < 100:  # Threshold check
            
            # Getting the vertex labels, adjacency list, and edge weights
            vertex_labels = self.vertex_labels
            adj_list = self.adj_list
            
            plt.figure(figsize=(12, 12))
            G = nx.Graph()

            G.add_nodes_from(range(self.order))

            # Adding edges from adjacency list

            for vertex, neighbors in adj_list.items():

                for neighbor in neighbors:

                    G.add_edge(vertex, neighbor)

            pos = nx.spring_layout(G, seed= 123 )  # positions for all nodes

            # Adding node labels

            labels = {node: str(label) for node, label in vertex_labels.items()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')

            # Drawing the graph
            nx.draw(G, pos, with_labels=False, node_color='powderblue', node_size=1500)

            # Drawing edge labels
            nx.draw_networkx_edge_labels(G, pos, edge_labels=self.edge_weights, font_color='indigo')
            
            plt.title('Edge Orregular Amalgamated Star Graph of S_{},{}'.format(self.n, self.m))
            plt.axis('off')

            # Save the graph as a PNG file
            plt.savefig("star_graph.png")
            
            plt.show()  # Uncomment to display the graph in the console

        else:
            print("The graph is too large to visualize.")
            
    def compute_complexity(self):
        """
        Computes the theoretical time complexity of the algorithm.

        Returns:
        str: The theoretical time complexity of the algorithm.
        """
        complexity = self.n * self.m * self.k
        return f"T({complexity})"  
            
            
## ------------------ ##

def write_to_output_file(file_name, sentence, value):
    """
    Writes the sentence and value to the output file.

    Parameters:
    - file_name (str): The name of the output file.
    - sentence (str): The sentence to be logged.
    - value (int or float): The value to be logged.

    Returns:
    None
    """
    with open(file_name, 'a') as file:               # Open the file in append mode and write the sentence and value to the file if it exists. If the file does not exist, create it and write the sentence and value to the file.
        file.write(f"{sentence} {value}\n")

def graph_output(graph, vertex_labels, edge_weights):
    """
    Outputs graph data to a graph_output.txt file.
    
    Parameters:
    - graph (Graph): The graph object.
    - vertex_labels (dict): A dictionary containing the vertex labels.
    - edge_weights (dict): A dictionary containing the edge weights.
    
    Returns:
    None
    """
    write_to_output_file("graph_output.txt", "Vertex Labels: ", vertex_labels)
    write_to_output_file("graph_output.txt", "Edge Weights: ", edge_weights)
    write_to_output_file("graph_output.txt", "Theoretical Time Complexity: ", graph.compute_complexity())

def start_output_files(n, m, k, order, d ):
    """
    Initializes the output files with the input parameters.
    
    Parameters:
    - n (int): The number of arms.
    - m (int): The number of leaves per arm.
    - k (int): The maximum label value.
    - order (int): The total number of vertices.
    - d (int): The value of d.
    
    Returns:
    None
    """
    with open("graph_output.txt", 'w') as file:
        file.write("\n+--------------------------+\n")
        file.write(f"n: {n}\n")
        file.write(f"m: {m}\n")
        file.write(f"k: {k}\n")
        file.write(f"order: {order}\n")
        file.write(f"d: {d}\n")
        file.write("\n")
        
    with open("edge_set.txt", 'w') as file:
        file.write("\n+--------------------------+\n")
        file.write(f"n: {n}\n")
        file.write(f"m: {m}\n")
        file.write(f"k: {k}\n")
        file.write(f"order: {order}\n")
        file.write(f"d: {d}\n")
        file.write("\n")
    
def build_and_visualize_graph(n, m):
    """
    Builds and visualizes the graph with n arms and m leaves per arm.
    
    Parameters:
    - n (int): The number of arms.
    - m (int): The number of leaves per arm.
    
    Returns:
    None
    """
    k = math.ceil((m * n  + 1) / 2)  # Maximum label value (ceiling)
    order = m * n + 1                # Total number of vertices
    d = k/(n-1)                      # Value of d 
    graph = Graph(n, m, k, order, d) # Create a graph object
    start_output_files(n, m, k, order, d)  # Initialize the output files with the input parameters
    graph.build_graph()              # Build the graph
    graph.vertex_k_labeling()        # Label the vertices
    graph_output(graph, graph.vertex_labels, graph.edge_weights)  # Output graph data to a file
    graph.visualize_graph()          # Visualize the graph
    
def build_graph_test(n, m):
    k = math.ceil((m * n  + 1) / 2)  # Maximum label value (ceiling)
    order = m * n + 1                # Total number of vertices
    d = k/(n-1)                      # Value of d 
    graph = Graph(n, m, k, order, d) # Create a graph object
    graph.build_graph()              # Build the graph
def timeout_handler(signum, frame):
    raise TimeoutError("Timeout reached")
    
    
def test_limits(n, m, increment):
    """
    Tests the hardware limits by building a graph with n arms and constant m leaves per arm,
    including a timeout feature to stop operations that take too long.

    Parameters:
    - n (int): The number of arms.
    - m (int): The constant number of leaves per arm.
    - increment (int): The increment value for n.

    Returns:
    tuple: A tuple containing the maximum supported n and m values.
    """
    max_n = n  # Maximum supported n value
    max_m = m  # Constant m value, not changing
    timeout = 60  # Timeout in seconds
    stop_event = threading.Event()  # Event to signal timeout
    try:
        while True:
            timer = threading.Timer(timeout, stop_event.set)  # Setup the timer to set the stop event on timeout
            try:
                if psutil.virtual_memory().available < 1024 * 1024 * 1024:  # Check if available memory is less than 1GB
                    print("Insufficient memory. Exiting...")
                    break
                timer.start()  # Start the timer
                build_graph_test(n, m)  # Build the graph with constant m
                if stop_event.is_set():  # Check if the timeout has been reached
                    print(f"Operation stopped due to timeout with n = {n}, m = {m}")
                    break
                max_n = n  # Update max_n to the last successful n
                n += increment  # Increment only n, m remains constant
            except MemoryError:
                print(f"Memory limit reached with n = {n}, m = {m}")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
            finally:
                timer.cancel()  # Ensure the timer is canceled to avoid unwanted side effects
                stop_event.clear()  # Reset the stop event for the next iteration
    except KeyboardInterrupt:
        print(f"Keyboard Interrupt: Current values - n = {max_n} , m = {m}")

    return max_n, max_m

def available_memory():
    """
    Returns the available memory in bytes.
    """
    return psutil.virtual_memory().available


def main():
    user_input = input("Enter 'test' to find the hardware limits or 'build' to specify n and m: ").strip().lower()

    if user_input == "test":
        print("Testing hardware limits. This might take a while...")
        max_n, max_m = test_limits(3, 4, 1) # Test the hardware limits for n>=3 and m=2
        print(f"Maximum supported n: {max_n}, Maximum supported m: {max_m}")
    elif user_input == "build":
        n = int(input("Enter the number of arms (n): "))
        m = int(input("Enter the number of leaves per arm (m): "))
        start_time = time.time()
        build_and_visualize_graph(n, m)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time} seconds")
        build_and_visualize_graph(n, m)
    else:
        print("Invalid input. Please enter 'test' or 'build'.")
    
if __name__ == "__main__":
    main()
    
    
