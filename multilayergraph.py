import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx

# Multiplex Graph Class
class MultilayerGraph(object):

    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None, focus_edge=None, pos=None):

        # class attributes
        self.graphs = graphs                # list of graphs to combine into a multiplex graph
        self.total_layers = len(graphs)     # total number of layers in multiplex graph
        self.node_labels = node_labels      # node labels used for visualizing multiplex graph
        self.layout = layout                # layout used for visualizing multiplex graph
        self.focus_edge = focus_edge        # edge to highlight when visualizing multiplex graph
        self.pos = pos                      # node positions used in spatial graph convolution
        self.xmin = 0                       # boundary used for visualizing multiplex graph
        self.xmax = 0                       # boundary used for visualizing multiplex graph
        self.ymin = 0                       # boundary used for visualizing multiplex graph
        self.ymax = 0                       # boundary used for visualizing multiplex graph
        if ax:
            self.ax = ax
            self.fig = plt.gcf()  # Get current figure
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        # methods 
        self.get_nodes()                    
        self.get_edges_within_layers()      
        self.get_edges_between_layers()
        self.get_node_positions()
        self.get_multilayer_graph()
        self.draw()
        self.close()

    # get an internal representation of nodes with the format (node ID, layer).
    def get_nodes(self):
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])

    # get edges in the individual layers by the node IDs.
    def get_edges_within_layers(self):
        self.edges_within_layers = []
        self.intralayer_edges = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])
            self.intralayer_edges.extend([(chr(z + 65) + str(source), chr(z + 65) + str(target)) for source, target in g.edges()])

    # connect interlayer edges where nodes with the same id have the same position.
    def get_edges_between_layers(self):
        self.edges_between_layers = []
        self.interlayer_edges = []

        # connect nodes with the same position
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())

            # connect cross edges between different nodes completely 
            for i in shared_nodes:
                for j in shared_nodes:
                    if i != j:
                        if self.focus_edge == None:
                            self.edges_between_layers.append(((i,z1), (j,z2)))
                        else:
                            self.edges_between_layers.append(((self.focus_edge[0],z1), (self.focus_edge[1],z2)))    
                        self.interlayer_edges.append((chr(z1 + 65) + str(i), chr(z2 + 65) + str(j))) 

    # get node positions for spatial graph convolution
    def get_node_positions(self, *args, **kwargs):
        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        if self.pos == None:
            pos = self.layout(composition, *args, **kwargs)
        else:
            pos = self.pos
        self.node_positions = dict()
        for z, g in enumerate(self.graphs):       
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})

     # combine individual graphs into a single multilayer graph
    def get_multilayer_graph(self):
        self.multilayer_graph = self.create_simulated_multiplex_graph()

    # assign edge weights to multilayer graph
    def set_edge_weights(self):
        self.weights_within_layers = []
        for edge in self.edges_within_layers:
            self.weights_within_layers.append(round(random.uniform(0, 1),2))

        self.weights_between_layers = []
        for edge in self.edges_between_layers:
            self.weights_between_layers.append(round(random.uniform(0, 1),2))        

    # draw nodes for visualization
    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    # draw edges for visualization
    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

    # calculate layer boundaries for visualization
    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        ymin = xmin if xmin <= ymin else ymin
        ymax = xmax if xmax >= ymax else ymax
        dx = xmax - xmin
        dy = ymax - ymin
        if xmin < self.xmin:
            self.xmin = xmin
        if xmax > self.xmax:
            self.xmax = xmax
        if ymin < self.ymin:
            self.ymin = ymin
        if ymax > self.ymax:
            self.ymax = ymax
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)

    # draw layers for visualization
    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=1.0)
        xmin = ymin = -3.0
        xmax = ymax = 3.0
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)
        self.ax.text(x=xmin-0.5, y=ymin+2.0, z=z, s='Layer ' + chr(z+65), weight='bold')   

    # draw node labels for visualization
    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                self.ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)
   
    # main draw method for visualization
    def draw(self):
        self.draw_edges(self.edges_within_layers,  color='black', alpha=1.0, linestyle='-', zorder=3)
        self.draw_edges(self.edges_between_layers, color='red', alpha=1.0, linestyle='--', zorder=3)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=4, color='green')
            self.draw_nodes([node for node in self.nodes if node[1]==z], c='black', alpha=1.0, s=200, zorder=2)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  color='white',
                                  zorder=100) 
            
    # Create a simulated multilayer graph    
    def create_simulated_multiplex_graph(self):
        G = nx.Graph()
        n = len(self.graphs[0].nodes)   # n = number of nodes in each layer
        m = self.total_layers           # m = number of layers       

        self.intralayer_weights = []
        self.interlayer_weights = []   

        # Generate spatial positions for each node
        positions = [(np.random.randint(0, 11), np.random.randint(0, 11)) for _ in range(n)]
        
        # initialize nodes and intralayer edges with weights
        layer_names = [(chr(65 + i), i) for i in range(m)]
        for layer, z in layer_names:
            for i, (x, y) in enumerate(positions):
                G.add_node(f'{layer}{i}', pos=(x, y, z), feature=round(np.random.rand(),2))
                for j in range(i + 1, n):
                    intralayer_weight = round(np.random.rand(),2)
                    G.add_edge(f'{layer}{i}', f'{layer}{j}', weight=intralayer_weight)
                    self.intralayer_weights.append(intralayer_weight)
        
        # update features based on normalized weighted sum using number of weighted neighbors
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:  # If the node has neighbors
                total_weight = sum(G[node][neighbor]['weight'] for neighbor in neighbors)
                weighted_sum = sum(G[node][neighbor]['weight'] * G.nodes[neighbor]['feature'] for neighbor in neighbors)               
                G.nodes[node]['feature'] = round(weighted_sum / total_weight if total_weight > 0 else G.nodes[node]['feature'], 3) # Normalize by weights
        
        # add interlayer edges weights based on updated features        
        for layer in range(m-1):
            source_layer = chr(layer+65)
            target_layer = chr(layer+66)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        source_node = source_layer + str(i)
                        target_node = target_layer + str(j)
                        interlayer_weight = round((G.nodes[source_node]['feature'] + G.nodes[target_node]['feature']) / 2,2)
                        G.add_edge(source_node, target_node, weight=interlayer_weight)
                        self.interlayer_weights.append(interlayer_weight)      

        # assign graph target for regression (target = summation of feature weights)
        target = 0
        for node in G.nodes:
            target += G.nodes[node]['feature']
        G.graph['y'] = target

        # assign graph label used for binary classification
        label = 0 
        target = target % 1  # get decimal part
        if target > 0.5:
            label = 1
        G.graph['label']  = label

        return G
    
    def close(self):
            plt.close(self.fig)