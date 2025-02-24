"""
Multiplex Spatial Convolutional Graph Networks (MSGCN)
Author: Steven Wilson, University of Oklahoma, 2024

This file is the main program used to run experiments for the MSGCN project.  
"""

# load common libraries
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging

# load network libraries
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

# load multilayer graph library
from multilayergraph import MultilayerGraph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpatialGraphConv(MessagePassing):
    """
    Spatial Graph Convolution
    """
    def __init__(self, coors, in_channels, out_channels, hidden_size, dropout=0, activation='relu'):
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.dropout = dropout
        self.activation = activation
        self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, pos, edge_index, edge_attr):
               
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1, num_nodes=x.size(0))

        out = self.propagate(edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr, aggr='add')
        
        return out 
    
    # where magic happens for spatial GCN
    def message(self, pos_i, pos_j, x_j, edge_attr):

        relative_pos = pos_j - pos_i # distance between nodes

        spatial_scaling = F.relu(self.lin_in(relative_pos))  # apply relu to the distance to remove negative distances            
        
        n_edges = spatial_scaling.size(0) # ignore, number of edges

        spatial_scaling_result = spatial_scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1) # multiply by neighbor feature value

        result = spatial_scaling_result * edge_attr.view(-1, 1) # reshaping so it's compatible with next layer

        return result # send back result       

    def update(self, aggr_out):

        aggr_out = self.lin_out(aggr_out)
        aggr_out = F.relu(aggr_out)            
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        return aggr_out

class SGCN(torch.nn.Module):
    """
    Spatial Graph Convolution Network
    """
    def __init__(self, dim_coor, out_dim, input_features,
                 layers_num, model_dim, out_channels_1, dropout,
                 use_cluster_pooling):
        super(SGCN, self).__init__()
        self.layers_num = layers_num
        self.use_cluster_pooling = use_cluster_pooling

        self.conv_layers = [SpatialGraphConv(coors=dim_coor,
                                             in_channels=input_features,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout)] + \
                           [SpatialGraphConv(coors=dim_coor,
                                             in_channels=model_dim,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.fc1 = torch.nn.Linear(model_dim, out_dim)

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch

        for i in range(self.layers_num):
            x = self.conv_layers[i](x, pos, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        
        return x
    
class NonNegativeCustomLoss(nn.Module):
    """
    Non negative custom loss to reduce oversmoothing
    """
    def __init__(self, mse_weight=1.0, spread_weight=0.5, range_penalty_weight=0.1):
        super(NonNegativeCustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.spread_weight = spread_weight
        self.range_penalty_weight = range_penalty_weight

    def forward(self, predicted, target):
        # Calculate Mean Squared Error (MSE) loss
        mse_loss = torch.mean((predicted - target)**2)

        # Calculate the inverse of variance of the predicted values
        # This will be small when variance is large, and large when variance is small
        spread_loss = 1.0 / (torch.var(predicted) + 1e-8)  # Adding epsilon to avoid division by zero

        # Calculate range penalty term
        # Penalize the squared distance from the predictions to the edges of the range [0, 1]
        # This encourages predictions to occupy the extremes of the range
        predicted_range = torch.max(predicted) - torch.min(predicted)
        target_range = torch.max(target) - torch.min(target)
        range_penalty = torch.mean((predicted_range - target_range)**2)

        # Combine MSE loss, spread term, and range penalty term
        total_loss = (self.mse_weight * mse_loss +
                      self.spread_weight * spread_loss +
                      self.range_penalty_weight * range_penalty)

        return total_loss
    
def convert_to_torch_graph(nx_graph):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    """
    # Convert node attributes to tensor format
    x = torch.tensor([[data['feature']] for _, data in nx_graph.nodes(data=True)], dtype=torch.float)

    # convert graph label to tensor format
    y = torch.tensor([nx_graph.graph['y']], dtype=torch.float)

    # convert node positions to tensor format
    pos = torch.tensor([data['pos'] for _, data in nx_graph.nodes(data=True)], dtype=torch.float)

    # define variables to hold multilayer edges and true weights
    multilayer_edges = [(u,v) for u, v in nx_graph.edges()]
    intralayer_edges = []
    interlayer_edges = []
    interlayer_attr = []

    # prepare multilayer edges
    for edge in multilayer_edges:
        if edge[0][0] != edge[1][0]: # interlayer edge
            if edge[0][1:len(edge[0])] == edge[1][1:len(edge[1])]:  # remove at same position
                nx_graph.remove_edge(edge[0], edge[1])
            else:  # save at different positions
                interlayer_edges.append((edge[0], edge[1]))
                # save interlayer_edge weight for prediction loss
                interlayer_attr.append(nx_graph[edge[0]][edge[1]]['weight'])
                # reset interlayer_edge weight to random value for training
                nx_graph[edge[0]][edge[1]]['weight'] = round(np.random.uniform(0, 1),2)
        else: # intralayer edge
            intralayer_edges.append((edge[0],edge[1]))

    # Convert node labels to contiguous integers
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    
    # Create edge index tensor with the new node labels
    edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in nx_graph.edges()]).t().contiguous()

    # create interlayer edge tensor with new node labels
    interlayer_edges = nx_graph.graph['inter_edge_true'] 
    interlayer_weights = nx_graph.graph['inter_edge_weight']
    swap_node = nx_graph.graph['swap_node']
    
    # Convert edge weights to tensor format
    edge_attr = torch.tensor([nx_graph[u][v]['weight'] for u, v in nx_graph.edges], dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, swap_node=swap_node, interlayer_edges=interlayer_edges, interlayer_weights=interlayer_weights)

def train(loader, model, optimizer, criterion, num_nodes):
    model.train()
    loss_all = 0
    results = []

    count = 0
    for data in loader:
        if len(data.edge_index) > 0:
            # print('data.x', data.x)
            optimizer.zero_grad()
            out = model(data)
            # print('predicted:', out.tolist()[0][0])
            # print('actual:', data['interlayer_weights'][0])
            
            if num_nodes <= 2:
                loss = F.mse_loss(out, torch.tensor(data.interlayer_weights, dtype=torch.float))
            else:
                loss = criterion(out, torch.tensor(data.interlayer_weights, dtype=torch.float))

            # print('prediction:', out, 'actual:', data.interlayer_weights)
            # print('raw loss:', loss, 'avg loss:', loss/len(data.interlayer_weights[0]), 'num edges:', len(data.interlayer_weights[0]))

            loss_all += data.num_graphs * loss.item()
            # print('data_item:', count, 'loss:', loss.item(), 'pred:', out.tolist()[0], 'actual:', data.interlayer_weights[0])

            i = 0
            for item in range(len(data['interlayer_edges'][0])):
                result = {}
                result['swap_node'] = data['swap_node'][0]
                result['edge'] = data['interlayer_edges'][0][i]
                result['actual_weight'] = data['interlayer_weights'][0][i]
                result['predicted_weight'] = out.tolist()[0][0][i]
                results.append(result)                
                i += 1
                
            loss.backward()
            optimizer.step()
            count += 1

    return loss_all / len(loader), results

def test(loader, model, optimizer, criterion, num_nodes):
    model.eval()
    loss_all = 0
    results = []

    count = 0
    for data in loader:
        if len(data.edge_index) > 0:
            out = model(data)

            if num_nodes <= 2:
                loss = F.mse_loss(out, torch.tensor(data.interlayer_weights, dtype=torch.float))
            else:
                loss = criterion(out, torch.tensor(data.interlayer_weights, dtype=torch.float))
                
            loss_all += data.num_graphs * loss.item()
            # print('data_item:', count, 'loss:', loss.item(), 'pred:', out.tolist()[0], 'actual:', data.interlayer_weights[0])
            
            i = 0
            for item in range(len(data['interlayer_edges'][0])):
                result = {}
                result['swap_node'] = data['swap_node'][0]
                result['edge'] = data['interlayer_edges'][0][i]
                result['actual_weight'] = data['interlayer_weights'][0][i]
                result['predicted_weight'] = out.tolist()[0][0][i]
                results.append(result)                
                i += 1

            count += 1

    return loss_all / len(loader), results

def create_parser():
    """
    Arguments provided in the experiment run
    """
    parser = argparse.ArgumentParser(description='MSGCN Experiments', fromfile_prefix_chars='@')
    parser.add_argument('-num_trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('-trial', type=int, default=0, help='Current trial number')
    parser.add_argument('-data_path', type=str, default='data', help='Path to the dataset')
    parser.add_argument('-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('-lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('-dim_coor', type=int, default=3, help='number of spatial coordinates')
    parser.add_argument('-out_dim', type=int, default=1, help='number of output classes (classification) or nodes (regression)')
    parser.add_argument('-label_dim', type=int, default=1, help='number of input node features to sgcn layer')
    parser.add_argument('-layers_num', type=int, default=1, help='number of sgcn layers')
    parser.add_argument('-model_dim', type=int, default=16, help='number of output dimensions from each sgcn layer')
    parser.add_argument('-out_channels_1', type=int, default=32, help='sgcn hidden layer size (inner convolutions)')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout percentage')
    parser.add_argument('-use_cluster_pooling', type=bool, default=False, help='flag to use cluster pooling')
    parser.add_argument('-mse_option', type=float, default=1.0, help='custom loss mse percentage')
    parser.add_argument('-spread_option', type=float, default=0.0, help='custom loss spread percentage')
    parser.add_argument('-range_option', type=float, default=0.0, help='custom loss range percentage')
    parser.add_argument('-graph_types', type=str, nargs='+', default=['complete'], help='Allowed network types are ["complete","random","smallworld"]')
    parser.add_argument('-num_graphs', type=int, default=0, help='Number of synthetic networks to generate')
    parser.add_argument('-num_nodes', type=int, nargs='+', default=[4], help='List of node counts for the graphs')
    parser.add_argument('-num_layers', type=int, nargs='+', default=[2], help='List of layer counts for the graphs')
    parser.add_argument('-num_neighbors', type=int, nargs='+', default=[2], help='List of k nearest neighbors (smallworld)')
    parser.add_argument('-link_probability', type=float, nargs='+', default=[0.3], help='List of edge addition probabilities for smallworld networks')
    return parser

def generate_data(n, l, k, p, num_graphs=100, graph_type='complete'):
    """
    Generates synthetic network data based on the given parameters.
    """
    graph_list = []
    node_labels = {nn: str(nn) for nn in range(n)}

    for i in range(num_graphs):
        graph_layers = []
        for layer in range(l):
            if graph_type == 'complete':
                graph_layers.append(nx.complete_graph(n))
            if graph_type == 'random':
                graph_layers.append(nx.erdos_renyi_graph(n, p))
            if graph_type == 'smallworld':
                graph_layers.append(nx.newman_watts_strogatz_graph(n, k, p))                           
        graph_list.append(MultilayerGraph(graph_layers, node_labels=node_labels, ax=None, layout=nx.spring_layout))

        if i % 10 == 0:  # Log progress every 10 graphs
            logging.info(f"{i + 1}/{num_graphs} graphs created...")

    logging.info("Graph generation complete.")
    return graph_list

def save_data(data, filename):
    """
    Saves synthetic network data to a file for use in experiments.
    """
    try:
        with open(filename, 'wb') as fileObj:
            pickle.dump(data, fileObj)
        logging.info(f"Data saved to {filename}")
    except IOError as e:
        logging.error(f"Failed to save data: {e}")

def load_data(graph_type,num_nodes,num_layers):
    """
    Loads synthetic networks based on the graph type, number of nodes, and number of layers
    """
    filename = 'data/%s_%s_nodes_%s_layers.pkl'%(graph_type,num_nodes,num_layers)
    logging.info(f"Loading Filename: {filename}")

    fileObj = open(filename, 'rb')
    graphs = pickle.load(fileObj)
    fileObj.close() 

    return graphs

def preprocess(graphs):
    """
    Pre-processing needed for random graphs and small world graphs.
    This is done so that all graphs will have the same number of interlayer edges for the projection step.
    Adds all missing edges and assigns weight=0 so they will not impact the network. 
    """
    all_edges = []
    for src_node in graphs[0].multilayer_graph.nodes():
        for dst_node in graphs[0].multilayer_graph.nodes():
            if (src_node != dst_node) and (src_node[0] == dst_node[0]):
                all_edges.append((src_node, dst_node))
    return all_edges

def run_experiment(args, graph_type, num_nodes, num_layers):
    """
    main function to run experiments
    """
    # parse experiment parameters
    num_experiments = args.num_trials
    mse_option = args.mse_option
    spread_option = args.spread_option
    range_option = args.range_option

    # parse sgcn parameters
    num_epoch=args.epochs
    lr=args.lr
    dropout=args.dropout
    layers_num=args.layers_num               
    model_dim=args.model_dim                
    out_channels_1=args.out_channels_1          
    use_cluster_pooling=args.use_cluster_pooling
    dim_coor=args.dim_coor                   
    label_dim=args.label_dim                 
    out_dim=args.out_dim                   

    # load multilayer graphs based on the graph type
    graphs = load_data(graph_type,num_nodes,num_layers)

    # create empty dataframe to hold results
    df_results = pd.DataFrame()

    # create list to hold results
    history = []

    for experiment in range(num_experiments):
        logging.info(f'************ Experiment {experiment} ************')

        # pre-processing needed for random graphs and small world graphs
        all_edges = preprocess(graphs)

        # define lists used for projection
        layers = []
        ids = []
        proxy_graphs = []
        num_nodes = int(len(graphs[0].nodes) / num_layers)

        # list to hold projection graphs
        A = [[] for n in range(num_nodes)]
        A_layers = [[[] for _ in range(num_nodes)] for _ in range(num_layers - 1)]
        
        # loop graphs in dataset
        logging.info(f"number of graphs in dataset: {len(graphs)}")
        for graph in graphs:
            graph_layers = []

            # get networkx graph
            nx_graph = graph.multilayer_graph 

            # get missing edges and assign weight = 0 for each missing edge
            missing_edges = [(x[0], x[1], 0.0) for x in all_edges if x not in nx_graph.edges()]

            # assign missing edges to graph
            nx_graph.add_weighted_edges_from(n for n in missing_edges)
            
            # get layers and node ids
            for node in nx_graph.nodes:
                layer, id = node[:1], node[1:]
                if layer not in layers:
                    layers.append(layer)
                if id not in ids:
                    ids.append(id)

            # separate layers
            for layer in layers:
                layer_nodes = []
                for id in ids:
                    node_label = str(layer) + str(id)
                    layer_nodes.append(node_label) 

                # create new layer graph
                graph_layer = nx.Graph()

                # Copy graph attributes from nx_graph to graph_layer
                graph_layer.graph.update(nx_graph.graph)

                # Add nodes with attributes
                for node in layer_nodes:
                    graph_layer.add_node(node, **nx_graph.nodes[node])            

                # Add edges with attributes (if they exist in the original graph)
                for u, v in nx_graph.edges(layer_nodes):
                    if u in layer_nodes and v in layer_nodes:
                        graph_layer.add_edge(u, v, **nx_graph.edges[u, v])            
                
                graph_layers.append(graph_layer)   

            # graph layer nodes
            for i in range(len(graph_layers) - 1):

                # build proxy graphs using node projection for each node
                for idx, node in enumerate(graph_layers[i+1].nodes):
                    proxy_graph = graph_layers[i+1].copy() # initialize proxy graph with layer 1
                    source_nodes = graph_layers[i].nodes # initalize source nodes with layer 0 nodes
                    source_values = nx.get_node_attributes(graph_layers[i], 'feature') # get source node values from layer 0
                    
                    proxy_feature = {node: {'feature': source_values[list(source_nodes)[idx]]}} # get proxy feature from source values
                    
                    proxy_graph.graph['source_node'] = list(source_nodes)[idx]
                    proxy_graph.graph['swap_node'] = node
                    proxy_graph.graph['neighbors'] = [n for n in proxy_graph.neighbors(node)]
                    proxy_graph.graph['inter_edges_pred'] = [(list(source_nodes)[idx], n) for n in proxy_graph.neighbors(node)]
                    proxy_graph.graph['interlayer_nodes'] = graph_layers[i].nodes
                    
                    inter_edge_trues = []
                    inter_edge_weights = []
                    for edge in [(list(source_nodes)[idx], n) for n in proxy_graph.neighbors(node)]:
                        edge_idx = 0
                        for interlayer_edge in graph.interlayer_edges:
                            if edge == interlayer_edge:
                                inter_edge_trues.append(edge)
                                inter_edge_weights.append(graph.interlayer_weights[edge_idx])
                            edge_idx += 1
                    proxy_graph.graph['inter_edge_true'] = inter_edge_trues
                    proxy_graph.graph['inter_edge_weight'] = inter_edge_weights

                    nx.set_node_attributes(proxy_graph, proxy_feature)  # set proxy feature

                    if idx != 0:
                        proxy_graphs.append(proxy_graph)  # save proxy graph to list
                    A[idx].append(proxy_graph)
                    A_layers[i][idx].append(proxy_graph)

                # define history list to hold results
                total_result_count = 0 

        for A in A_layers: 

            for n in range(num_nodes):
                result_count = 0

                # split 80% of the data for training
                split_ratio = 0.8  
                train_size = int(split_ratio * len(A[n]))

                # split the proxy graphs into training and test sets
                train_graphs = A[n][:train_size]
                validation_graphs = A[n][train_size:]

                # split test graphs from validation graphs
                validation_size = int(0.5 * len(validation_graphs))
                test_graphs = validation_graphs[:validation_size]
                validation_graphs = validation_graphs[validation_size:]

                # show number of training and test graphs
                logging.info(f"# training graphs: {len(train_graphs)}")
                logging.info(f"# validation graphs: {len(validation_graphs)}")
                logging.info(f"# test graphs: {len(test_graphs)}")
                logging.info(f"projection node: {train_graphs[0].graph['swap_node']}")
                logging.info(f"interlayer edge: {train_graphs[0].graph['inter_edge_true']}")

                # set output dimensions for the number of interlayer edges
                out_dim = len(train_graphs[0].graph['inter_edge_true']) 
                logging.info("number of output dimensions: {out_dim}")

                # Convert the training and test graphs to Data format
                train_data = [convert_to_torch_graph(g) for g in train_graphs]
                validation_data = [convert_to_torch_graph(g) for g in validation_graphs]            

                # Create DataLoaders
                train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
                validation_loader = DataLoader(validation_data, batch_size=1, shuffle=True)

                # initialize device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # define model
                model = SGCN(dim_coor=dim_coor,
                            out_dim=out_dim,
                            input_features=label_dim,
                            layers_num=layers_num,
                            model_dim=model_dim,
                            out_channels_1=out_channels_1,
                            dropout=dropout,
                            use_cluster_pooling=use_cluster_pooling).to(device)

                # define optimizer
                criterion = NonNegativeCustomLoss(mse_weight=mse_option, spread_weight=spread_option, range_penalty_weight=range_option)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # train model
                for epoch in range(num_epoch):
                    loss_all, train_results = train(train_loader, model, optimizer, criterion, num_nodes)
                    train_acc, validation_results = test(validation_loader, model, optimizer, criterion, num_nodes)
                    
                    # show training status
                    if epoch % 5 == 0:
                        logging.info(f'Epoch: {epoch}, Training Loss: {loss_all:.4f}, Validation Loss: {train_acc:.4f}')        
                    
                    # add results to history
                    result = {}
                    result['graph_type'] = graph_type
                    result['n'] = num_nodes
                    result['experiment'] = experiment + 1
                    result['epoch'] = epoch
                    result['swap_node'] = train_results[0]['swap_node']
                    result['train_loss'] = loss_all
                    result['validation_loss'] = train_acc

                    df_train_results = pd.DataFrame(train_results)
                    df_train_results['result_type'] = 'train'
                    df_train_results['loss'] = loss_all               
                    df_train_results['graph_type'] = graph_type
                    df_train_results['n'] = num_nodes
                    df_train_results['experiment'] = experiment + 1
                    df_train_results['epoch'] = epoch 
                    df_train_results['swap_node'] = train_results[0]['swap_node']
                    df_results = pd.concat([df_results, df_train_results])

                    df_validation_results = pd.DataFrame(validation_results)
                    df_validation_results['result_type'] = 'validation'
                    df_validation_results['loss'] = train_acc   
                    df_validation_results['graph_type'] = graph_type
                    df_validation_results['n'] = num_nodes
                    df_validation_results['experiment'] = experiment + 1
                    df_validation_results['epoch'] = epoch 
                    df_validation_results['swap_node'] = train_results[0]['swap_node']
                    df_results = pd.concat([df_results, df_validation_results])

                    # evaluate model
                    test_data = [convert_to_torch_graph(g) for g in test_graphs]
                    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)            
                    test_acc, test_results = test(test_loader, model, optimizer, criterion, num_nodes)

                    df_test_results = pd.DataFrame(test_results)
                    df_test_results['result_type'] = 'test'
                    df_test_results['loss'] = test_acc  
                    df_test_results['graph_type'] = graph_type
                    df_test_results['n'] = num_nodes
                    df_test_results['experiment'] = experiment + 1
                    df_test_results['epoch'] = epoch 
                    df_test_results['swap_node'] = train_results[0]['swap_node']
                    df_results = pd.concat([df_results, df_test_results])

                history.append(result)
                result_count += 1

                total_result_count += result_count
                logging.info(f'total result count: {total_result_count}')
                logging.info(f'history size: {len(history)}')
                df_result = pd.DataFrame(history)
                df_results = pd.concat([df_results, df_result])

    # save results
    results_filename = 'trials/results_%s_trials_%s_nodes_%s_layers_%s.csv'%(num_experiments, num_nodes, num_layers, graph_type)
    df_results['edge_loss'] = (df_results['actual_weight'] - df_results['predicted_weight'])**2
    df_results.to_csv(results_filename, index=False)
    logging.info(f'results saved to: {results_filename}')

    return df_results

#################################################################
if __name__ == "__main__":
    # get command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # loop for number of layer parameters
    for l in args.num_layers:

        # loop for number of node parameters
        for n in args.num_nodes: 

            # loop for each graph type
            for graph_type in args.graph_types:

                # generate synthetic data if number of synthetic graphs > 0
                if args.num_graphs > 0:
                    logging.info(f"Generating {args.num_graphs} synthetic networks")
                    output_filename = f"{args.data_path}/{graph_type}_{n}_nodes_{l}_layers.pkl"
                    if graph_type in ["random","smallworld"]:
                        for k in args.num_neighbors:
                            for p in args.link_probability:
                                multilayer_graphs = generate_data(n, l, k, p, num_graphs=args.num_graphs, graph_type=graph_type)
                    else:
                        multilayer_graphs = generate_data(n, l, k=0, p=0, num_graphs=args.num_graphs, graph_type=graph_type) 
                    save_data(multilayer_graphs, output_filename)

                # run experiment with arguments
                run_experiment(args, graph_type, num_nodes=n, num_layers=l)
