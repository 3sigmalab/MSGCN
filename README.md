# MSGCN: Muiltiplex Spatial Graph Convolutional Networks

This repository contains the code for the Multiplex Spatial Graph Convolution Network (MSGCN), a multilayer link weight prediction method for spatial multiplex networks. The code also includes synthetic data generation to test the MSGCN method with different multiplex network types and sizes. The supported network types are complete, random, and small-world. 

## Acknowledgement

Please cite the following paper if you use this code. 

```Steven Ewan Wilson and Sina Khanmohammadi. "MSGCN: Multiplex Spatial Graph Convolutional Network for Interlayer Link Weight Prediction." arXiv preprint arXiv:TBD (2025).```

[https://arxiv.org/](https://arxiv.org/).

## Authors

1. <strong>Steven Ewan Wilson:</strong>   </a> <a href="https://scholar.google.com/citations?user=W3IE7YgAAAAJ&hl=en" target="_blank">
        <img src="https://img.shields.io/badge/Google Scholar-Link-lightblue"> 
    
2. <strong>Sina Khanmohammadi:</strong>  </a> <a href="https://scholar.google.co.uk/citations?hl=en&user=K6sMFj4AAAAJ&view_op=list_works&sortby=pubdate" target="_blank">
        <img src="https://img.shields.io/badge/Google Scholar-Link-lightblue">

## Instructions

### Installation

1. **Prerequisites:** Make sure you have the following installed:
   * Python (3.6 or later)
   * Required libraries:  `argparse`, `numpy`, `pandas`, `pickle`, `logging`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, `networkx`, `torch`, `torch_geometric` 
   * Required folders:  Be sure the following folders are present to save files that are created during experimental runs: `data`, `plots`, `results`, `trials`

# **MSGCN Experiments**

## **Usage**
To run all experiments use the `main.py` script with default arguments as shown below:

```bash
python main.py
```

Any of the default arguments can also be overridden as shown in the following example command:
```bash
python main.py -num_trials 5 -data_path "./datasets" -epochs 100 -lr 0.001
```

## **Command-Line Arguments**
The table below describes the available command-line arguments that can be used:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-num_trials` | `int` | `1` | Number of trials to repeat for the same set of parameters |
| `-data_path` | `str` | `"data"` | Path to the dataset |
| `-epochs` | `int` | `40` | Number of training epochs |
| `-lr` | `float` | `0.0005` | Learning rate |
| `-dim_coor` | `int` | `3` | Number of spatial coordinates |
| `-out_dim` | `int` | `1` | Number of output classes for classification or output nodes for regression |
| `-label_dim` | `int` | `1` | Number of input node features to SGCN layer |
| `-layers_num` | `int` | `1` | Number of SGCN layers |
| `-model_dim` | `int` | `16` | Number of output dimensions from each SGCN layer |
| `-out_channels_1` | `int` | `32` | SGCN hidden layer size (inner convolutions) |
| `-dropout` | `float` | `0.0` | Dropout percentage |
| `-use_cluster_pooling` | `bool` | `False` | Flag to use cluster pooling |
| `-mse_option` | `float` | `1.0` | Custom loss MSE percentage |
| `-spread_option` | `float` | `0.0` | Custom loss spread percentage |
| `-range_option` | `float` | `0.0` | Custom loss range percentage |
| `-graph_types` | `str` | `["complete"]` | Allowed network types: `["complete", "random", "smallworld"]` |
| `-num_graphs` | `int` | `0` | Number of synthetic networks to generate |
| `-num_nodes` | `int (list)` | `[4]` | List of node counts for the graphs |
| `-num_layers` | `int (list)` | `[2]` | List of layer counts for the graphs |
| `-num_neighbors` | `int (list)` | `[2]` | List of k-nearest neighbors (small-world) |
| `-link_probability` | `float (list)` | `[0.3]` | List of edge addition probabilities for small-world networks |


# **Plotting Results**

To visualize all experimental results use the `plot.py` script with default arguments as shown below:

```bash
python plot.py
```

## **Command-Line Arguments**
The table below describes the available command-line arguments for `plot.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-trials` | `int` | `1` | Number of trials to plot |
| `-num_nodes` | `int (list)` | `[4]` | List of node counts for the graphs |
| `-num_layers` | `int (list)` | `[2]` | List of layer counts for the graphs |
| `-graph_types` | `str (list)` | `["complete"]` | Allowed network types: `["complete", "random", "smallworld"]` |
| `-data_path` | `str` | `"trials/"` | Path to the dataset |

# References

[1]  Danel, T. et al. (2020). "Spatial Graph Convolutional Networks." In: Yang, H., Pasupa, K., Leung, A.CS., Kwok, J.T., Chan, J.H., King, I. (eds) Neural Information Processing. ICONIP 2020. Communications in Computer and Information Science, vol 1333. Springer, Cham. [https://doi.org/10.1007/978-3-030-63823-8_76](https://doi.org/10.1007/978-3-030-63823-8_76)
