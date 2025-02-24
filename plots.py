"""
Multiplex Spatial Convolutional Graph Networks (MSGCN)
Author: Steven Wilson, University of Oklahoma, 2024

This file is the plotting program used to create results for the MSGCN project.  
"""

# load common libraries
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", palette="pastel")
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

def create_parser():
    """
    Arguments provided for plotting
    """

    parser = argparse.ArgumentParser(description='MSGCN Plots', fromfile_prefix_chars='@')
    parser.add_argument('-trials', type=int, default=1, help='Number of trials to plot')
    parser.add_argument('-num_nodes', type=int, nargs='+', default=[4], help='List of node counts for the graphs')
    parser.add_argument('-num_layers', type=int, nargs='+', default=[2,3], help='List of layer counts for the graphs')
    parser.add_argument('-graph_types', type=str, nargs='+', default=["complete","random","smallworld"], help='Allowed network types are ["complete","random","smallworld"]')
    parser.add_argument('-data_path', type=str, default='trials/', help='Path to the dataset')

    return parser

def load_data(data_path, trials,num_nodes,num_layers,graph_type):

    # load multilayer graphs based on the graph type
    filename = data_path + 'results_%s_trials_%s_nodes_%s_layers_%s.csv'%(trials,num_nodes,num_layers,graph_type)
    print('loaded data file:', filename)
    df = pd.read_csv(filename)

    return df

def prepare_results(args):
    results_df = pd.DataFrame()
    columns = ['graph_type','n','edge','actual_weight','predicted_weight','edge_loss']

    for graph_type in args.graph_types:
        for n in args.num_nodes:
            for m in args.num_layers:
                df = load_data(args.data_path, args.trials, n, m, graph_type)

                # get rows for final epoch
                max_epoch = df['epoch'].max()
                final_df = df[(df['epoch']==max_epoch) & (df['result_type']=='test')][columns]
                final_df['Layers'] = m

                # Add a new column 'sample_number' based on repeating integers from 1 to the total number of samples for each edge
                final_df['sample_number'] = final_df.groupby(['graph_type', 'edge']).cumcount() + 1

                results_df = pd.concat([results_df,final_df])

    results_df.to_csv('results/all_mse.csv',index=False)

    # Group the dataset by sample_number, graph_type, Layers, and n, then calculate the average edge loss
    average_edge_loss = results_df.groupby(['sample_number', 'graph_type', 'Layers', 'n'])['edge_loss'].mean().reset_index()

    average_edge_loss.to_csv('results/avg_edge_loss.csv', index=False)   

    return results_df, average_edge_loss

# Function to annotate p-value
def annotate_pvalue(ax, x1, x2, y, h, p_value, significance):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((x1+x2)*0.5, y+h+0.005, f'{significance}', ha='center', va='bottom', color='k')  

def plot_robustness_by_network(average_edge_loss):

    # Initialize a dictionary to store p-values for each graph type
    p_values = {}

    # Loop through each graph type
    for graph_type in average_edge_loss['graph_type'].unique():
        # Filter the data for 2-layer and 3-layer networks
        data_2_layers = average_edge_loss[(average_edge_loss['graph_type'] == graph_type) & (average_edge_loss['Layers'] == 2)]['edge_loss']
        data_3_layers = average_edge_loss[(average_edge_loss['graph_type'] == graph_type) & (average_edge_loss['Layers'] == 3)]['edge_loss']
        
        # Perform a t-test if both groups have data
        if not data_2_layers.empty and not data_3_layers.empty:
            t_stat, p_value = ttest_ind(data_2_layers, data_3_layers, equal_var=False)
            p_values[graph_type] = p_value

    # Create a boxplot to visualize the average edge loss
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=average_edge_loss, x='graph_type', y='edge_loss', hue='Layers', palette=["m", "g"], showfliers=False)

    # Capitalize x-axis labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()], fontweight='bold', fontsize=14)

    # Set y-axis limits
    plt.ylim(0, 0.07)

    # Add titles and labels
    plt.title('Robustness by Network and Layers', fontsize=16, fontweight='bold', fontname='Calibri')
    plt.xlabel('', fontsize=12, fontweight='bold', fontname='Calibri')
    plt.ylabel('Edge Error', fontsize=14, fontweight='bold', fontname='Calibri')
    plt.legend(title="Layers", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=True)

    # calculate p-value for Layer comparison
    annotation_positions = [[-0.2, 0.2, 0.04, 0.003],[0.8, 1.2, 0.04, 0.003],[1.8, 2.2, 0.04, 0.003]]
    i = 0
    for key, p_value in p_values.items():
        if p_value > 0.05:
                significance = 'p > 0.05' 
        else:
                significance =  f"{p_value:.2e}"

        x1 = annotation_positions[i][0] # left x
        x2 = annotation_positions[i][1] # right x
        y = annotation_positions[i][2] # height
        h = annotation_positions[i][3] # y position
        annotate_pvalue(ax, x1, x2, y, h, p_value, significance)
        i += 1

    # Remove the bottom border
    sns.despine(left=False, bottom=True)

    # Show the plot
    plt.tight_layout()

    # save the plot
    plt.savefig('plots/robustness_by_network.svg')

    plt.show()

def plot_robustness_by_nodes(average_edge_loss):
    # Initialize a dictionary to store p-values for each graph type
    p_values = {}

    # Loop through each graph type
    for n in average_edge_loss['n'].unique():
        # Filter the data for 2-layer and 3-layer networks
        data_2_layers = average_edge_loss[(average_edge_loss['n'] == n) & (average_edge_loss['Layers'] == 2)]['edge_loss']
        data_3_layers = average_edge_loss[(average_edge_loss['n'] == n) & (average_edge_loss['Layers'] == 3)]['edge_loss']
        
        # Perform a t-test if both groups have data
        if not data_2_layers.empty and not data_3_layers.empty:
            t_stat, p_value = ttest_ind(data_2_layers, data_3_layers, equal_var=False)
            p_values[n] = p_value

    # Create a boxplot to visualize the average edge loss
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=average_edge_loss, x='n', y='edge_loss', hue='Layers', palette=["m", "g"], showfliers=False)

    # Set y-axis limits
    plt.ylim(0, 0.10)

    # Add titles and labels
    plt.title('Robustness by Nodes and Layers', fontsize=16, fontweight='bold', fontname='Calibri')
    plt.xlabel('Number of Nodes', fontsize=14, fontweight='bold', fontname='Calibri')
    plt.ylabel('Edge Error', fontsize=14, fontweight='bold', fontname='Calibri')
    plt.legend(title="Layers", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=True)

    # calculate p-value for Layer comparison
    annotation_positions = [[-0.2, 0.2, 0.07, 0.003],[0.8, 1.2, 0.07, 0.003],[1.8, 2.2, 0.07, 0.003],[2.8, 3.2, 0.07, 0.003],[3.8, 4.2, 0.07, 0.003]]
    i = 0
    for key, p_value in p_values.items():
        if p_value > 0.05:
                significance = 'p > 0.05' 
        else:
                significance =  f"{p_value:.2e}"

        x1 = annotation_positions[i][0] # left x
        x2 = annotation_positions[i][1] # right x
        y = annotation_positions[i][2] # height
        h = annotation_positions[i][3] # y position
        annotate_pvalue(ax, x1, x2, y, h, p_value, significance)
        i += 1

    # Remove the bottom border
    sns.despine(left=False, bottom=True)

    # Show the plot
    plt.tight_layout()

    # save the plot
    plt.savefig('plots/robustness_by_nodes.svg')

    plt.show()

def plot_generalization(data, col="graph_type", row="Layers", hue="n"):
    # Create the FacetGrid with correlation coefficient annotations
    g = sns.FacetGrid(data, col=col, row=row, hue=hue, height=4, aspect=1.2, margin_titles=True)
    g.map(sns.scatterplot, "actual_weight", "predicted_weight", alpha=0.7)
    g.map(add_perfect_accuracy_line)
    g.map(add_corr_coeff, "actual_weight", "predicted_weight")

    # Add a legend and adjust plot settings
    g.add_legend(title="# Nodes", fontsize=10, loc='upper right', bbox_to_anchor=(0.26, 1.0), ncol=5, frameon=True)
    g.set_axis_labels("Actual Weight", "Predicted Weight", fontweight='bold', fontname='Arial')

    # Convert column names to title case manually
    g.col_names = [col.title() for col in g.col_names]
    g.set_titles(row_template="{row_name} Layers", col_template="{col_name} Network", fontweight='bold', fontname='Arial')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Method Generalization", fontweight='bold', fontname='Arial', fontsize=16, ha='center', x=.41)

    plt.show()

# Function to add diagonal line representing perfect accuracy
def add_perfect_accuracy_line(*args, **kwargs):
    ax = plt.gca()
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(limits, limits, '-', color='lightgrey', linewidth=1, alpha=0.2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

# Function to add Pearson correlation coefficient to each subplot
def add_corr_coeff(x, y, **kwargs):
    corr_coeff = np.corrcoef(x, y)[0, 1]
    plt.gca().annotate(f"r = {corr_coeff:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))

#################################################################
if __name__ == "__main__":
    # get command line arguments
    parser = create_parser()
    args = parser.parse_args()

    data, average_edge_loss = prepare_results(args)
    plot_robustness_by_network(average_edge_loss)
    plot_robustness_by_nodes(average_edge_loss)
    plot_generalization(data)
