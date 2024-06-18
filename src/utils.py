
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import matplotlib as mpl



def compute_average_matrix(matrices, normalise=False):
    """
    Compute the average confusion matrix from a 3D array of confusion matrices.

    Parameters
    ----------
    matrices : 3D numpy array
        The adjacency matrices to average
    normalise : bool
        Whether to normalise the average matrix

    Returns
    -------
    mean_cm : 2D numpy array
    """
   
    mean_cm = np.zeros((matrices.shape[1],matrices.shape[2]))

    # Iterate over all matrices
    for cm in matrices:
        # Iterate over all rows
        for i in range(matrices.shape[1]):
            # Iterate over all columns
            for j in range(matrices.shape[2]):
                # Add value to mean matrix
                mean_cm[i,j] += cm[i,j]

    # Divide by number of matrices to get mean
    mean_cm = mean_cm / matrices.shape[0]

    # Normalise
    if normalise and np.max(mean_cm) != 0:
        mean_cm = mean_cm / np.max(mean_cm)

    return mean_cm



def plot_matrices(avg_trigger, avg_no_trigger, algorithm, scaling, differences, window_size, features, features_short):
    """
    Plot two adjacency matrices for two causal graphs side by side.

    Parameters
    ----------
    avg_trigger : 2D numpy array
        The adjacency matrix of the causal graph with trigger
    avg_no_trigger : 2D numpy array
        The adjacency matrix of the causal graph without trigger
    algorithm : str
        The name of the causal discvoery algorithm used to compute the causal graph
    scaling : str
        The scaling method used to scale the time series data
    differences : bool
        Whether differences were used as input to compute the causal graph
    window_size : int
        The window size used to compute one causal graph in the MWCD algorithm
    features : list
        List of feature names
    features_short : list
        List of feature names abbreviated

    Returns
    -------
    fig : matplotlib figure
    """

    # Plot matrices of both arrays in two columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))

    im = axs[0].matshow(avg_no_trigger, interpolation='nearest')
    axs[0].set_title(f'{algorithm} - {scaling}-{differences}-{window_size} - No Trigger')

    im2 = axs[1].matshow(avg_trigger, interpolation='nearest')
    axs[1].set_title(f'{algorithm} - {scaling}-{differences}-{window_size} - Trigger')

    axs[0].set_yticks(axs[0].get_yticks().tolist()); axs[0].set_xticks(axs[0].get_xticks().tolist())
    axs[1].set_yticks(axs[1].get_yticks().tolist()); axs[1].set_xticks(axs[1].get_xticks().tolist())

    axs[0].set_yticklabels(features)
    axs[0].set_xticklabels(features_short)

    axs[1].set_yticklabels(features_short)
    axs[1].set_xticklabels(features_short)

    divider = make_axes_locatable(axs[0])
    divider2 = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)

    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')

    return fig



def plot_DAGs(no_trigger, trigger, seed, threshold, labels):
    """
    Plot two directed acyclic graphs (DAGs) side by side.

    Parameters
    ----------
    no_trigger : 2D numpy array
        The adjacency matrix of the causal graph without trigger
    trigger : 2D numpy array
        The adjacency matrix of the causal graph with trigger
    seed : int
        Seed for the spring layout algorithm
    threshold : float
        Threshold for the edges in the adjacency matrix
    labels : dict
        Dictionary with the labels of the nodes

    Returns
    -------
    fig : matplotlib figure
    """

    adj=no_trigger
    G=nx.DiGraph(directed=True)
    G.add_nodes_from(range(len(adj)))
    G.add_edges_from([(i,j) for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold])
    pos = nx.spring_layout(G, seed=seed)
    node_sizes = [1000 for i in range(len(G))]
    edge_colors = [adj[i,j] for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold]
    cmap = plt.cm.Blues

    adj2=trigger
    G2=nx.DiGraph(directed=True)
    G2.add_nodes_from(range(len(adj2)))
    G2.add_edges_from([(i,j) for i in range(len(adj2)) for j in range(len(adj2)) if adj2[i,j] > threshold])
    pos2 = nx.spring_layout(G2, seed=seed)
    node_sizes2 = [1000 for i in range(len(G2))]
    edge_colors2 = [adj2[i,j] for i in range(len(adj2)) for j in range(len(adj2)) if adj2[i,j] > threshold]
    cmap2 = plt.cm.Blues

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Axs0
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="tab:blue", ax=axs[0])
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
        ax=axs[0]
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=axs[0])
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    axs[0].set_axis_off()
    plt.colorbar(pc, ax=axs[0])


    # Axs1
    nodes = nx.draw_networkx_nodes(G2, pos2, node_size=node_sizes2, node_color="tab:blue", ax=axs[1])
    edges = nx.draw_networkx_edges(
        G2,
        pos2,
        node_size=node_sizes2,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors2,
        edge_cmap=cmap2,
        width=2,
        ax=axs[1]
    )
    nx.draw_networkx_labels(G2, pos2, labels, font_size=10, ax=axs[1])
    pc1 = mpl.collections.PatchCollection(edges, cmap=cmap2)
    pc1.set_array(edge_colors2)
    axs[1].set_axis_off()
    plt.colorbar(pc1, ax=axs[1])

    axs[0].set_title('No Trigger')
    axs[1].set_title('Trigger')
        
    return fig



def plot_pruned_DAGs(no_trigger, trigger, threshold, labels, target_nodes, algortihm, scaling, differences, window_size):    
    """
    Plot two pruned directed acyclic graphs (DAGs) side by side.

    Parameters
    ----------
    no_trigger : 2D numpy array
        The adjacency matrix of the causal graph without trigger
    trigger : 2D numpy array
        The adjacency matrix of the causal graph with trigger
    threshold : float
        Threshold for the edges in the adjacency matrix
    labels : dict
        Dictionary with the labels of the nodes
    target_nodes : list
        List of target nodes to filter the adjacency matrix
    algortihm : str
        The name of the causal discvoery algorithm used to compute the causal graph
    scaling : str
        The scaling method used to scale the time series data
    differences : bool
        Whether differences were used as input to compute the causal graph
    window_size : int
        The window size used to compute one causal graph in the MWCD algorithm

    Returns
    -------
    fig : matplotlib figure
    """

    adj=no_trigger
    G=nx.DiGraph(directed=True)
    G.add_nodes_from(range(len(adj)))
    G.add_edges_from([(i,j) for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0])
    pos = nx.spring_layout(G, seed=12)
    G.remove_nodes_from(list(nx.isolates(G)))
    node_sizes = [1000 for i in range(len(G))]
    edge_colors = [adj[i,j] for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0]
    edge_weights = [adj[i,j]*10 for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0]
    cmap = plt.cm.Blues

    adj2=trigger
    G2=nx.DiGraph(directed=True)
    G2.add_nodes_from(range(len(adj2)))
    G2.add_edges_from([(i,j) for i in range(len(adj2)) for j in range(len(adj2)) if adj2[i,j] > threshold and j in target_nodes and i != 0])
    pos2 = nx.spring_layout(G2, seed=12)
    G2.remove_nodes_from(list(nx.isolates(G2)))
    node_sizes2 = [1000 for i in range(len(G2))]
    edge_colors2 = [adj2[i,j] for i in range(len(adj2)) for j in range(len(adj2)) if adj2[i,j] > threshold and j in target_nodes and i != 0]
    edge_weights2 = [adj2[i,j]*10 for i in range(len(adj2)) for j in range(len(adj2)) if adj2[i,j] > threshold and j in target_nodes and i != 0]
    cmap2 = plt.cm.Blues

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Axs0
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="tab:blue", ax=axs[0])
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=edge_weights,
        ax=axs[0],
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=axs[0], font_color='white')
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    axs[0].set_axis_off()
    plt.colorbar(pc, ax=axs[0])


    # Axs1
    nodes1 = nx.draw_networkx_nodes(G2, pos2, node_size=node_sizes2, node_color="tab:blue", ax=axs[1])
    edges1 = nx.draw_networkx_edges(
        G2,
        pos2,
        node_size=node_sizes2,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors2,
        edge_cmap=cmap,
        width=edge_weights2,
        ax=axs[1],
    )
    nx.draw_networkx_labels(G2, pos2, labels, font_size=10, ax=axs[1], font_color='white')
    pc1 = mpl.collections.PatchCollection(edges1, cmap=cmap)
    pc1.set_array(edge_colors2)
    axs[1].set_axis_off()
    plt.colorbar(pc1, ax=axs[1])

    axs[0].set_title('No Trigger')
    axs[1].set_title('Trigger')

    fig.suptitle(f'Pruned DAGs {algortihm}, {scaling}, {differences}, {window_size}')
        
    return fig


def plot_single_pruned_DAG(adj, threshold, labels, target_nodes, title):
    """
    Plot a single pruned directed acyclic graph (DAG).

    Parameters
    ----------
    adj : 2D numpy array
        The adjacency matrix of the causal graph
    threshold : float
        Threshold for the edges in the adjacency matrix
    labels : dict
        Dictionary with the labels of the nodes
    target_nodes : list
        List of target nodes to filter the adjacency matrix
    algortihm : str
        The name of the causal discvoery algorithm used to compute the causal graph
    scaling : str
        The scaling method used to scale the time series data
    differences : bool
        Whether differences were used as input to compute the causal graph
    window_size : int
        The window size used to compute one causal graph in the MWCD algorithm

    Returns
    -------
    fig : matplotlib figure
    """

    G=nx.DiGraph(directed=True)
    G.add_nodes_from(range(len(adj)))
    G.add_edges_from([(i,j) for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0])

    # layers


    pos = nx.spring_layout(G, seed=12)
    G.remove_nodes_from(list(nx.isolates(G)))
    node_sizes = [1000 for i in range(len(G))]
    edge_colors = [adj[i,j] for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0]
    edge_weights = [adj[i,j]*10 for i in range(len(adj)) for j in range(len(adj)) if adj[i,j] > threshold and j in target_nodes and i != 0]
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Axs0
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="tab:blue", ax=ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=edge_weights,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax, font_color='white')
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)

    fig.suptitle(f'{title}')
        
    return fig
