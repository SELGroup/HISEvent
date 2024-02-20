import networkx as nx
from SE import SE
from itertools import combinations, chain
import numpy as np

def search_stable_points(embeddings, max_num_neighbors = 200):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    
    all_1dSEs = []
    seg = None
    for i in range(max_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges = [(s+1, d+1, corr_matrix[s, d]) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))
    
    #print('all_1dSEs: ', all_1dSEs)
    stable_indices = []
    for i in range(1, len(all_1dSEs) - 1):
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]:
            stable_indices.append(i)
    if len(stable_indices) == 0:
        print('No stable points found after checking k = 1 to ', max_num_neighbors)
        return 0, 0
    else:
        stable_SEs = [all_1dSEs[index] for index in stable_indices]
        index = stable_indices[stable_SEs.index(min(stable_SEs))]
        print('stable_indices: ', stable_indices)
        print('stable_SEs: ', stable_SEs)
        print('First stable point: k = ', stable_indices[0]+1, ', correspoding 1dSE: ', stable_SEs[0]) # n_neighbors should be index + 1
        print('Global stable point within the searching range: k = ', index + 1, \
            ', correspoding 1dSE: ', all_1dSEs[index]) # n_neighbors should be index + 1
    return stable_indices[0]+1, index + 1 # first stable point, global stable point

def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i+1] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i+1)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))

def get_knn_edges(embeddings, default_num_neighbors):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    knn_edges = []
    for i in range(default_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges += [(s+1, d+1) if s < d else (d+1, s+1) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
    return list(set(knn_edges))

def get_global_edges(attributes, embeddings, default_num_neighbors, e_a = True, e_s = True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(embeddings, default_num_neighbors)
    return list(set(knn_edges + graph_edges))

def get_subgraphs_edges(clusters, graph_splits, weighted_global_edges):
    '''
    get the edges of each subgraph

    clusters: a list containing the current clusters, each cluster is a list of nodes of the original graph
    graph_splits: a list of (start_index, end_index) pairs, each (start_index, end_index) pair indicates a subset of clusters, 
        which will serve as the nodes of a new subgraph
    weighted_global_edges: a list of (start node, end node, edge weight) tuples, each tuple is an edge in the original graph

    return: all_subgraphs_edges: a list containing the edges of all subgraphs
    '''
    all_subgraphs_edges = []
    for split in graph_splits:
        subgraph_clusters = clusters[split[0]:split[1]]
        subgraph_nodes = list(chain(*subgraph_clusters))
        subgraph_edges = [edge for edge in weighted_global_edges if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        all_subgraphs_edges.append(subgraph_edges)
    return all_subgraphs_edges

def hier_2D_SE_mini(weighted_global_edges, n_messages, n = 100):
    '''
    hierarchical 2D SE minimization
    '''
    ite = 1
    # initially, each node (message) is in its own cluster
    # node encoding starts from 1
    clusters = [[i+1] for i in range(n_messages)]
    while True:
        print('\n=========Iteration ', str(ite), '=========')
        n_clusters = len(clusters)
        graph_splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)] # [s, e)
        all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)
        last_clusters = clusters
        clusters = []
        for i, subgraph_edges in enumerate(all_subgraphs_edges):
            print('\tSubgraph ', str(i+1))
            g = nx.Graph()
            g.add_weighted_edges_from(subgraph_edges)
            seg = SE(g)
            seg.division = {j: cluster for j, cluster in enumerate(last_clusters[graph_splits[i][0]:graph_splits[i][1]])}
            seg.add_isolates()
            for k in seg.division.keys():
                for node in seg.division[k]:
                    seg.graph.nodes[node]['comm'] = k
            seg.update_struc_data()
            seg.update_struc_data_2d()
            seg.update_division_MinSE()

            clusters += list(seg.division.values())
        if len(graph_splits) == 1:
            break
        if clusters == last_clusters:
            n *= 2
    return clusters
