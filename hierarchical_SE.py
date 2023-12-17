'''
global graph edges + global NN edges
global cooccurrence matrix? -> local
#===========
edges higher threshold (in proportion to 1st layer cluster sizes) min(cluster_1_size, cluster_2_size),
iterate through all the edges, find those have one node in cluster_1 and the other in cluster_2.
'''
import sys
import math
import pickle
import numpy as np
import pandas as pd
from itertools import combinations, chain
from graph.build_graph import BuildGraph, decode_two_dimension_tree, aggregate, insert_edges, evaluate
import datetime
from algorithm.high_dimensional_structural_entropy_algorithm import HighDimensionalStructureEntropyAlgorithm
from algorithm.priority_tree import compute_structural_entropy_of_node
import torch
from graph.graph import Graph
import collections
from sklearn.cluster import KMeans
from os.path import exists

def eval(block):
    save_path = './temp_' + str(block) + '.pkl'
    with open(save_path, 'rb') as f:
        decoded_labels, all_n_clusters, all_labels = pickle.load(f)
    #print('all_n_clusters[1]:\n', all_n_clusters[1], '\n')
    #print('all_labels:\n', all_labels, '\n')
    '''
    all_n_clusters = all_n_clusters[:-1]
    all_labels = all_labels[:-1]
    decoded_labels = decode(all_n_clusters, all_labels)
    print('decoded_labels: ', decoded_labels)
    '''

    save_path = './data/SBERT_embedding/'
    folder = save_path +  str(block) + '/'
    df_path = folder + str(block) + '.csv'
    incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
    labels_true = incr_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    nmi, ami, ari = evaluate(labels_true, decoded_labels)
    print('n_clusters pred: ', all_n_clusters[-1])
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    '''
    nmi:  0.6140448048297716
    ami:  0.5043068108842466
    ari:  0.21974903590535685
    '''

def test_cat():
    graph_splits_cluster_reprs = [torch.rand(3, 5)]
    graph_splits_cluster_reprs.append(torch.rand(3, 5))
    graph_splits_cluster_reprs.append(torch.rand(1, 5))
    print('graph_splits_cluster_reprs: ', graph_splits_cluster_reprs)
    result = torch.cat(graph_splits_cluster_reprs, dim = 0)
    print('result: ', result)

def test_get_attributes():
    '''
    a = [0, 0, 1, 1, 0, 1, 1]
    indices = [i for i in range(len(a)) if a[i] == 1]
    print('indices: ', indices)
    '''
    attributes = [['aa', 'a'], ['bb', 'b'], ['cc', 'c'], ['dd', 'd'], ['ee', 'e'], ['ff', 'f', 'ee'], ['gg', 'g']]
    '''
    indices = [0, 2]
    #print(attributes[3:6][indices])
    result = [attributes[3:6][i] for i in indices]
    print('result: ', result)
    result_2 = []
    for each in result:
        result_2 += each
    print('result_2: ', result_2)
    result_3 = list(chain(*result))
    print('result_3: ', result_3)
    '''
    graph_splits = [(0, 3), (3, 6), (6, 7)]
    graph_splits_labels = [[0, 0, 1], [0, 1, 1], [0]]
    graph_splits_n_clusters = [2, 2, 1]
    result = get_attributes(attributes, graph_splits, graph_splits_labels, graph_splits_n_clusters)
    print('result: ', result)
    return

def get_attributes(attributes, graph_splits, graph_splits_labels, graph_splits_n_clusters):
    new_attributes = []
    for i, n_clusters in enumerate(graph_splits_n_clusters):
        for c in range(n_clusters):
            split = graph_splits[i]
            indices = [index for index in range(len(graph_splits_labels[i])) if graph_splits_labels[i][index] == c]
            new_attributes.append(list(set(chain(*[attributes[split[0]:split[1]][index] for index in indices]))))
    return new_attributes

def decode(all_n_clusters, all_labels):
    #print('all_labels: ', all_labels)
    all_labels = [[n + sum(all_n_clusters[i][:j]) for j, c in enumerate(all_labels[i]) for n in c] \
        for i in range(len(all_labels))]
    #print('all_labels: ', all_labels)
    for i in range(len(all_n_clusters) - 1):
        index = len(all_n_clusters) - i - 2
        #print('\nindex: ', index)
        dic = {i:v for i, v in enumerate(all_labels[index+1])}
        #print('dic: ', dic)
        all_labels[index] = [dic[n] for n in all_labels[index]]
        #print('all_labels[index]: ', all_labels[index])
        
    decoded_labels = all_labels[0]
    return decoded_labels

def test_decode():
    '''
    # test case 1:
    n = 3
    all_n_clusters = [[1] * 10, [2, 2, 1, 1], [2]]

    l_0 = [[0] for i in range(10)]
    l_1 = [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]]
    l_2 = [[0, 1, 0, 1, 1, 1]]
    all_l = []
    all_l.append(l_0)
    all_l.append(l_1)
    all_l.append(l_2)
    #print(all_l) # [[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]], [[0, 1, 0, 1, 1, 1]]]
    '''
    '''
    all_l = [[n + sum(all_n_clusters[i][:j]) for j, c in enumerate(all_l[i]) for n in c] for i in range(len(all_l))]
    #print(all_l) # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 1, 2, 3, 3, 4, 4, 4, 5], [0, 1, 0, 1, 1, 1]]

    for i in range(len(all_n_clusters) - 1):
        index = len(all_n_clusters) - i - 2
        print('index: ', index)
        dic = {i:v for i, v in enumerate(all_l[index+1])}
        print('dic: ', dic)
        all_l[index] = [dic[n] for n in all_l[index]]
        print('all_l[index]: ', all_l[index])
        print()
    '''
    # test case 2:
    print('\ntest case 2')
    n = 3
    all_n_clusters = [[1] * 10, [2, 2, 1, 1], [3, 2], [3]]

    l_0 = [[0] for i in range(10)]
    l_1 = [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]]
    l_2 = [[0, 1, 2], [0, 1, 0]]
    l_3 = [[0, 1, 2, 1, 2]]
    all_l = []
    all_l.append(l_0)
    all_l.append(l_1)
    all_l.append(l_2)
    all_l.append(l_3)

    decoded_labels = decode(all_n_clusters, all_l)
    print('decoded_labels: ', decoded_labels)
    
    # test case 3:
    print('\ntest case 3')
    n = 3
    all_n_clusters = [[1] * 10, [2, 2, 1, 1]]

    l_0 = [[0] for i in range(10)]
    l_1 = [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]]
    all_l = []
    all_l.append(l_0)
    all_l.append(l_1)

    decoded_labels = decode(all_n_clusters, all_l)
    print('decoded_labels: ', decoded_labels)
    
    # test case 4:
    print('\ntest case 4')
    n = 3
    all_n_clusters = [[1] * 10, [2, 2, 1, 1], [3, 2], [3, 2], [3, 2], [3, 2]]

    l_0 = [[0] for i in range(10)]
    l_1 = [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]]
    l_2 = [[0, 1, 2], [0, 1, 0]]
    l_3 = [[0, 1, 2], [0, 1]]
    l_4 = [[0, 1, 2], [0, 1]]
    l_5 = [[0, 1, 2], [0, 1]]
    all_l = []
    all_l.append(l_0)
    all_l.append(l_1)
    all_l.append(l_2)
    all_l.append(l_3)
    all_l.append(l_4)
    all_l.append(l_5)

    decoded_labels = decode(all_n_clusters, all_l)
    print('decoded_labels: ', decoded_labels)
    return

def test_aggregate():
    '''
    a = torch.rand(5,3)
    indices = [1, 3, 4]
    indices = torch.tensor(indices)
    b = torch.index_select(a, 0, indices)
    weights = [0.1, 0.2, 0.7]
    weights = torch.tensor(weights)
    weights = torch.unsqueeze(weights, -1)
    print('weights.shape: ', weights.shape)
    c = torch.mul(b, weights)
    print(a)
    print(b)
    print(c)
    c = torch.sum(c, 0)
    print(c)
    '''
    a = torch.rand(5,3)
    indices = [1]
    indices = torch.tensor(indices)
    b = torch.index_select(a, 0, indices)
    weights = [1]
    weights = torch.tensor(weights)
    weights = torch.unsqueeze(weights, -1)
    print('weights.shape: ', weights.shape)
    c = torch.mul(b, weights)
    print(a)
    print(b)
    print(c)
    c = torch.sum(c, 0)
    print(c)
    return

def test_get_SE():
    n_nodes = 5
    edges = [(1, 2, 2), (1, 3, 4)]
    graph = Graph(n_nodes)
    graph = insert_edges(graph, edges)

    two_dimension_tree = HighDimensionalStructureEntropyAlgorithm(graph).two_dimension()
    labels_pred, n_clusters_pred = decode_two_dimension_tree(two_dimension_tree.get_root(), graph.vertices_number)
    print('\tn_clusters_pred: ', n_clusters_pred)
    print('\tlabels_pred: ', labels_pred)

    SE = 0
    for children in two_dimension_tree.get_root().get_children():
        SE += children.get_structural_entropy_of_node()
    print('SE: ', SE) # SE:  0.7233083338141042

    SE_2 = 0
    for node in two_dimension_tree.get_root().get_children():
        print('node: \n', node)
        gi = node.get_cut()
        degree_sum = graph.get_degree_sum()
        vi = node.get_own_volumn()
        vifa = graph.get_degree_sum()
        node_SE = compute_structural_entropy_of_node(gi, degree_sum, vi, vifa)
        print('gi: ', gi)
        print('degree_sum: ', degree_sum)
        print('vi: ', vi)
        print('vifa: ', vifa)
        print('node_SE: \n', node_SE, '\n')
        SE_2 += node_SE
        #node_se = 0
        #for children in node.get_children():
        #    node_se += compute_structural_entropy_of_node(children.get_cut(), 2 * graph.get_degree_sum(), children.get_own_volumn(), node.get_own_volumn())
        #SE_2 += node_se
    print('SE_2: ', SE_2) # SE_2:  0.7233083338141042
    #assert SE == SE_2 # False

    SE_3 = two_dimension_tree.get_root().get_structural_entropy_of_node()
    print('SE_3: ', SE_3) # SE_3:  0.0

    #SE_gt = - ((6/12)*math.log2(6/8) + (2/12)*math.log2(2/8) + (4/12)*math.log2(4/12) + (4/12)*math.log2(8/12))
    SE_gt = - ((4/12)*math.log2(4/12) + (4/12)*math.log2(8/12))
    print('SE_gt: ', SE_gt) # SE_gt:  0.7233083338141042
    return

def test_combinations():
    l = ['a', 'b', 'c', 'd']
    result = combinations(l, 2)
    print(list(result))

    l = ['a', 'c', 'b', 'd']
    result = combinations(l, 2)
    print(list(result))

    l.sort()
    result = combinations(l, 2)
    print(list(result))

    a = {1: [1, 2, 3], 2: [2, 3, 1], 3: [3, 1, 2]}
    print(a)
    for i in a.keys():
        a[i].sort()
    print(a)
    return

def get_graph_edges(attributes):
    #print('attributes: ', attributes)
    #print('len(attributes): ', len(attributes))
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

def get_global_edges(ini_embeddings, ini_attributes, max_neighbor = 30, first_stable = True, default_num_neighbors = 10):
    '''
    # Choose K and get knn edges
    print('\tChoose K and get knn edges:')
    corr_matrix = np.corrcoef(ini_embeddings)
    np.fill_diagonal(corr_matrix, 0)
    bg = BuildGraph(corr_matrix)
    bg.build(max_neighbor = max_neighbor, first_stable = first_stable, default_num_neighbors = default_num_neighbors)
    knn_edges = bg.get_knn_edges()
    print('\tnumber of knn edges: ', len(knn_edges))
    '''
    if default_num_neighbors != 0:
        # Choose K and get knn edges
        print('\tChoose K and get knn edges:')
        corr_matrix = np.corrcoef(ini_embeddings)
        np.fill_diagonal(corr_matrix, 0)
        bg = BuildGraph(corr_matrix)
        #bg.build(max_neighbor = max_neighbor, first_stable = first_stable, default_num_neighbors = default_num_neighbors)
        bg.index = default_num_neighbors
        knn_edges = bg.get_knn_edges()
        print('\tnumber of knn edges: ', len(knn_edges))

    # Get graph edges
    print('\n\tGet graph edges:')
    graph_edges = get_graph_edges(ini_attributes)
    print('\tnumber of graph edges: ', len(graph_edges))

    # Merge the two to get global edges
    if default_num_neighbors != 0:
        global_edges = list(set(knn_edges + graph_edges))
    else:
        global_edges = graph_edges
    print('\tnumber of global edges: ', len(global_edges))
    return global_edges

def get_cluster_edges(global_edges, all_n_clusters, all_labels):
    #print('\n In get_cluster_edges()...')
    # get decoded_labels
    decoded_labels = decode(all_n_clusters, all_labels)
    decoded_labels = [l+1 for l in decoded_labels] # cluster (new node) indexing starts from 1
    # get cluster_n_nodes
    cluster_n_nodes = collections.Counter(decoded_labels)
    #print('cluster_n_nodes: ', cluster_n_nodes)
    # map global edges
    #mapped_global_edges = [(decoded_labels[edge[0]], decoded_labels[edge[1]]) for edge in global_edges]
    mapped_global_edges = []
    for edge in global_edges:
        #print(edge[0], ' ', edge[1])
        mapped_global_edges.append((decoded_labels[edge[0]-1], decoded_labels[edge[1]-1]))
    # filter out intra-cluster edges, adjust edges so that start always < end
    filtered_mapped_global_edges = []
    for edge in mapped_global_edges:
        if edge[0] != edge[1]:
            if edge[0] < edge[1]:
                filtered_mapped_global_edges.append(edge)
            else:
                filtered_mapped_global_edges.append((edge[1], edge[0]))
    # count inter-cluster edges
    cluster_edges_count = collections.Counter(filtered_mapped_global_edges)
    # filter inter-cluster edges by counts
    #threshold = max(cluster_n_nodes[edge[0]], cluster_n_nodes[edge[1]])
    cluster_edges = [edge for edge in cluster_edges_count.keys() \
        #if cluster_edges_count[edge] >= min(cluster_n_nodes[edge[0]], cluster_n_nodes[edge[1]])]
        #if cluster_edges_count[edge] >= (cluster_n_nodes[edge[0]] + cluster_n_nodes[edge[1]])/2]
        if cluster_edges_count[edge] >= max(cluster_n_nodes[edge[0]], cluster_n_nodes[edge[1]])]
        #if cluster_edges_count[edge] >= cluster_n_nodes[edge[0]]*cluster_n_nodes[edge[1]]/2]
        #if cluster_edges_count[edge] >= cluster_n_nodes[edge[0]]*cluster_n_nodes[edge[1]]]
        #if cluster_edges_count[edge] >= threshold]
    return cluster_edges

def test_get_cluster_edges():
    '''
    decoded_labels = [0, 0, 1, 2, 3, 3, 4, 4, 4, 5]
    cluster_n_nodes = collections.Counter(decoded_labels)
    print('cluster_n_nodes: ', cluster_n_nodes)
    
    a = [0, 0, 1, 2, 3, 3, 4, 4, 4, 5]
    b = [each for each in a if each <= 3]
    print('b: ', b)
    '''
    return

def get_split_edges_obsolete(cluster_edges, splits):
    splits_edges = []
    for split in splits:
        s = split[0] + 1 # node indexing starts from 1
        e = split[1]
        edges = [edge for edge in cluster_edges if (edge[0] >= s and edge[1] <= e)]
        mapped_edges = [(edge[0]-split[0], edge[1]-split[0]) for edge in edges] # map so that the node indexing of each split starts from 1
        splits_edges.append(mapped_edges)
    return splits_edges

def get_split_edges(cluster_edges, splits):
    splits_edges = []
    for split in splits:
        s = split[0] + 1 # node indexing starts from 1
        e = split[1]
        edges = [edge for edge in cluster_edges if (edge[0] >= s and edge[1] <= e)]
        mapped_edges = [(edge[0]-split[0], edge[1]-split[0]) for edge in edges] # map so that the node indexing of each split starts from 1
        splits_edges.append(mapped_edges)
    return splits_edges

def test_get_split_edges():
    n_clusters = 10
    n = 3
    splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)]
    print('splits: ', splits)
    cluster_edges = [(1, 2), (1, 3), (1, 4), (4, 10), (5, 6), (7, 8), (7, 9), (8, 9)]
    splits_edges = get_split_edges(cluster_edges, splits)
    print('splits_edges: ', splits_edges)
    return

def hierarchical_SE(ini_embeddings, ini_attributes, n = 100, default_num_neighbors = 30, save_path = None):
    '''
    ini_embeddings: a list of embeddings of all the nodes in the graph.
    ini_attributes: a list of attributes of all the nodes in the graph.
    n: max number of nodes in each subgraph.
    '''
    embeddings = ini_embeddings
    attributes = ini_attributes
    n_clusters = len(ini_embeddings)
    all_SE = [0]
    all_n_clusters = [[1] * n_clusters] # e.g.: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 1, 1], [2]]
    all_labels = [[[0] for i in range(n_clusters)]] # e.g.: [[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], [[0, 0, 1], [0, 1, 1], [0, 0, 0], [0]], [[0, 1, 0, 1, 1, 1]]]

    if save_path is not None:
        global_edges_path = save_path + 'global_edges.pkl'
        if not exists(global_edges_path):
            global_edges = get_global_edges(ini_embeddings, ini_attributes, max_neighbor = 1, first_stable = True, default_num_neighbors = default_num_neighbors)
            with open(global_edges_path, 'wb') as f:
                pickle.dump(global_edges, f)
        else:
            with open(global_edges_path, 'rb') as f:
                global_edges = pickle.load(f)
    else:
        global_edges = get_global_edges(ini_embeddings, ini_attributes, max_neighbor = 1, first_stable = True, default_num_neighbors = default_num_neighbors)

    cluster_edges = global_edges

    while True:
        #print('n_clusters: ', n_clusters)
        graph_splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)] # [s, e), indexing starts from 0
        #print('graph_splits: ', graph_splits)
        splits_edges = get_split_edges(cluster_edges, graph_splits)

        graph_splits_n_clusters = []
        #graph_splits_SEs = []
        graph_splits_labels = []
        graph_splits_cluster_reprs = []
        for i, split in enumerate(graph_splits):
        #for i, split in enumerate(graph_splits[:1]):
            print('\n', datetime.datetime.now(), str(i) + 'th subgraph...')
            split_embeddings = embeddings[split[0]:split[1]]

            corr_matrix = np.corrcoef(split_embeddings)
            #print('corr_matrix: ', corr_matrix)
            #print('np.shape(corr_matrix): ', np.shape(corr_matrix))

            try:
                np.fill_diagonal(corr_matrix, 0)
                #print('corr_matrix: ', corr_matrix)
            except ValueError:
                assert corr_matrix == 1 # the split has only one node
                n_clusters_pred = 1
                labels_pred = [0]
                reprs = split_embeddings
                graph_splits_n_clusters.append(n_clusters_pred)
                graph_splits_labels.append(labels_pred)
                graph_splits_cluster_reprs.append(reprs)
                continue

            graph = Graph(corr_matrix.shape[0])
            edges_w_weights = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in splits_edges[i] \
                if corr_matrix[edge[0]-1, edge[1]-1] > 0]
            print('len(edges_w_weights): ', len(edges_w_weights))
            graph = insert_edges(graph, edges_w_weights)

            '''
            bg = BuildGraph(corr_matrix)
            
            # Use knn edges to construct graph
            print('\tUse knn edges to construct graph:')
            # Insert edges via common attributes
            if len(all_n_clusters) == 1:
                split_attributes = attributes[split[0]:split[1]]
                split_graph_edges = get_graph_edges(split_attributes)
                #max_neighbor = None
                max_neighbor = 1
            else:
                split_graph_edges = []
                max_neighbor = 1
            #print('split_graph_edges: ', split_graph_edges)
            #print('len(split_graph_edges): ', len(split_graph_edges))
            graph = bg.build(graph_path = None, graph_edges = split_graph_edges, max_neighbor = max_neighbor, first_stable = True, default_num_neighbors = 10)
            '''
            print('\tvertices_number: ', graph.vertices_number) 
            print('\tdegree_sum: ', graph.degree_sum)

            print('\tCalculate 2D SE:')
            print('\t', datetime.datetime.now())
            algorithm = HighDimensionalStructureEntropyAlgorithm(graph)
            two_dimension_tree = algorithm.two_dimension()
            reprs = []
            for children in two_dimension_tree.get_root().get_children():
                reprs.append(aggregate(children, split_embeddings, graph))
            reprs = torch.stack(reprs)
            #print('reprs: ', reprs)

            labels_pred, n_clusters_pred = decode_two_dimension_tree(two_dimension_tree.get_root(), graph.vertices_number)
            '''
            SE = 0
            for children in two_dimension_tree.get_root().get_children():
                SE += children.get_structural_entropy_of_node()
            #print('SE: ', SE)
            '''

            # print('\tn_clusters_pred: ', n_clusters_pred)
            # print('\tlabels_pred: ', labels_pred)
            # print('\treprs: ', reprs)
            # print('\treprs.shape: ', reprs.shape)
            #exit()
        
            graph_splits_n_clusters.append(n_clusters_pred)
            #graph_splits_SEs.append(SE)
            graph_splits_labels.append(labels_pred)
            graph_splits_cluster_reprs.append(reprs)

        all_n_clusters.append(graph_splits_n_clusters)
        all_labels.append(graph_splits_labels)
        #all_SE.append(sum(graph_splits_SEs))
        #if all_SE[-1] == all_SE[-2] or len(all_n_clusters[-1]) == 1:
        #if len(all_n_clusters[-1]) == 1 or all_n_clusters[-1] == all_n_clusters[-2]:
        #if len(all_n_clusters[-1]) == 1 and all_n_clusters[-1] == all_n_clusters[-2]:

        if save_path is not None:
            all_n_clusters_path = save_path + 'all_n_clusters.pkl'
            all_labels_path = save_path + 'all_labels.pkl'
            with open(all_n_clusters_path, 'wb') as f:
                pickle.dump(all_n_clusters, f)
            with open(all_labels_path, 'wb') as f:
                pickle.dump(all_labels, f)

        if len(all_n_clusters[-1]) == 1:
            break
        if all_n_clusters[-1] == all_n_clusters[-2]:
            n *= 2
        
        # prepare for the next iteration
        embeddings = torch.cat(graph_splits_cluster_reprs, dim = 0)
        #attributes = get_attributes(attributes, graph_splits, graph_splits_labels, graph_splits_n_clusters)
        cluster_edges = get_cluster_edges(global_edges, all_n_clusters, all_labels)
        n_clusters = len(embeddings)
        # shuffle


    # decode
    decoded_labels = decode(all_n_clusters, all_labels)

    return decoded_labels, all_n_clusters, all_labels

def test_1203():
    block = 12

    save_path = './data/SBERT_embedding/'
    folder = save_path +  str(block) + '/'
    embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    attributes_root_path = './data/df_split_1113/'
    attributes_path = attributes_root_path + str(block) + '/' + str(block) + '.npy'
    df_np = np.load(attributes_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['original_index', 'event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases', 'date'])
    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
    #print('all_node_features[:10]: ', all_node_features[:10])
    
    for default_num_neighbors in [3, 10, 30, 50, 60, 70, 80, 90]:
    #for default_num_neighbors in [3, 10, 30, 50]:
        print('\n\n====================================================')
        print('default_num_neighbors: ', default_num_neighbors)
        decoded_labels, all_n_clusters, all_labels = hierarchical_SE(ini_embeddings = embeddings, ini_attributes = all_node_features, default_num_neighbors = default_num_neighbors)
        save_path = './temp_' + str(block) + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([decoded_labels, all_n_clusters, all_labels], f)
        # with open(save_path, 'rb') as f:
        #     decoded_labels, all_n_clusters, all_labels = pickle.load(f)
        print('decoded_labels:\n', decoded_labels, '\n')
        print('all_n_clusters:\n', all_n_clusters[1:], '\n')
        print('all_labels:\n', all_labels[1:], '\n')
        print('len(all_n_clusters):\n', len(all_n_clusters), '\n')
        print('default_num_neighbors: ', default_num_neighbors, ' result:')
        eval(block)
    
    return

def test_all_1203(n = 100):
    for i in range(21):
        block = i + 1
        print('\n\n====================================================')
        print('block: ', block)

        save_path = './data/SBERT_embedding/'
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        attributes_root_path = './data/df_split_1113/'
        attributes_path = attributes_root_path + str(block) + '/' + str(block) + '.npy'
        df_np = np.load(attributes_path, allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=['original_index', 'event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
            'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases', 'date'])
        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
        #print('all_node_features[:10]: ', all_node_features[:10])
        
        default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        decoded_labels, all_n_clusters, all_labels = hierarchical_SE(ini_embeddings = embeddings, ini_attributes = all_node_features, n = n, default_num_neighbors = default_num_neighbors)
        save_path = './temp_' + str(block) + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([decoded_labels, all_n_clusters, all_labels], f)
        # with open(save_path, 'rb') as f:
        #     decoded_labels, all_n_clusters, all_labels = pickle.load(f)
        print('decoded_labels:\n', decoded_labels, '\n')
        print('all_n_clusters:\n', all_n_clusters[1:], '\n')
        print('all_labels:\n', all_labels[1:], '\n')
        print('len(all_n_clusters):\n', len(all_n_clusters), '\n')
        print('block: ', block)
        print('default_num_neighbors: ', default_num_neighbors, ' result:')
        eval(block)
        '''
        for default_num_neighbors in [3, 10, 30, 50, 60, 70, 80, 90]:
        #for default_num_neighbors in [3, 10, 30, 50]:
            print('\n\n====================================================')
            print('default_num_neighbors: ', default_num_neighbors)
            decoded_labels, all_n_clusters, all_labels = hierarchical_SE(ini_embeddings = embeddings, ini_attributes = all_node_features, default_num_neighbors = default_num_neighbors)
            save_path = './temp_' + str(block) + '.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump([decoded_labels, all_n_clusters, all_labels], f)
            # with open(save_path, 'rb') as f:
            #     decoded_labels, all_n_clusters, all_labels = pickle.load(f)
            print('decoded_labels:\n', decoded_labels, '\n')
            print('all_n_clusters:\n', all_n_clusters[1:], '\n')
            print('all_labels:\n', all_labels[1:], '\n')
            print('len(all_n_clusters):\n', len(all_n_clusters), '\n')
            print('default_num_neighbors: ', default_num_neighbors, ' result:')
            eval(block)
        '''
    return

def test_all_stable_points_1203(n = 150):
    #first_stable_points = [167, 169, 29, 24, 14, 12, 109, 7, 39, 7, 34, 0, 13, 24, 0, 11, 11, 176, 11, 31, 137]
    #stable_points = [167, 169, 29, 24, 112, 63, 109, 7, 39, 105, 34, 0, 13, 24, 0, 11, 11, 176, 14, 31, 137]
    #max_neighbor = 170
    first_stable_points = [167, 169, 29, 24, 14, 12, 109, 7, 39, 7, 34, 0, 13, 24, 0, 11, 11, 0, 11, 31, 137]
    stable_points = [167, 169, 29, 24, 112, 63, 109, 7, 39, 105, 34, 0, 13, 24, 0, 11, 11, 0, 14, 31, 137]
    #for i in range(21):
    for i in [4, 5, 9, 18]:
        block = i + 1
        print('\n\n====================================================')
        print('block: ', block)

        save_path = './data/SBERT_embedding/'
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        attributes_root_path = './data/df_split_1113/'
        attributes_path = attributes_root_path + str(block) + '/' + str(block) + '.npy'
        df_np = np.load(attributes_path, allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=['original_index', 'event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
            'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases', 'date'])
        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
        #print('all_node_features[:10]: ', all_node_features[:10])
        
        #default_num_neighbors = first_stable_points[i]
        default_num_neighbors = stable_points[i]
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        decoded_labels, all_n_clusters, all_labels = hierarchical_SE(ini_embeddings = embeddings, ini_attributes = all_node_features, n = n, default_num_neighbors = default_num_neighbors)
        save_path = './temp_' + str(block) + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([decoded_labels, all_n_clusters, all_labels], f)
        # with open(save_path, 'rb') as f:
        #     decoded_labels, all_n_clusters, all_labels = pickle.load(f)
        print('decoded_labels:\n', decoded_labels, '\n')
        print('all_n_clusters:\n', all_n_clusters[1:], '\n')
        print('all_labels:\n', all_labels[1:], '\n')
        print('len(all_n_clusters):\n', len(all_n_clusters), '\n')
        print('block: ', block)
        print('default_num_neighbors: ', default_num_neighbors, ' result:')
        eval(block)
        
    return

def generate_latex(file_path = './nohup_kmeans_results.out'):
    nmi, ami, ari = [], [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    #print(lines)
    for line in lines:
        if line.startswith('nmi:'):
            nmi.append(float(line.split()[1]))
        elif line.startswith('ami:'):
            ami.append(float(line.split()[1]))
        elif line.startswith('ari:'):
            ari.append(float(line.split()[1]))
    #print('nmi: ', nmi)
    #print('ami: ', ami)
    #print('ari: ', ari)
    '''
    nmi_str = 'nmi'
    for each in nmi:
        nmi_str += ' & '
        nmi_str += '{:.2f}'.format(each)
    print('nmi_str: ', nmi_str)
    '''
    nmi_str = 'nmi'
    for each in nmi:
        nmi_str += ' & '
        nmi_str += '{:.2f}'.format(each)
    print('\nnmi_str: ', nmi_str)
    print('avg_nmi: ', sum(nmi)/len(nmi))

    ami_str = 'ami'
    for each in ami:
        ami_str += ' & '
        ami_str += '{:.2f}'.format(each)
    print('\nami_str: ', ami_str)
    print('avg_ami: ', sum(ami)/len(ami))

    ari_str = 'ari'
    for each in ari:
        ari_str += ' & '
        ari_str += '{:.2f}'.format(each)
    print('\nari_str: ', ari_str)
    print('avg_ari: ', sum(ari)/len(ari))
    return

def close_set():
    '''
    # Load labels of all tweets
    # load data (68841 tweets, multiclasses filtered)
    df_np = np.load('./data/event2012_offline/68841_tweets_multiclasses_filtered_0722.npy', allow_pickle=True)
    print("Data loaded.")
    df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
        "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
        "words", "filtered_words", "sampled_words"])
    print("Data converted to dataframe.")
    print('df.head(10): ', df.head(10))
    tweet_ids = df['tweet_id'].values
    labels = df['event_id'].values
    '''
    
    # load binary test mask
    mask_path = './data/event2012_offline/masks/'
    # load binary test mask
    test_mask = torch.load(mask_path + 'test_mask.pt').cpu().detach().numpy()
    # convert binary mask to indices
    test_mask = list(np.where(test_mask==True)[0])
    #print('test_mask: ', test_mask)
    print('len(test_mask): ', len(test_mask)) # len(test_mask):  13769

    # ground_truth = [labels[i] for i in test_mask]
    # n_clusters = len(list(set(ground_truth)))
    # print('n_clusters: ', n_clusters) # n_clusters:  488

    # sample_ids = [tweet_ids[i] for i in test_mask]
    # print('len(sample_ids): ', len(sample_ids)) # len(sample_ids):  13769


    df_path_2 = './data/preprocessed_df_1113.npy'
    df_np_2 = np.load(df_path_2, allow_pickle=True)
    df_2 = pd.DataFrame(data=df_np_2, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])
    #tweet_ids_2 = df_2['tweet_id'].values
    labels_2 = df_2['event_id'].values
    #sample_ids_2 = [tweet_ids_2[i] for i in test_mask]
    ground_truth_2 = [labels_2[i] for i in test_mask]
    #assert sample_ids_2 == sample_ids
    #assert ground_truth_2 == ground_truth

    # load embeddings of the test set messages
    embedding_path = './data/event2012_offline/test_set_embeddings/SBERT_embeddings.pkl'
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    test_u = [df_2['user_id'].values[i] for i in test_mask]
    test_um = [df_2['user_mentions'].values[i] for i in test_mask]
    test_hs = [df_2['hashtags'].values[i] for i in test_mask]
    test_e = [df_2['entities'].values[i] for i in test_mask]
    assert len(test_u) == len(test_um) == len(test_hs) == len(test_e) == 13769
    
    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(test_u, test_um, test_hs, test_e)]
    
    #default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
    default_num_neighbors = 10
    n = 1000

    save_path = './data/event2012_offline/result/n_' + str(n) + '_neighbor_' + str(default_num_neighbors) + '/'
    decoded_labels, all_n_clusters, all_labels = hierarchical_SE(ini_embeddings = embeddings, ini_attributes = all_node_features, n = n, default_num_neighbors = default_num_neighbors, save_path = save_path)

    result_path = save_path + 'result.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump([decoded_labels, all_n_clusters, all_labels], f)

    nmi, ami, ari = evaluate(ground_truth_2, decoded_labels)
    print('n_clusters pred: ', all_n_clusters[-1])
    print('len(all_n_clusters): ', len(all_n_clusters))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    '''
    n = 100
    save_path = './data/event2012_offline/result/n_' + str(n) + '/'
    result_path = save_path + 'result.pkl'
    with open(result_path, 'rb') as f:
        decoded_labels, all_n_clusters, all_labels = pickle.load(f)
    print('len(all_n_clusters): ', len(all_n_clusters)) # len(all_n_clusters): 5
    print('n_clusters pred: ', all_n_clusters[-1]) # 54
    '''
    return

def close_set_search_k():

    # load embeddings of the test set messages
    embedding_path = './data/event2012_offline/test_set_embeddings/SBERT_embeddings.pkl'
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    # Choose K and get knn edges
    print('\tChoose K and get knn edges:')
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    bg = BuildGraph(corr_matrix)
    bg.build(max_neighbor = 138, first_stable = False)
    #knn_edges = bg.get_knn_edges()
    #print('\tnumber of knn edges: ', len(knn_edges))

    return

def close_set_kmeans():
    
    # load binary test mask
    mask_path = './data/event2012_offline/masks/'
    # load binary test mask
    test_mask = torch.load(mask_path + 'test_mask.pt').cpu().detach().numpy()
    # convert binary mask to indices
    test_mask = list(np.where(test_mask==True)[0])
    #print('test_mask: ', test_mask)
    #print('len(test_mask): ', len(test_mask)) # len(test_mask):  13769

    df_path_2 = './data/preprocessed_df_1113.npy'
    df_np_2 = np.load(df_path_2, allow_pickle=True)
    df_2 = pd.DataFrame(data=df_np_2, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])
    labels_2 = df_2['event_id'].values
    ground_truth_2 = [labels_2[i] for i in test_mask]
    #n_clusters = len(list(set(ground_truth_2)))
    n_clusters = 110

    # load embeddings of the test set messages
    embedding_path = './data/event2012_offline/test_set_embeddings/SBERT_embeddings.pkl'
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    decoded_labels = kmeans.labels_.tolist()

    nmi, ami, ari = evaluate(ground_truth_2, decoded_labels)
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    return

if __name__ == '__main__':
    '''
    block = 12

    save_path = './data/SBERT_embedding/'
    folder = save_path +  str(block) + '/'
    embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    attributes_root_path = './data/df_split_1113/'
    attributes_path = attributes_root_path + str(block) + '/' + str(block) + '.npy'
    df_np = np.load(attributes_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['original_index', 'event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases', 'date'])

    #print('df.head(10): \n', df.head(10))
    #df = df.head(10)
    # all_node_features = [[str(u)] + \
    #             [str(each) for each in um] + \
    #             [h.lower() for h in hs] + \
    #             e + \
    #             n
    #             for u, um, hs, e, n in \
    #             zip(df['user_id'], df['user_mentions'], df['hashtags'], df['entities'], df['noun_phrases'])]
    #print(df['noun_phrases'].to_numpy()) # noun_phrases to be fixed

    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
    #print('all_node_features[:10]: ', all_node_features[:10])
    
    decoded_labels, all_n_clusters, all_labels = hierarchical_SE(embeddings, all_node_features)
    save_path = './temp_' + str(block) + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump([decoded_labels, all_n_clusters, all_labels], f)
    with open(save_path, 'rb') as f:
        decoded_labels, all_n_clusters, all_labels = pickle.load(f)
    print('decoded_labels:\n', decoded_labels, '\n')
    print('all_n_clusters:\n', all_n_clusters, '\n')
    print('all_labels:\n', all_labels, '\n')
    '''
    #test_combinations()
    #test_get_SE()
    #test_aggregate()
    #test_decode()
    #test_get_attributes()
    #test_cat()
    #eval(block)
    #test_get_graph_edges()
    #get_global_edges(embeddings, all_node_features)
    #test_get_cluster_edges()
    #test_get_split_edges()

    #test_1203()
    #test_all_1203()
    #test_all_1203(n = 150) # nohup_1203_all_n150.out
    #test_all_stable_points_1203(n = 150) # nohup_1203_all_n150_first_stable_points.out nohup_1203_all_n150_stable_points.out
    #generate_latex()
    #generate_latex('./nohup_1203_all_n150_first_stable_points.out')
    #close_set()
    close_set_kmeans()
    #close_set_search_k()