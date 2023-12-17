import sys
from graph.graph import Graph
from graph.edge import Edge
from numpy import sort
import copy
from algorithm.high_dimensional_structural_entropy_algorithm import HighDimensionalStructureEntropyAlgorithm
import pickle
import numpy as np
from algorithm.priority_tree import compute_structural_entropy_of_node
import datetime
import math
import copy
from math import isclose
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import pandas as pd
import torch

def kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_.tolist()

def evaluate(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari

def update(original_graph, original_SE, updated_graph, affected_nodes):
    original_degree_sum = original_graph.degree_sum
    updated_degree_sum = updated_graph.degree_sum
    updated_SE = (original_degree_sum / updated_degree_sum) * (original_SE - math.log2(original_degree_sum / updated_degree_sum))
    for i in affected_nodes:
        d_i_original = original_graph.get_vertice_degree_list()[i]
        d_i_updated = updated_graph.get_vertice_degree_list()[i]
        if not d_i_original == d_i_updated:
            updated_SE -= compute_structural_entropy_of_node(d_i_original, updated_degree_sum, d_i_original, updated_degree_sum)
            updated_SE += compute_structural_entropy_of_node(d_i_updated, updated_degree_sum, d_i_updated, updated_degree_sum)
    return updated_SE

def test_update():
    graphs = []
    SEs = []

    n_nodes = 5
    edges = [(1, 2, 2), (1, 3, 4)]
    graph = Graph(n_nodes)
    graph = insert_edges(graph, edges)
    graphs.append(graph)

    SE = HighDimensionalStructureEntropyAlgorithm(graph).one_dimension()
    print('\nSE: ', SE) #SE:  1.4591479170272446
    SE_calculated = - ((6/12)*math.log2(6/12) + (2/12)*math.log2(2/12) + (4/12)*math.log2(4/12))
    print('SE_calculated: ', SE_calculated)
    assert SE == SE_calculated
    SEs.append(SE)

    # insert one edge
    updated_graph_1 = copy.deepcopy(graphs[-1])
    updated_edges_1 = [(1, 4, 2)]
    #updated_edges_1 = [(1, 4, 2), (1, 4, 2), (4, 1, 2), (4, 1, 2)]
    updated_graph_1 = insert_edges(updated_graph_1, updated_edges_1)
    updated_SE_1 = update(graphs[-1], SEs[-1], updated_graph_1, [1, 4])
    print('\nupdated_SE_1: ', updated_SE_1)
    updated_SE_1_calculated = - ((8/16)*math.log2(8/16) + (2/16)*math.log2(2/16) + (4/16)*math.log2(4/16) + (2/16)*math.log2(2/16))
    print('updated_SE_1_calculated: ', updated_SE_1_calculated)
    assert isclose(updated_SE_1, updated_SE_1_calculated)
    graphs.append(updated_graph_1)
    SEs.append(updated_SE_1)

    # insert one more edge
    updated_graph_2 = copy.deepcopy(graphs[-1])
    updated_edges_2 = [(2, 5, 4)]
    #updated_edges_2 = [(2, 5, 4), (2, 5, 4), (5, 2, 4), (1, 4, 2)]
    updated_graph_2 = insert_edges(updated_graph_2, updated_edges_2)
    updated_SE_2 = update(graphs[-1], SEs[-1], updated_graph_2, [2, 5])
    print('\nupdated_SE_2: ', updated_SE_2)
    updated_SE_2_calculated = - ((8/24)*math.log2(8/24) + (6/24)*math.log2(6/24) + \
        (4/24)*math.log2(4/24) + (2/24)*math.log2(2/24) + (4/24)*math.log2(4/24))
    print('updated_SE_2_calculated: ', updated_SE_2_calculated)
    assert isclose(updated_SE_2, updated_SE_2_calculated)

    for g in graphs:
        print('\nvertices_number: ', g.vertices_number) # 3
        print('edges_number: ', g.edges_number) # 0
        print('degree_sum: ', g.degree_sum) # 12.0
        print('vertice_degree_list: ', g.vertice_degree_list)
    return

def test_sort_corr_matrix():
    store_path = '../data/toy/'
    embeddings_toy_path = store_path + 'embeddings_toy.pkl'
    with open(embeddings_toy_path, 'rb') as f:
        node_embeddings_toy = pickle.load(f)
    #print('node_embeddings_toy.size(): ', node_embeddings_toy.size()) # torch.Size([9, 384])
    node_embeddings_toy = node_embeddings_toy.cpu().clone().detach().numpy()
    #print('node_embeddings_toy: ', node_embeddings_toy)

    corr_matrix = np.corrcoef(node_embeddings_toy)
    print('corr_matrix: ', corr_matrix)
    #print('np.shape(corr_matrix): ', np.shape(corr_matrix))

    np.fill_diagonal(corr_matrix, 0)
    print('corr_matrix: ', corr_matrix)

    #corr_matrix_sorted = np.sort(corr_matrix)
    #print('corr_matrix_sorted: ', corr_matrix_sorted)

    corr_matrix_sorted_indices = np.argsort(corr_matrix) # indices of the least similar neighbor, ..., the most similar neighbor
    print('corr_matrix_sorted_indices: ', corr_matrix_sorted_indices)
    return

def inspect_stable_points():
    save_path = '../data/SBERT_embedding/'
    for block in range(1, 22):
        print('==========block '+ str(block) + '===========')
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        '''
        df_path = folder + str(block) + '.csv'
        incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
        labels_true = incr_df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters: ', n_clusters)
        '''
        
        corr_matrix = np.corrcoef(embeddings)
        #print('corr_matrix: ', corr_matrix)
        #print('np.shape(corr_matrix): ', np.shape(corr_matrix))

        np.fill_diagonal(corr_matrix, 0)
        #print('corr_matrix: ', corr_matrix)

        bg = BuildGraph(corr_matrix)
        '''
        graphs = bg.build(ini_graph_path)
        for g in graphs:
            print('\nvertices_number: ', g.vertices_number) 
            #print('edges_number: ', g.edges_number) 
            print('degree_sum: ', g.degree_sum) 
            #print('vertice_degree_list: ', g.vertice_degree_list)
        '''
        
        graph = bg.build(max_neighbor = 200)
        print('\nvertices_number: ', graph.vertices_number) 
        #print('edges_number: ', graph.edges_number) 
        print('degree_sum: ', graph.degree_sum, '\n')
    return

def run_kmeans():
    save_path = '../data/SBERT_embedding/'
    for block in range(1, 22):
        print('==========block '+ str(block) + '===========')
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_path = folder + str(block) + '.csv'
        incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
        labels_true = incr_df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters: ', n_clusters)
        
        labels_pred = kmeans(embeddings, n_clusters)
        nmi, ami, ari = evaluate(labels_true, labels_pred)
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari, '\n')
    
    return

def run_knn_graph(max_neighbor = 31):
    save_path = '../data/SBERT_embedding/'
    graph_root_path = '../data/df_split_1113/'
    #for block in range(1, 22):
    for block in [2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 16, 18, 19]: # small graphs
        print('\n==========block '+ str(block) + '===========')
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_path = folder + str(block) + '.csv'
        incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
        labels_true = incr_df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters: ', n_clusters)
        
        corr_matrix = np.corrcoef(embeddings)
        #print('corr_matrix: ', corr_matrix)
        #print('np.shape(corr_matrix): ', np.shape(corr_matrix))

        np.fill_diagonal(corr_matrix, 0)
        #print('corr_matrix: ', corr_matrix)

        bg = BuildGraph(corr_matrix)
        
        # Use knn edges to construct graph
        print('\tUse knn edges to construct graph:')
        graph = bg.build(max_neighbor = max_neighbor, first_stable = True, default_num_neighbors = 30)
        print('\tvertices_number: ', graph.vertices_number) 
        print('\tdegree_sum: ', graph.degree_sum)

        algorithm = HighDimensionalStructureEntropyAlgorithm(graph)
        two_dimension_tree = algorithm.two_dimension()
        labels_pred, n_clusters_pred = decode_two_dimension_tree(two_dimension_tree.get_root(), graph.vertices_number)
        print('\tn_clusters_pred: ', n_clusters_pred)
        print('\tlabels_pred: ', labels_pred)

        nmi, ami, ari = evaluate(labels_true, labels_pred)
        print('\tnmi: ', nmi)
        print('\tami: ', ami)
        print('\tari: ', ari)

        # Insert common attribute edges into the graph
        print('\n\tInsert common attribute edges into the graph:')
        graph_path = graph_root_path + str(block) + '/' + str(block) + '.txt'
        graph = bg.update_graph(graph, graph_path)
        print('\tvertices_number: ', graph.vertices_number) 
        print('\tdegree_sum: ', graph.degree_sum)

        algorithm = HighDimensionalStructureEntropyAlgorithm(graph)
        two_dimension_tree = algorithm.two_dimension()
        labels_pred, n_clusters_pred = decode_two_dimension_tree(two_dimension_tree.get_root(), graph.vertices_number)
        print('\tn_clusters_pred: ', n_clusters_pred)
        print('\tlabels_pred: ', labels_pred)

        nmi, ami, ari = evaluate(labels_true, labels_pred)
        print('\tnmi: ', nmi)
        print('\tami: ', ami)
        print('\tari: ', ari)

    return

def run_knn(max_neighbor = 31):
    save_path = '../data/SBERT_embedding/'
    #for block in range(1, 22):
    for block in [3, 4, 5, 6, 8, 9, 10, 11, 13, 16, 18, 19]: # small graphs except 2
        print('\n==========block '+ str(block) + '===========')
        folder = save_path +  str(block) + '/'
        embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_path = folder + str(block) + '.csv'
        incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
        labels_true = incr_df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters: ', n_clusters)
        
        corr_matrix = np.corrcoef(embeddings)
        #print('corr_matrix: ', corr_matrix)
        #print('np.shape(corr_matrix): ', np.shape(corr_matrix))

        np.fill_diagonal(corr_matrix, 0)
        #print('corr_matrix: ', corr_matrix)

        bg = BuildGraph(corr_matrix)
        
        # Use knn edges to construct graph
        print('\tUse knn edges to construct graph:')
        print('\t', datetime.datetime.now())
        graph = bg.build(max_neighbor = max_neighbor, first_stable = True, default_num_neighbors = 30)
        print('\tvertices_number: ', graph.vertices_number) 
        print('\tdegree_sum: ', graph.degree_sum)

        print('\tCalculate 2D SE:')
        print('\t', datetime.datetime.now())
        algorithm = HighDimensionalStructureEntropyAlgorithm(graph)
        two_dimension_tree = algorithm.two_dimension()
        labels_pred, n_clusters_pred = decode_two_dimension_tree(two_dimension_tree.get_root(), graph.vertices_number)
        print('\tn_clusters_pred: ', n_clusters_pred)
        print('\tlabels_pred: ', labels_pred)

        print('\tEvaluate:')
        print('\t', datetime.datetime.now())
        nmi, ami, ari = evaluate(labels_true, labels_pred)
        print('\tnmi: ', nmi)
        print('\tami: ', ami)
        print('\tari: ', ari)

    return

class BuildGraph(object):
    '''
    Choose k-nn by dynamically inserting edges and updating 1d SE. 
    Running time O(n^2), while the running time of BuildGraph.build() is O(n^3).
    '''
    def __init__(self, corr_matrix):
        '''
        corr_matrix: (optionally de-noised) Pearson product-moment correlation coefficients 
        of shape n*n, where n is the number of nodes.
        '''
        self.corr_matrix = corr_matrix
        #print('self.corr_matrix: \n', self.corr_matrix)
        # indices of the least similar neighbor, ..., the most similar neighbor
        self.corr_matrix_sorted_indices = np.argsort(corr_matrix)
        self.index = -1

    def build(self, graph_path = None, graph_edges = None, max_neighbor = 30, first_stable = True, default_num_neighbors = 10):
        SEs = []
        graphs = []
        index = None

        # construct initial graph
        ini_graph = Graph(self.corr_matrix.shape[0])
        # num_neighbors = 1 (only connect each node to its most similar neighbor)
        k_ini = 1
        #print('\t', datetime.datetime.now(), ' ', k_ini)
        dst_ids = self.corr_matrix_sorted_indices[:, -k_ini]
        ini_edges = [(s+1, d+1, self.corr_matrix[s, d]) for s, d in enumerate(dst_ids) if self.corr_matrix[s, d] > 0]
        ini_graph = insert_edges(ini_graph, ini_edges)
        graphs.append(ini_graph)
        ini_SE = HighDimensionalStructureEntropyAlgorithm(ini_graph).one_dimension()
        #print('\nini_SE: ', ini_SE)
        SEs.append(ini_SE)

        # dynamically add more neighbors
        if max_neighbor is None:
            max_neighbor = self.corr_matrix.shape[0]
        for k in range(k_ini + 1, max_neighbor + 1):
            #print('\t', datetime.datetime.now(), ' ', k)
            graph = copy.deepcopy(graphs[-1])
            dst_ids = self.corr_matrix_sorted_indices[:, -k]
            edges = [(s+1, d+1, self.corr_matrix[s, d]) for s, d in enumerate(dst_ids) if self.corr_matrix[s, d] > 0]
            #print(k, edges)
            graph = insert_edges(graph, edges)
            affected_nodes = set(list(list(zip(*edges))[0]) + list(list(zip(*edges))[1])) if len(edges) > 0 else []
            #print('affected_nodes: ', affected_nodes)
            SE = update(graphs[-1], SEs[-1], graph, affected_nodes)
            graphs.append(graph)
            SEs.append(SE)

            if first_stable:
                for i in range(1, len(SEs) - 1):
                    if SEs[i] < SEs[i - 1] and SEs[i] < SEs[i + 1]:
                        index = i
                        break
                if index is not None:
                    print('SEs: ', SEs)
                    print('index: ', index)
                    graph = graphs[index]
                    break

        if first_stable:
            if index is None:
                index = min(default_num_neighbors, len(SEs)-1)
                print('SEs: ', SEs)
                print('index: ', index)
                graph = graphs[index]
        else:
            # get stable points and choose the one with minimum 1d SE
            print('SEs: ', SEs)
            stable_indices = [len(SEs) - 1]
            for i in range(1, len(SEs) - 1):
                if SEs[i] < SEs[i - 1] and SEs[i] < SEs[i + 1]:
                    stable_indices.append(i)
            stable_SEs = [SEs[index] for index in stable_indices]
            index = stable_indices[stable_SEs.index(min(stable_SEs))]
            print('stable_indices: ', stable_indices)
            print('stable_SEs: ', stable_SEs)
            print('index: ', index) # n_neighbors should be index + 1
            graph = graphs[index]

        self.index = index

        if graph_path:
            # add edges between messages that share common attributes 
            # (hashtags, user/user mentions, named entities, noun phrases)
            print('\t', datetime.datetime.now(), ' inserting graph edges...')
            graph_df = pd.read_csv(graph_path, sep='\t')
            graph_edges = [(src_id, dst_id, self.corr_matrix[src_id-1, dst_id-1]) 
                for src_id, dst_id in zip(graph_df['src_id'], graph_df['dst_id'])
                if self.corr_matrix[src_id-1, dst_id-1] > 0]
            graph = copy.deepcopy(graphs[index])
            graph = insert_edges(graph, graph_edges)
        elif graph_edges:
            print('\t', datetime.datetime.now(), ' inserting graph edges...')
            graph_edges = [(edge[0], edge[1], self.corr_matrix[edge[0]-1, edge[1]-1]) 
                for edge in graph_edges
                if self.corr_matrix[edge[0]-1, edge[1]-1] > 0]
            print('\t num_graph_edges to be inserted: ', len(graph_edges))
            graph = copy.deepcopy(graphs[index])
            graph = insert_edges(graph, graph_edges)
        
        return graph
    
    def get_knn_edges(self):
        assert self.index != -1
        print('k: ', self.index)
        knn_edges = []
        for i in range(self.index):
            dst_ids = self.corr_matrix_sorted_indices[:, -(i+1)]
            knn_edges += [(s+1, d+1) if s < d else (d+1, s+1) \
                for s, d in enumerate(dst_ids) if self.corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        return list(set(knn_edges))

    def update_graph(self, graph, graph_path):
        print('\t', datetime.datetime.now(), ' inserting graph edges...')
        graph_df = pd.read_csv(graph_path, sep='\t')
        graph_edges = [(src_id, dst_id, self.corr_matrix[src_id-1, dst_id-1]) 
            for src_id, dst_id in zip(graph_df['src_id'], graph_df['dst_id'])
            if self.corr_matrix[src_id-1, dst_id-1] > 0]
        graph = insert_edges(graph, graph_edges)
        return graph

'''
Given node_embeddings (shape: num_nodes * embedding dimension), 
construct correlation coefficient matrix between nodes and
correct all weights in it.
'''
def correct_weights(node_embeddings):
    num_nodes = node_embeddings.shape[0]
    #print('num_nodes: ', num_nodes)
    cf = 1 / (2 * num_nodes)
    cs = np.corrcoef(node_embeddings)
    for i in range(num_nodes):
        cs[i, i] = 0
    cf *= cs.mean()
    cs += cf
    return cs, num_nodes

def aggregate(node, embeddings, graph):
    all_leaves = node.get_all_leaves()
    if len(all_leaves) == 1:
        weights = [1]
    else:
        entropy_of_childtree = node.get_entropy_of_childtree()
        weights = entropy_of_childtree/sum(entropy_of_childtree)
    weights = torch.tensor(weights)
    weights = torch.unsqueeze(weights, -1)
    all_leaves_indices = [int(i)-1 for i in all_leaves]
    all_leaves_indices = torch.tensor(all_leaves_indices)
    repr_node = torch.index_select(embeddings, 0, all_leaves_indices)
    repr_node = torch.sum(torch.mul(repr_node, weights), 0)
    return repr_node

def decode_two_dimension_tree(root, nodes_number):
    label = [0] * nodes_number
    for i, children in enumerate(root.get_children()):
        for node_id in children.get_all_leaves():
            label[int(node_id) - 1] = i
    return np.array(label), len(root.get_children())

def insert_edges(graph, edges):
    for each in edges:
        src_id = each[0]
        dst_id = each[1]
        weight = each[2]
        edge1 = Edge(src_id, dst_id, weight)
        edge2 = Edge(dst_id, src_id, weight)
        bool_src = edge1 in graph.get_vertice_connect_edge_list()[src_id]
        bool_dst = edge2 in graph.get_vertice_connect_edge_list()[dst_id]
        graph.get_vertice_connect_edge_list()[src_id].add(edge1)
        graph.get_vertice_connect_edge_list()[dst_id].add(edge2)

        if not bool_src:
            graph.get_vertice_degree_list()[src_id] += weight
        if not bool_dst:
            graph.get_vertice_degree_list()[dst_id] += weight
        if not bool_src and not bool_dst:
            graph.set_degree_sum(graph.get_degree_sum() + 2 * weight)

    return graph

def test_graph_copy():
    n_nodes = 3
    edges = [(1, 2, 2), (1, 3, 4)]
    graph = Graph(n_nodes)
    graph = insert_edges(graph, edges)
    print('graph.vertices_number: ', graph.vertices_number) # 3
    print('graph.edges_number: ', graph.edges_number) # 0
    print('graph.degree_sum: ', graph.degree_sum) # 12.0
    print()

    updated_graph = copy.deepcopy(graph)
    updated_edges = [(2, 3, 2)]
    updated_graph = insert_edges(updated_graph, updated_edges)
    '''
    # shallow copy affects the original graph
    updated_edges = [(2, 3, 2)]
    updated_graph = insert_edges(graph, updated_edges)
    '''
    print('updated_graph.vertices_number: ', updated_graph.vertices_number) # 3
    print('updated_graph.edges_number: ', updated_graph.edges_number) # 0
    print('updated_graph.degree_sum: ', updated_graph.degree_sum) # 16.0
    print()

    print('graph.vertices_number: ', graph.vertices_number) # 3
    print('graph.edges_number: ', graph.edges_number) # 0
    print('graph.degree_sum: ', graph.degree_sum) # 12.0
    return

if __name__ == '__main__':
    #test_predict()
    #test_graph_copy()
    #test_update()
    #test_sort_corr_matrix()
    #test_BuildGraph_1114()
    #test_BuildGraph_1114_2() # nohup_20_dynamic_2.out/ nohup_20_dynamic_5.out (up to 200 neighbors)
    #test_BuildGraph_1114_2(denoise = False, ini_graph_path = '../data/df_split_1113/20/20.txt') # nohup_20_dynamic_3.out (killed) /nohup_20_dynamic_4.out (up to 500 neighbors)
    #inspect_stable_points()
    #run_kmeans()
    #run_knn_graph()
    #run_knn()
    '''
    data = [(1, 2),(2,'ojaswi'),
        (3,'bobby'),(4, 4),
        (5,'gnanesh')]
  
    # get first element using zip
    #print(list(zip(*data))[0])
    #print(*data) #(1, 'sravan') (2, 'ojaswi') (3, 'bobby') (4, 'rohith') (5, 'gnanesh')
    print(set(list(list(zip(*data))[0]) + list(list(zip(*data))[1])))
    
   
    #SEs = [0, 1, 3, 2, 3, 4, 1, 2] #
    SEs = [9.19327783161424, 9.253419966995224, 9.296385864896376, 9.33154247009026, 9.358691516133973, 9.383644453823466, 9.40586714736586, 9.424726138360429, 9.440134322623601, 9.456201783049, 9.469788211815732, 9.483013546847022, 9.494671671955437, 9.506258548750777, 9.516060248720718, 9.52580118161739, 9.533730509610479, 9.54093457079681, 9.547407170940575, 9.553824315994879, 9.559899874836015, 9.565291356500552, 9.570259924459558, 9.575198794811003, 9.579799214700211, 9.584428680054685, 9.588241248004595, 9.592248865477677, 9.5954012747229, 9.599025798928237, 9.60228534229658, 9.605046813087188, 9.607856480804555, 9.610768604954846, 9.613371588404831, 9.615950555456068, 9.61839024944408, 9.620743580619807, 9.623067099802885, 9.625271823793224, 9.627779583566182, 9.629694496191695, 9.631755573439118, 9.633697941149059, 9.63550448557019, 9.63686991162043, 9.638249468131534, 9.639733463206726, 9.640829259464567, 9.642090678488266, 9.643175446794604, 9.643914444562963, 9.645112762629726, 9.646300219839, 9.647231835248533, 9.647723872313485, 9.648232394406174, 9.648761395545208, 9.649620954123739, 9.650216554115591, 9.650958531088545, 9.651526495648993, 9.652340596790026, 9.652865812858169, 9.653295791033868, 9.653691648055487, 9.65412991188174, 9.654662147144068, 9.655128279714761, 9.655695931469124, 9.656151355511232, 9.656660780335312, 9.657361635390664, 9.657899335068107, 9.658364951073258, 9.658850703601221, 9.659176889412182, 9.65943652634094, 9.659576784925372, 9.659809174367291, 9.660145205596113, 9.660365693744003, 9.660661357333797, 9.660971189437744, 9.66124825468126, 9.66143740668112, 9.661637831409019, 9.66179102505278, 9.661999051152462, 9.662007696330974, 9.662024519863024, 9.662207366460898, 9.662501952269466, 9.66274718106272, 9.663072873169465, 9.663344890264492, 9.663656156565114, 9.664039994112413, 9.664417703895747, 9.664656203186864, 9.66512840916922, 9.665514066261357, 9.665845258755237, 9.66608758578638, 9.666374908230706, 9.666575044107367, 9.666912990579096, 9.667181665717267, 9.667423409685549, 9.667720997891834, 9.668048115068455, 9.668260917280648, 9.668514742946357, 9.668695030863098, 9.668930087340852, 9.669203182339078, 9.669388555774292, 9.66969120030822, 9.669973254772914, 9.670239848013196, 9.670540818291087, 9.67081125629485, 9.671125099337054, 9.671482077308083, 9.671747280747194, 9.672187673865755, 9.672573501677553, 9.672714888494754, 9.673023485477655, 9.67341285144652, 9.673813125629735, 9.674069819986128, 9.674458409459637, 9.674700471984853, 9.674937905854978, 9.675202000127939, 9.675473253959792, 9.675795156822682, 9.676115062911654, 9.676367845452871, 9.676554581436001, 9.676764971748975, 9.67701155574611, 9.677305514004834, 9.677628381837275, 9.677797265927653, 9.678026837199782, 9.678363327232313, 9.67865320997937, 9.678927951453236, 9.679176210260094, 9.679470849364511, 9.67969296222339, 9.679991326625178, 9.680162694016131, 9.680370430121126, 9.680652144804766, 9.680966737816695, 9.681284446582415, 9.68150846132534, 9.681756243698816, 9.682053625391516, 9.682355767466339, 9.68250272639561, 9.682779942932163, 9.683091566234255, 9.683345514232396, 9.68362805211859, 9.683989987383004, 9.684295158298662, 9.68455599411751, 9.684887056135404, 9.685183836675563, 9.685448954457863, 9.685792993934545, 9.686029848565278, 9.686306714107271, 9.686618700318599, 9.686975036521579, 9.687274188029276, 9.68764545500748, 9.687982316529911, 9.688284795018808, 9.688543387032794, 9.68880902120742, 9.688980018703987, 9.689203764492781, 9.689384057845068, 9.68967300780461, 9.689923093785815, 9.690176452887664, 9.690425655972318, 9.690715113055042, 9.691004754439025, 9.691253060718031, 9.69141807878165, 9.69165163742049, 9.691915815564899, 9.692199921489763, 9.692496786477337, 9.692836595377036, 9.69306865595908, 9.693320113885786, 9.69363294167422, 9.693974192548795, 9.69424739921041, 9.694566716044795, 9.694876630549086, 9.695218132456827, 9.69552101269973, 9.695878241738793, 9.696158216656974, 9.69647568418504, 9.696789021239251, 9.69713889995099, 9.69741271867899, 9.69775878828946, 9.698028780527368, 9.698333597740778, 9.698643164703325, 9.698959703212656, 9.699226443914643, 9.699545916130702, 9.69983153183658, 9.700144870157446, 9.70043176370063, 9.70071760933324, 9.700963346954616, 9.7012661990007, 9.701504101753548, 9.701782162180862, 9.702039680211575, 9.702305152124156, 9.702536857379684, 9.702816134541361, 9.703044506158893, 9.703313048550974, 9.7036173683002, 9.70389371104208, 9.704123826377833, 9.704371159344053, 9.70463057968023, 9.704941626074426, 9.705214238488644, 9.70539631816575, 9.705628593708816, 9.705880740016273, 9.706095218553866, 9.706340070310313, 9.706547715867504, 9.706770422443414, 9.707007999486454, 9.707242044525234, 9.707491638249762, 9.707691717493981, 9.707937477199387, 9.708082211472934, 9.708336153869508, 9.708544933391915, 9.708769060754062, 9.708988119416663, 9.709195000995209, 9.709449182548104, 9.709645239116746, 9.709863045472293, 9.71003195923102, 9.710250213079956, 9.71041484927675, 9.710609290379933, 9.710822771890463, 9.710996372939013, 9.711221735842987, 9.711418323947276, 9.711654995812912, 9.711853148435631, 9.712013295618304, 9.712197407391207, 9.71237283508539, 9.712538300386216, 9.712740665527718, 9.712934185641426, 9.71312054549435, 9.71328277393416, 9.713439013156123, 9.713642444886576, 9.71380554251276, 9.713961975958957, 9.714103070143185, 9.71428518553361, 9.714449636910988, 9.714612006548826, 9.714773902729949, 9.71491680614052, 9.715083242275327, 9.715232296494053, 9.71537760048093, 9.715541457898786, 9.715703149331002, 9.715865648606028, 9.716012359373913, 9.716169853111506, 9.716339224016682, 9.716496130478351, 9.7166273321525, 9.716783020019127, 9.716942545081654, 9.71705815570884, 9.717200891961827, 9.717330071499163, 9.717465961063144, 9.717602581541396, 9.717734383605682, 9.717857309695507, 9.718027281884543, 9.718189375699756, 9.718316342697827, 9.718444587618444, 9.718578025634228, 9.718729038490391, 9.718867662414532, 9.718990929032733, 9.719110078848445, 9.719226082533213, 9.719347130181154, 9.71947768004219, 9.719600587450458, 9.719734242705675, 9.71986366721856, 9.719997278705673, 9.720118518746004, 9.720250974979983, 9.720383537137081, 9.72049272930821, 9.72058863631445, 9.72073935043653, 9.720860620026627, 9.720980853069374, 9.721097810604679, 9.721227232816853, 9.721308639921778, 9.72140484396713, 9.72153413098398, 9.721673019053709, 9.721804491844962, 9.721936187626406, 9.722035736843795, 9.722160504270313, 9.72228692010753, 9.722388405303576, 9.722483228620591, 9.722565453184924, 9.722665227669381, 9.722769693748301, 9.722886384180901, 9.72297256830343, 9.723088230216609, 9.723175135521624, 9.723286671157801, 9.723401760284833, 9.72350240691644, 9.723618849756601, 9.723714793265998, 9.723820877076918, 9.723903669829541, 9.7240215870908, 9.724130009336346, 9.724217587934005, 9.724315878129742, 9.724426933303683, 9.724527257818295, 9.724618381837304, 9.724711315945568, 9.72481436409444, 9.724904619734122, 9.724986548138919, 9.725062966668597, 9.725169823165702, 9.725251016423448, 9.725346579784988, 9.725432938600877, 9.72552295224567, 9.725618267703291, 9.725694400585505, 9.725791725684346, 9.72588998438867, 9.725983167704946, 9.726080689217532, 9.726167594533589, 9.726261131598882, 9.726351809577729, 9.726434169232592, 9.726523684598831, 9.726612384735317, 9.72670814730919, 9.726775869024994, 9.726856075808758, 9.726950334208988, 9.727051308846027, 9.727142799598031, 9.727237608705861, 9.727311570452684, 9.727405147912954, 9.727486472821111, 9.7275760175069, 9.7276752236475, 9.727740354429498, 9.727835481100673, 9.727902121981078, 9.727987745765526, 9.728068006731727, 9.728147774187077, 9.728241926851341, 9.728321665582929, 9.728406604626219, 9.728489053817794, 9.728560130555802, 9.728635398975037, 9.728717892666566, 9.728795421672322, 9.72888947863116, 9.728974374894452, 9.729043545264076, 9.729136797255446, 9.729221475058267, 9.729312087796059, 9.729391320624297, 9.729474203254584, 9.729524527878366, 9.729577983240247, 9.729650161504416, 9.729744334374637, 9.72982366566451, 9.72988961611704, 9.729964317110888, 9.7300408300321, 9.730105036182138, 9.730160416257876, 9.730229175093129, 9.730291022366103, 9.730350751645728, 9.730426196375632, 9.730493415981297, 9.730568849142305, 9.730642044010494, 9.730713436451682, 9.730802211010111, 9.73088609376958, 9.730970325815319, 9.731038893995576, 9.73110615193086, 9.731178534429977, 9.731242266608175, 9.731315360951884, 9.731389928863809, 9.731450483268866, 9.731510345526438, 9.731583245644508, 9.731644089446771, 9.731703342792583, 9.731763375306052, 9.731835595946025, 9.731925231368141, 9.731974174561449, 9.73203413126813, 9.73208395851052, 9.732142909354195, 9.732197093722053, 9.732257817501146, 9.732303827425023, 9.732364168975687, 9.732426112683088, 9.73250277400761, 9.73256301981995, 9.732618594730804, 9.732681626056388, 9.732741533259558, 9.732808650764966, 9.732886387981946, 9.732950082600214, 9.733012717142469, 9.733075930871482, 9.73313934999085, 9.733188020925429, 9.7332418435188, 9.733307320141932, 9.733368549341435, 9.733427071518712, 9.733472479152914, 9.733552534820594, 9.733619132005243, 9.733665169339877, 9.733717912976594, 9.733781852891749, 9.733842403200606, 9.733889922891366, 9.733948875178129, 9.734003140162237, 9.734058687710437, 9.734108673601192, 9.734170295679784]
    stable_indices = [len(SEs) - 1]
    for i in range(1, len(SEs) - 1):
        if SEs[i] < SEs[i - 1] and SEs[i] < SEs[i + 1]:
            stable_indices.append(i)
    stable_SEs = [SEs[index] for index in stable_indices]
    index = stable_indices[stable_SEs.index(min(stable_SEs))]
    print('stable_indices: ', stable_indices)
    print('stable_SEs: ', stable_SEs)
    print('index: ', index)
    '''
    