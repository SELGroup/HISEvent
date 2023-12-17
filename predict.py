import pandas as pd
import numpy as np
import os
from os.path import exists
import re
import pickle
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from graph.build_graph import aggregate, decode_two_dimension_tree
from graph.get_real_network import GetRealNetwork
import datetime
from algorithm.high_dimensional_structural_entropy_algorithm import HighDimensionalStructureEntropyAlgorithm

def evaluate(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari

def predict_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_.tolist()

def predict_SE(graph_file_path, vertices_number, embeddings):
    graph = GetRealNetwork(graph_file_path, vertices_number).get_graph()

    # Construct hierarchical tree of height 2
    print(datetime.datetime.now(), 'constructing hierarchical tree...')
    algorithm = HighDimensionalStructureEntropyAlgorithm(graph)
    two_dimension_tree = algorithm.two_dimension()

    # Decode the hierarchical tree to get:
    # n_clusters: number of clusters in the graph
    # reprs: representations of all the clusters
    # labels: labels of the nodes in the graph
    print(datetime.datetime.now(), 'decoding hierarchical tree...')
    reprs = []
    for children in two_dimension_tree.get_root().get_children():
        reprs.append(aggregate(children, embeddings, graph))
    labels, n_clusters = decode_two_dimension_tree(two_dimension_tree.get_root(), vertices_number)

    return labels, n_clusters, reprs

if __name__ == '__main__':
    '''
    labels_true = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    labels_pred = [0, 1, 1, 1, 0, 2, 2, 2, 2]
    nmi, ami, ari = evaluate(labels_true, labels_pred)
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    '''
    '''
    store_path = './data/df_split/'
    construct_incremental_dataset(df, store_path)
    '''
    '''
    block = 20
    save_path = './data/df_split/'
    folder = save_path +  str(block) + '/'
    df_path = folder + str(block) + '.csv'
    embeddings_path = folder + 'embeddings_' + str(block) + '.pkl'
    incr_df = pd.read_csv(df_path, sep='\t', lineterminator='\n')
    labels_true = incr_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters: ', n_clusters)
    
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f) 

    #labels_pred = kmeans(embeddings, n_clusters)

    labels_pred, n_clusters_pred, reprs_pred = predict(embeddings)
    print('n_clusters_pred: ', n_clusters_pred)
    #print('labels_pred: ', labels_pred)

    nmi, ami, ari = evaluate(labels_true, labels_pred)
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    #==========================
    kmeans block 1
    nmi:  0.39884854606243353
    ami:  0.38029951655349864
    ari:  0.03102626737475962

    kmeans block 20
    ground truth n_clusters:  34
    nmi:  0.8316022492224712
    ami:  0.7986622266123837
    ari:  0.5246332109611228

    SE fully connected block 20
    n_clusters_pred:  16
    nmi:  0.8305027073724546
    ami:  0.8122047282540823
    ari:  0.6757677311847485
    
    #construct_graph()
    
    construct_graph_cos()
    labels_pred, n_clusters_pred, reprs_pred = predict_cos()
    print('n_clusters_pred: ', n_clusters_pred)
    nmi, ami, ari = evaluate(labels_true, labels_pred)
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    '''

    df_np_path = './data/df_split_1113/20/20.npy'
    df_np = np.load(df_np_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['original_index', 'event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases', 'date'])
    #df = pd.DataFrame(data=df_np)
    print('df.head(10): \n', df.head(10))
    print('df.shape[0]: ', df.shape[0])
    
    labels_true = df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters: ', n_clusters)

    graph_file_path = './data/df_split_1113/20/20.txt'
    vertices_number = df.shape[0]
    embedding_path = './data/df_split/20/embeddings_20.pkl'
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f) 
    labels_pred, n_clusters_pred, reprs_pred = predict_SE(graph_file_path, vertices_number, embeddings)
    print('n_clusters_pred: ', n_clusters_pred)
    nmi, ami, ari = evaluate(labels_true, labels_pred)
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    