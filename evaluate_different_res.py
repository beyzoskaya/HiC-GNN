import numpy as np
from node2vec import Node2Vec
import sys
import utils  
import networkx as nx
import os
from models import GATNetHeadsChanged4LayersLeakyReLU
import torch
from torch.nn import MSELoss
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def drmsd(dist_pred, dist_true):
    return torch.sqrt(((dist_pred - dist_true) ** 2).mean())

if __name__ == "__main__":
    base_data_dir = 'Data/Data_GATNetHeadsChanged4LayersLeakyReLU_lr_0.0005_dropout_0.3_threshold_1e-8_node2vecParamsChanged_p_1.5_q_0.5_GM12878_ChIA-PET_evaluation'
    base_output_dir = 'Outputs/GATNetHeadsChanged4LayersLeakyReLU_lr_0.0005_dropout_0.3_threshold_1e-8_node2vecParamsChanged_p_1.5_q_0.5_GM12878_ChIA-PET_evaluation'
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    parser = argparse.ArgumentParser(description='Evaluate a pre-trained HiC-GNN model with alignment.')
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., GM12878).')
    parser.add_argument('resolution_trained', type=str, help='Resolution for trained embeddings (e.g., 1mb).')
    parser.add_argument('resolution_untrained', type=str, help='Resolution for untrained embeddings (e.g., 500kb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('weights_path', type=str, help='Path to pre-trained model weights.')

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset 
    resolution_trained = args.resolution_trained 
    resolution_untrained = args.resolution_untrained  
    chromosome = args.chromosome 
    weights_path = args.weights_path  
    conversion = 1  

    filepath_trained = os.path.join(dataset_folder, subdataset, resolution_trained, f'{chromosome}_{resolution_trained}.txt')
    adj_trained = np.loadtxt(filepath_trained)
    if adj_trained.shape[1] == 3:
        adj_trained = utils.convert_to_matrix(adj_trained)
    np.fill_diagonal(adj_trained, 0)
    normed_trained = np.loadtxt(f'{base_data_dir}/{subdataset}_{chromosome}_{resolution_trained}_matrix_KR_normed.txt')

    G_trained = nx.from_numpy_array(adj_trained)
    node2vec_trained = Node2Vec(G_trained, dimensions=512, walk_length=100, num_walks=20, p=2, q=0.5, workers=1, seed=42)
    node2vec_model_trained = node2vec_trained.fit(window=15, min_count=1, batch_words=4)
    embeddings_trained = np.array([node2vec_model_trained.wv[str(node)] for node in G_trained.nodes()])

    filepath_untrained = os.path.join(dataset_folder, subdataset, resolution_untrained, f'{chromosome}_{resolution_untrained}.txt')
    adj_untrained = np.loadtxt(filepath_untrained)
    if adj_untrained.shape[1] == 3:
        adj_untrained = utils.convert_to_matrix(adj_untrained)
    np.fill_diagonal(adj_untrained, 0)
    normed_untrained = np.loadtxt(f'{base_data_dir}/{subdataset}_{chromosome}_{resolution_untrained}_matrix_KR_normed.txt')

    G_untrained = nx.from_numpy_array(adj_untrained)
    node2vec_untrained = Node2Vec(G_untrained, dimensions=512, walk_length=100, num_walks=20, p=2, q=0.5, workers=1, seed=42)
    node2vec_model_untrained = node2vec_untrained.fit(window=15, min_count=1, batch_words=4)
    embeddings_untrained = np.array([node2vec_model_untrained.wv[str(node)] for node in G_untrained.nodes()])

    aligned_embeddings = utils.domain_alignment(adj_trained, adj_untrained, embeddings_trained, embeddings_untrained)

    model = GATNetHeadsChanged4LayersLeakyReLU()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    data_untrained = utils.load_input(normed_untrained, aligned_embeddings)

    truth = utils.cont2dist(data_untrained.y, conversion)

    with torch.no_grad():
        coords = model.get_model(data_untrained.x, data_untrained.edge_index)
    
    idx = torch.triu_indices(data_untrained.y.shape[0], data_untrained.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]

    SpRho = spearmanr(dist_truth.detach().numpy(), dist_out.detach().numpy())[0]

    mse_loss = MSELoss()
    mse_value = mse_loss(dist_out, dist_truth)
    drmsd_value = drmsd(dist_out, dist_truth).item()

    print(f'Optimal dSCC (Spearman correlation) for {chromosome}: {SpRho}')
    print(f'MSE for {chromosome}: {mse_value.item()}')
    print(f'dRMSD for {chromosome}: {drmsd_value}')

    eval_log_path = f'{base_output_dir}/{subdataset}_{chromosome}_{resolution_untrained}_evaluation_log.txt'
    with open(eval_log_path, 'w') as f:
        f.write(f'Optimal dSCC: {SpRho}\n')
        f.write(f'MSE: {mse_value.item()}\n')
        f.write(f'dRMSD: {drmsd_value}\n')

    pdb_file = f'{base_output_dir}/{subdataset}_{chromosome}_{resolution_untrained}_structure.pdb'
    utils.WritePDB(coords * 100, pdb_file)
    print(f'Saved optimal structure to {pdb_file}')
