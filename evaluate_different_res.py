import numpy as np
from node2vec import Node2Vec
import sys
import utils  
import networkx as nx
import os
from models import GATNetHeadsChanged3LayersLeakyReLUv2
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
    base_data_dir = 'Data/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000_alpha_0.01_ChIA-PET_evaluation_500kb_res_evaluation'
    base_data_dir_normalization = 'Data/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000_alpha_0.01'
    base_output_dir = 'Outputs/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000_alpha_0.01_ChIA-PET_evaluation_500kb_res_evaluation'
    
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
    parser.add_argument('-alpha', '--alpha', type=float, default=1, help='Weight for dSCC loss component.')

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset 
    resolution_trained = args.resolution_trained 
    resolution_untrained = args.resolution_untrained  
    chromosome = args.chromosome 
    alpha = args.alpha
    weights_path = args.weights_path  
    conversion = 1  
    # chr1_1mb_RAWobserved.txt
    filepath_trained = os.path.join(dataset_folder, subdataset, resolution_trained, f'{chromosome}_{resolution_trained}_RAWobserved.txt')
    adj_trained = np.loadtxt(filepath_trained)
    if adj_trained.shape[1] == 3:
        adj_trained = utils.convert_to_matrix(adj_trained)
    np.fill_diagonal(adj_trained, 0)
    normed_trained = np.loadtxt(f'{base_data_dir_normalization}/{subdataset}_{chromosome}_{resolution_trained}_matrix_KR_normed.txt')

    G_trained = nx.from_numpy_array(adj_trained)
    node2vec_trained = Node2Vec(G_trained, dimensions=512, walk_length=100, num_walks=20,p=4,q=2.5, workers=1, seed=42)
    node2vec_model_trained = node2vec_trained.fit(window=15, min_count=1, batch_words=4)
    embeddings_trained = np.array([node2vec_model_trained.wv[str(node)] for node in G_trained.nodes()])

    name  = f'{subdataset}_{chromosome}_{resolution_untrained}'
    filepath_untrained = os.path.join(dataset_folder, subdataset, resolution_untrained, f'{chromosome}_{resolution_untrained}.RAWobserved.txt')
    adj_untrained = np.loadtxt(filepath_untrained)
    if adj_untrained.shape[1] == 3:
        adj_untrained = utils.convert_to_matrix(adj_untrained)
    np.fill_diagonal(adj_untrained, 0)
    matrix_path_untrained = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path_untrained, adj_untrained, delimiter='\t')

    os.system(f'Rscript normalize.R {name}_matrix')
    normed_untrained_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt'
    normed_untrained = np.loadtxt(normed_untrained_path)
    

    G_untrained = nx.from_numpy_array(adj_untrained)
    node2vec_untrained =  Node2Vec(G_untrained, dimensions=512, walk_length=100, num_walks=20,p=4,q=2.5, workers=1, seed=42)
    node2vec_model_untrained = node2vec_untrained.fit(window=15, min_count=1, batch_words=4)
    embeddings_untrained = np.array([node2vec_model_untrained.wv[str(node)] for node in G_untrained.nodes()])

    aligned_embeddings = utils.domain_alignment(adj_trained, adj_untrained, embeddings_trained, embeddings_untrained)

    model = GATNetHeadsChanged3LayersLeakyReLUv2()
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
    SpRho = spearmanr(dist_truth.detach().numpy(), dist_out.detach().numpy())[0]
    dSCC_loss = (1 - SpRho)
    combined_loss = mse_value + alpha * dSCC_loss
    drmsd_value = drmsd(dist_out, dist_truth).item()

    print(f'Optimal dSCC (Spearman correlation) for {chromosome}: {SpRho}')
    print(f'MSE for {chromosome}: {mse_value.item()}')
    print(f'dRMSD for {chromosome}: {drmsd_value}')
    print(f'Combined Loss (MSE + {alpha} * dSCC loss) for {chromosome}: {combined_loss.item()}')

    eval_log_path = f'{base_output_dir}/{subdataset}_{chromosome}_{resolution_untrained}_evaluation_log.txt'
    with open(eval_log_path, 'w') as f:
        f.write(f'Optimal dSCC: {SpRho}\n')
        f.write(f'MSE: {mse_value.item()}\n')
        f.write(f'dRMSD: {drmsd_value}\n')
        f.write(f'Combined Loss (MSE + {alpha} * dSCC loss): {combined_loss.item()}\n')

    pdb_file = f'{base_output_dir}/{subdataset}_{chromosome}_{resolution_untrained}_structure.pdb'
    utils.WritePDB(coords * 100, pdb_file)
    print(f'Saved optimal structure to {pdb_file}')

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dist_truth)), dist_truth, label='True Distances', color='blue')
    plt.plot(range(len(dist_out)), dist_out, label='Predicted Distances', color='orange')
    plt.xlabel('Pairwise Distances Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name}_distance_comparison_plot.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(dist_truth.cpu().numpy(), dist_out.cpu().numpy(), alpha=0.5)
    plt.plot([dist_truth.min(), dist_truth.max()], [dist_truth.min(), dist_truth.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel('True Pairwise Distance')
    plt.ylabel('Predicted Pairwise Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name}_scatter_distance_comparison_plot.png')

    plt.figure(figsize=(10, 6))
    plt.hist(dist_truth.cpu().numpy(), bins=50, alpha=0.5, label='True Distances', color='blue')
    plt.hist(dist_out.cpu().numpy(), bins=50, alpha=0.5, label='Predicted Distances', color='orange')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name}_histogram_distance_comparison_plot.png')

    distance_errors = torch.abs(dist_truth - dist_out)
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_truth.cpu().numpy(), distance_errors.cpu().numpy(), alpha=0.5, color='red')
    plt.xlabel('True Pairwise Distance')
    plt.ylabel('Absolute Error')
    plt.savefig(f'{base_output_dir}/{name}_distance_error_plot.png')