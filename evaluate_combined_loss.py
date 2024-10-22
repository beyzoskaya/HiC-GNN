import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
import torch
from torch.nn import MSELoss
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt
from models import GATNetHeadsChanged3LayersLeakyReLUv2

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_dRMSD(truth_distances, predicted_distances):
    squared_diff = torch.pow(truth_distances - predicted_distances, 2)
    mean_squared_diff = torch.mean(squared_diff)
    dRMSD = torch.sqrt(mean_squared_diff)
    return dRMSD.item()

if __name__ == "__main__":
    base_data_dir = 'Data/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000_ChIA-PET_evaluation'
    base_output_dir = 'Outputs/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000_ChIA-PET_evaluation'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
    
    parser = argparse.ArgumentParser(description='Evaluate a HiC-GAT model with combined MSE + dSCC loss.')
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., GM12878).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('weights_path', type=str, help='Path to pre-trained model weights.')
    parser.add_argument('-alpha', '--alpha', type=float, default=1, help='Weight for dSCC loss component.')

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset
    resolution = args.resolution
    chromosome = args.chromosome
    weights_path = args.weights_path
    alpha = args.alpha
    conversion = 1 

    # Load dataset
    filename = f'{chromosome}_{resolution}.txt'
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)

    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{subdataset}_{chromosome}_{resolution}'
    adj = np.loadtxt(filepath)
    print(f'Shape of contact map data: {adj.shape}')

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)  # Remove diagonal elements (self-loops)
    matrix_path = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # KR normalization
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt'
    normed = np.loadtxt(normed_matrix_path)

    # Build the graph from adjacency matrix
    G = nx.from_numpy_array(adj)
    node2vec = Node2Vec(G, dimensions=512, walk_length=100, num_walks=20,p=4,q=2.5, workers=1, seed=42) 
    node2vec_model= node2vec.fit(window=15, min_count=1, batch_words=4)

    embeddings = np.array([node2vec_model.wv[str(node)] for node in G.nodes()])
    print(f'Shape of node2vec embeddings: {embeddings.shape}')
    embedding_path = f'{base_data_dir}/{name}_embeddings_node2vec_GAT.txt'
    np.savetxt(embedding_path, embeddings)

    # Load input data
    data = utils.load_input(normed, embeddings)

    # Load pre-trained model
    model = GATNetHeadsChanged3LayersLeakyReLUv2()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Ground truth pairwise distances
    truth = utils.cont2dist(data.y, conversion)

    # Evaluation with the model
    with torch.no_grad():
        coords = model.get_model(data.x, data.edge_index)

    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]

    # Metrics calculation
    SpRho, _ = spearmanr(dist_truth.cpu().numpy(), dist_out.cpu().numpy())
    mse_loss = MSELoss()
    mse_value = mse_loss(dist_out, dist_truth).item()
    dRMSD_value = calculate_dRMSD(dist_truth, dist_out)

    # Calculate combined loss (same as training)
    dSCC_loss = (1 - SpRho)  # same as in training
    combined_loss = mse_value + alpha * dSCC_loss

    print(f'Optimal dSCC: {SpRho}')
    print(f'MSE: {mse_value}')
    print(f'dRMSD: {dRMSD_value}')
    print(f'Combined Loss: {combined_loss}')

    # Save evaluation results
    eval_log_path = f'{base_output_dir}/{name}_evaluation_log.txt'
    with open(eval_log_path, 'w') as f:
        f.write(f'Optimal dSCC: {SpRho}\n')
        f.write(f'MSE: {mse_value}\n')
        f.write(f'dRMSD: {dRMSD_value}\n')
        f.write(f'Combined Loss: {combined_loss}\n')

    # Save 3D structure
    pdb_file = f'{base_output_dir}/{name}_evaluation_structure.pdb'
    utils.WritePDB(coords * 100, pdb_file)

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
