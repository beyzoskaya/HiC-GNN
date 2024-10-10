import numpy as np
from ge import LINE 
import sys
import utils
import networkx as nx
import os
from models import Net  
import torch
from torch.nn import MSELoss
from scipy.stats import spearmanr
import argparse
import json
import matplotlib.pyplot as plt

def drmsd(dist_pred, dist_true):
    return torch.sqrt(((dist_pred - dist_true) ** 2).mean())

if __name__ == "__main__":
    base_data_dir = 'Data/Data_HiC_GNN_LINE_evaluation'
    base_output_dir = 'Outputs/HiC_GNN_LINE_evaluation'
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    parser = argparse.ArgumentParser(description='Evaluate a pre-trained HiC-GNN model using LINE embeddings.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., GM12878).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('weights_path', type=str, help='Path to pre-trained model weights.')

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset  # Sub-dataset name, e.g., GM12878
    resolution = args.resolution  # Resolution subfolder, e.g., 1mb
    chromosome = args.chromosome  # Chromosome, e.g., chr12
    weights_path = args.weights_path  # Path to the pre-trained weights
    conversion = 1  # No conversion factor grid search, just using 1

    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)
    
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{subdataset}_{chromosome}_{resolution}'

    adj = np.loadtxt(filepath)
    print(f'Shape of RAWobserved data: {adj.shape}')

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)  # Remove diagonal elements (self-loops)
    matrix_path = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt' 
    normed = np.loadtxt(normed_matrix_path)
    print(f"Shape of normed data: {normed.shape}")
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')
    
    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_array(adj)  # Use adjacency matrix to build graph

    line = LINE(G, embedding_size=512, order='second')  
    line.train(batch_size=128, epochs=10) 
    embeddings = np.array([line.get_embeddings()[node] for node in G.nodes()])
    print(f"Shape of LINE embeddings: {embeddings.shape}")
    embedding_path = f'{base_data_dir}/{name}_embeddings_LINE.txt' 
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    data = utils.load_input(normed, embeddings)

    model = Net()  
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    truth = utils.cont2dist(data.y, conversion)

    with torch.no_grad():
        coords = model.get_model(data.x, data.edge_index)
        print(f"Coordinations: {coords}")

    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
    
    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

    mse_loss = MSELoss()
    mse_value = mse_loss(dist_out, dist_truth)
    drmsd_value = drmsd(dist_out, dist_truth).item()
    
    print(f'Optimal dSCC (Spearman correlation) for {chromosome}: {SpRho}')
    print(f'MSE for {chromosome}: {mse_value.item()}')
    print(f'dRMSD for {chromosome}: {drmsd_value}')

    eval_log_path = f'{base_output_dir}/{name}_evaluation_log.txt'
    with open(eval_log_path, 'w') as f:
        f.write(f'Optimal dSCC: {SpRho}\n')
        f.write(f'MSE: {mse_value.item()}\n')
        f.write(f'dRMSD: {drmsd_value}\n')

    pdb_file = f'{base_output_dir}/{name}_evaluation_structure.pdb'
    utils.WritePDB(coords * 100, pdb_file)
    print(f'Saved optimal structure to {pdb_file}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dist_truth)), dist_truth, label='True Distances', color='blue')
    plt.plot(range(len(dist_out)), dist_out, label='Predicted Distances', color='orange') 
    plt.xlabel('Pairwise Distances Index')
    plt.ylabel('Distance')
    plt.title(f'Pairwise Distance Comparison for {name}')
    plt.legend()
    loss_plot_path = f'{base_output_dir}/{name}_distance_comparison_plot.png'
    plt.savefig(loss_plot_path)
    print(f'Saved loss comparison plot as {loss_plot_path}')
