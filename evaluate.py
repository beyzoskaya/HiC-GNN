import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import GATNetReduced
import torch
from torch.nn import MSELoss
from scipy.stats import spearmanr
import argparse
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create necessary directories if they do not exist
    base_data_dir = 'Data_node2vec_GAT_reduced_GM12878_evaluation'
    base_output_dir = 'Outputs/node2vec_GAT_reduced_GM12878_evaluation'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    parser = argparse.ArgumentParser(description='Evaluate a pre-trained HiC-GNN model.')
    
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
    conversion = 1

    # Generate file path for the input data
    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)
    
    # Check if the input file exists
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{subdataset}_{chromosome}_{resolution}'

    # Load the adjacency matrix from the RAWobserved data
    adj = np.loadtxt(filepath)

    # Convert coordinate list format to full adjacency matrix if needed
    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)  # Remove diagonal elements (self-loops)
    matrix_path = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # Normalize the adjacency matrix using normalize.R script
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt' 
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')
    
    normed = np.loadtxt(normed_matrix_path)

    # G = nx.from_numpy_matrix(adj)
    G = nx.from_numpy_array(adj)  # Changed to numpy_array for running in my desktop

    # Node2Vec model for creating embeddings
    node2vec = Node2Vec(G, dimensions=512, walk_length=80, num_walks=10, workers=4)
    node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([node2vec_model.wv[str(node)] for node in G.nodes()])
    embedding_path = f'Data_node2vec_GAT_reduced_GM12878_evaluation/{name}_embeddings_node2vec_GAT.txt' 
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    # Prepare input data
    data = utils.load_input(normed, embeddings)

    # Load pre-trained model
    model = GATNetReduced()
    model.load_state_dict(torch.load(weights_path))
    print(f'Loaded model weights from {weights_path}')

    truth = utils.cont2dist(data.y, conversion)
    model.eval()

    coords = model.get_model(data.x, data.edge_index)
    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    coords = model.get_model(data.x, data.edge_index)
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]

    # Calculate Spearman's correlation (dSCC)
    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

    print(f'Optimal dSCC (Spearman correlation) for {chromosome}: {SpRho}')

    # Save evaluation results
    eval_log_path = f'Outputs/node2vec_GAT_reduced_GM12878_evaluation/{name}_evaluation_log.txt'
    with open(eval_log_path, 'w') as f:
        f.write(f'Optimal dSCC: {SpRho}\n')

    print(f'Saved evaluation results to {eval_log_path}')
