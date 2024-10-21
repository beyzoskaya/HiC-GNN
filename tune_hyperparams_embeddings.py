import torch
from node2vec import Node2Vec
import networkx as nx
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from scipy.stats import spearmanr
from torch.nn import MSELoss
import utils
import argparse
import os
from models import GATNetHeadsChanged4LayersLeakyReLU

# I am using created normed matrices for the chromosomes.
def load_normalized_matrix(normed_matrix_path):
    if os.path.exists(normed_matrix_path):
        normed_matrix = np.loadtxt(normed_matrix_path)
        print(f"Loaded normalized matrix from: {normed_matrix_path}")
    else:
        raise FileNotFoundError(f"Normalized matrix file not found: {normed_matrix_path}")
    return normed_matrix

def objective(params):
    p, q, num_walks = params
    print(f"Evaluating for p={p}, q={q}, num_walks={num_walks}")
    
    print(f"Loading adjacency matrix from {filepath}")
    adj = np.loadtxt(filepath)
    adj = utils.convert_to_matrix(adj) if adj.shape[1] == 3 else adj
    np.fill_diagonal(adj, 0)  

    normed = load_normalized_matrix(normed_matrix_path)

    G = nx.from_numpy_array(adj)

    # create embeddings with specified hyper parameters
    print(f"Generating embeddings with Node2Vec for num_walks={num_walks}, p={p}, q={q}")
    node2vec = Node2Vec(G, dimensions=512, walk_length=100, num_walks=int(num_walks), p=p, q=q, workers=1, seed=42)
    node2vec_model = node2vec.fit(window=15, min_count=1, batch_words=4)
    
    embeddings = np.array([node2vec_model.wv[str(node)] for node in G.nodes()])
    
    print(f"Loading pre-trained model from {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at {weights_path}")
    # use the last model
    data = utils.load_input(normed, embeddings)
    model = GATNetHeadsChanged4LayersLeakyReLU()
    model.load_state_dict(torch.load(weights_path)) 

    # get the distances for optimization objective
    print(f"Computing distances")
    with torch.no_grad():
        coords = model.get_model(data.x, data.edge_index)
    
    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = utils.cont2dist(data.y, conversion)[idx[0, :], idx[1, :]]
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]

    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]
    
    mse_loss = MSELoss()
    mse_value = mse_loss(dist_out, dist_truth)
    
    # for better results, I combined the dSCC and mse value as a minimization (min -dSCC --> max dSCC / min MSE --> max -MSE)
    loss = -SpRho + mse_value.item()
    print(f"dSCC: {SpRho}, MSE: {mse_value.item()}, Loss: {loss}")
    
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Node2Vec parameters")

    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., GM12878).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('--weights_path', type=str, help='Path to pre-trained model weights.')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset
    resolution = args.resolution
    chromosome = args.chromosome
    weights_path = args.weights_path
    output_dir = args.output_dir
    conversion = 1  

    name = f'{subdataset}_{chromosome}_{resolution}'
    filepath = os.path.join(dataset_folder, subdataset, resolution, f'{chromosome}_{resolution}_RAWobserved.txt')
    normed_matrix_path = 'Data/GATNetHeadsChanged4LayersLeakyReLU_lr_0.0.0003_dropout_0.3_threshold_1e-8_p_6_q_2_num_walks_50_GM12878_batch_size_128/GM12878_chr1_1mb_matrix_KR_normed.txt'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    space = [
        Real(0.5, 10, name='p'),        # Range for p
        Real(0.1, 5, name='q'),         # Range for q
        Integer(10, 100, name='num_walks')  # Range for num_walks
    ]

    print("Starting Bayesian Optimization...")
    res = gp_minimize(objective, space, n_calls=30, random_state=42)
    print("Bayesian Optimization completed!")

    print(f"Best hyperparameters: p={res.x[0]}, q={res.x[1]}, num_walks={res.x[2]}")

    output_file = os.path.join(output_dir, f'{name}_best_hyperparams.txt')
    with open(output_file, 'w') as f:
        f.write(f"Best hyperparameters:\n")
        f.write(f"p={res.x[0]}\n")
        f.write(f"q={res.x[1]}\n")
        f.write(f"num_walks={res.x[2]}\n")
    print(f"Saved best hyperparameters to: {output_file}")
