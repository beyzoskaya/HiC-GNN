import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import GATNetHeadsChanged4LayersLeakyReLU
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import random
import ast
import argparse
import json
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_dRMSD(truth_distances, predicted_distances):
    squared_diff = torch.pow(truth_distances - predicted_distances, 2)
    mean_squared_diff = torch.mean(squared_diff)
    dRMSD = torch.sqrt(mean_squared_diff)
    return dRMSD.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune HiC-GAT model for another species.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing species data.')
    parser.add_argument('species', type=str, help='Species name (e.g., K562 or CH12-LX).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb, 100kb).')
    parser.add_argument('--pretrained_weights', type=str, default='path_to_gm12878_weights.pt', help='Path to pretrained GM12878 weights.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for fine-tuning.')  
    parser.add_argument('--threshold', type=float, default=1e-7, help='Loss threshold for fine-tuning.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    species = args.species  
    resolution = args.resolution  
    chromosome = args.chromosome 
    lr = args.lr  
    thresh = args.threshold
    pretrained_weights = args.pretrained_weights  
    epochs = args.epochs

    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, species, resolution, filename)
    
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{species}_{chromosome}_{resolution}'

    adj = np.loadtxt(filepath)
    print(f"Shape of raw data: {adj.shape}")

    if adj.shape[1] == 3:
        adj = utils.convert_to_matrix(adj)
    
    np.fill_diagonal(adj, 0)
    matrix_path = f'Data/Data_{species}_fine_tune/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'Data/Data_{species}_fine_tune/{name}_matrix_KR_normed.txt'
    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_array(adj)
    node2vec = Node2Vec(G, dimensions=512, walk_length=100, num_walks=20, p=2, q=0.5, workers=1, seed=42)
    model_node2vec = node2vec.fit(window=15, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(node)] for node in G.nodes()])

    data = utils.load_input(normed, embeddings)

    model = GATNetHeadsChanged4LayersLeakyReLU()
    
    model.load_state_dict(torch.load(pretrained_weights))
    model.train()

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    oldloss, lossdiff = 1, 1
    loss_history = []
    truth = utils.cont2dist(data.y, 1)  

    epoch = 0
    while lossdiff > thresh and epoch < epochs:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.float(), truth.float())
        lossdiff = abs(oldloss - loss)
        loss.backward()
        optimizer.step()
        oldloss = loss
        
        loss_history.append(loss.item())
        epoch += 1
        print(f'Epoch {epoch}, Loss: {loss.item()}', end='\r')

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss per Epoch', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {name} Fine-tuned on {species}')
    plt.legend()
    loss_plot_path = f'Outputs/{species}_fine_tune/{name}_loss_plot.png'
    plt.savefig(loss_plot_path)
    print(f'Saved fine-tuning loss plot to {loss_plot_path}')

    torch.save(model.state_dict(), f'Outputs/{species}_fine_tune/{name}_fine_tuned_weights.pt')
    print(f'Saved fine-tuned weights for {species} at {chromosome} to Outputs/{species}_fine_tune/{name}_fine_tuned_weights.pt')

    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    coords = model.get_model(data.x, data.edge_index)
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]
    dRMSD_value = calculate_dRMSD(dist_truth, dist_out)

    print(f'Fine-tuned dSCC: {SpRho}')
    print(f'Fine-tuned dRMSD: {dRMSD_value}')
