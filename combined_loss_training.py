import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt
import random
from models import GATNetHeadsChanged3LayersLeakyReLUv2
import json

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_dRMSD(truth_distances, predicted_distances):
    squared_diff = torch.pow(truth_distances - predicted_distances, 2) # ((y_true - y_pred)^2)
    mean_squared_diff = torch.mean(squared_diff) # ((1/n)(y_true - y_pred)^2) --> normalized
    dRMSD = torch.sqrt(mean_squared_diff) # (((1/n)(y_true - y_pred)^2))^(-1/2)
    return dRMSD.item() 

if __name__ == "__main__":
    base_data_dir = 'Data/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000'
    base_output_dir = 'Outputs/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0.0005_dropout_0.3_threshold_1e-8_p_4_q_2.5_GM12878_combined_loss_epoch_1000'

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
    
    parser = argparse.ArgumentParser(description='Train a HiC-GAT model with combined MSE + dSCC loss.')
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., CH12-LX).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=1000, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.0005, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    parser.add_argument('-alpha', '--alpha', type=float, default=1, help='Weight for dSCC loss component.')

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset
    resolution = args.resolution
    chromosome = args.chromosome
    batch_size = args.batchsize
    epochs = args.epochs
    conversion = 1
    lr = args.learningrate
    thresh = args.threshold
    alpha = args.alpha

    # chr1_1mb_RAWobserved.txt
    filename = f'{chromosome}_{resolution}_RAWobserved.txt'
    # Hi-C_dataset/GM12878/1mb/chr1_1mb_RAWobserved.txt
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)

    if not os.path.exists(filepath):
        print(f'File {filepath} not found.')
        sys.exit(1)
    
    name = f'{subdataset}_{chromosome}_{resolution}'
    adj = np.loadtxt(filepath)

    print(f"Shape of adj shape[1]: {adj.shape[1]}")
    if adj.shape[1] == 3:
        adj = utils.convert_to_matrix(adj)
    
    np.fill_diagonal(adj, 0)
    matrix_path = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # KR normalization
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt'
    normed = np.loadtxt(normed_matrix_path)

    # Convert data to graph
    G = nx.from_numpy_array(adj)

    # create node2vec embeddings
    node2vec = Node2Vec(G, dimensions=512, walk_length=100, num_walks=20,p=4,q=2.5, workers=1, seed=42) 
    node2vec_model= node2vec.fit(window=15, min_count=1, batch_words=4)
    embeddings = np.array([node2vec_model.wv[str(node)] for node in G.nodes()])

    data = utils.load_input(normed, embeddings)

    model = GATNetHeadsChanged3LayersLeakyReLUv2()
    criterion_mse = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    loss_history = [] # combined loss to plot
    dSCC_history = [] # dSCC values
    mse_history = [] # mse values
    dRMSD_history = [] # dRMSD values

    oldloss = float('inf') 
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # data --> 2D matrix of shape (num_nodes, num_features)
        # data.x --> node embeddings created with node2vec 
        # If the data has 1000 nodes then the shape of data.x --> (1000,512)

        #print(f"data.x : {data.x}")
        #print(f"data.edge_index: {data.edge_index}")
        out = model(data.x, data.edge_index)
        truth = utils.cont2dist(data.y, conversion)

        mse_loss = criterion_mse(out.float(), truth.float())

        """

        [[0, 1, 2],  --> idx gives [(0,1), (0,2), (1,2)]
        [1, 0, 3], 
        [2, 3, 0]]
        
        """

        
        idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        coords = model.get_model(data.x, data.edge_index)
        dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]] # calculates pairwise Euclidean distances between predicted 3D coords

        SpRho, _ = spearmanr(dist_truth.detach().numpy(), dist_out.detach().numpy())
        if np.isnan(SpRho):
            print(f"dSCC value is None")
            SpRho = 0

        dRMSD_value = calculate_dRMSD(dist_truth, dist_out)
        dSCC_loss = (1 - SpRho) # minimizing 1 - dSCC
        combined_loss = mse_loss + alpha * dSCC_loss 
        lossdiff = abs(oldloss - combined_loss.item())
        combined_loss.backward()
        optimizer.step()
        oldloss = combined_loss

        loss_history.append(combined_loss.item())
        mse_history.append(mse_loss.item())
        dSCC_history.append(SpRho)
        dRMSD_history.append(dRMSD_value)
        print(f'Epoch {epoch + 1}/{epochs}, Combined Loss: {combined_loss.item()}')


    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Combined Loss (MSE + dSCC)', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {name}')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name}_combined_loss_plot.png')
    print(f'Saved combined loss plot as {base_output_dir}/{name}_combined_loss_plot.png')

    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, label='MSE Loss', color='red')
    plt.plot(dSCC_history, label='dSCC (Spearman Correlation)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.title(f'MSE and dSCC Curves for {name}')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name}_mse_dSCC_plot.png')
    print(f'Saved MSE and dSCC plot as {base_output_dir}/{name}_mse_dSCC_plot.png')

    with open(f'{base_output_dir}/{name}_results.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {conversion}\n', f'Optimal dSCC: {SpRho}\n', f'Final MSE loss: {mse_loss}\n', f'dRMSD: {dRMSD_value}\n', f'Final combined loss: {combined_loss}\n' ])
        

    torch.save(model.state_dict(), f'{base_output_dir}/{name}_weights.pt')
    utils.WritePDB(coords * 100, f'{base_output_dir}/{name}_structure.pdb')
    print(f'Saved trained model to {base_output_dir}/{name}_weights.pt')
    print(f'Saved optimal structure to {base_output_dir}/{name}_structure.pdb')

    contact_counts = normed.flatten()
    raw_variance = np.var(contact_counts)
    log_variance = np.log10(raw_variance)

    log_variance_data = {}
    log_variance_path = f"{base_output_dir}/log_variances.json"
    if os.path.exists(log_variance_path):
        with open(log_variance_path, "r") as f:
            log_variance_data = json.load(f)
    
    log_variance_data[chromosome] = log_variance
    with open(log_variance_path, "w") as f:
        json.dump(log_variance_data, f)

    print(f'Saved log variance for {chromosome}: {log_variance}')

    