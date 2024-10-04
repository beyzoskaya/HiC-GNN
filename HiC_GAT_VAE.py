import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import Net, VAE_GAT  
import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from scipy.stats import spearmanr
import ast
import argparse
import json
import matplotlib.pyplot as plt

def vae_loss(recon_x, x, mu, logvar, mse_weight=1.0, kl_weight=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # normalized by the number of samples
    
    # KL Divergence Loss with normalization
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]  # normalized by the batch size
    
    # Total loss: combining normalized MSE and KL Divergence with respective weights
    total_loss = mse_weight * recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

if __name__ == "__main__":
    base_data_dir = 'Data_VAE_GAT_GM12878'
    base_output_dir = 'Outputs/VAE_GAT_GM12878'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model with VAE+GAT.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., CH12-LX).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate for training the model.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset  # Sub-dataset name, e.g., CH12-LX
    resolution = args.resolution  # Resolution subfolder, e.g., 1mb
    chromosome = args.chromosome  # Chromosome, e.g., chr12

    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold
    conversion = 1

    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)
    
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{subdataset}_{chromosome}_{resolution}'

    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0) 
    matrix_path = f'{base_data_dir}/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'{base_data_dir}/{name}_matrix_KR_normed.txt' 
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_array(adj) #changed to numpy_array for running in my desktop !!!!!!!

    node2vec = Node2Vec(G, dimensions=512, walk_length=80, num_walks=10, workers=4)
    node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([node2vec_model.wv[str(node)] for node in G.nodes()])
    embedding_path = f'Data_VAE_GAT_GM12878/{name}_embeddings_VAE_GAT.txt' 
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    data = utils.load_input(normed, embeddings)

    loss_history = []
    mse_history = []
    kl_history = []

    input_dim = 512  
    hidden_dim = 512
    latent_dim = 64
    output_dim = 3
    heads = 2

    model = VAE_GAT(input_dim, hidden_dim, latent_dim, output_dim, heads=heads)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    optimizer = Adam(model.parameters(), lr=lr)
    oldloss, lossdiff = 1,1
    truth = utils.cont2dist(data.y, conversion)

    epoch = 0
    while lossdiff > thresh:
        model.train()
        optimizer.zero_grad()

        recon_x, mu, logvar, _ = model(data.x, data.edge_index)

        total_loss, mse_loss, kl_loss = vae_loss(recon_x, data.x, mu, logvar, kl_weight=0.01)
        lossdiff = abs(oldloss - total_loss)
        total_loss.backward()
        optimizer.step()
        oldloss = total_loss
        
        loss_history.append(total_loss.item())
        mse_history.append(mse_loss.item())  
        kl_history.append(kl_loss.item())  
        epoch += 1
        print(f'Epoch {epoch}, Total Loss: {total_loss.item()}, MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}', end='\r')

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss', color='blue')
    plt.plot(mse_history, label='MSE Loss', color='green')
    plt.plot(kl_history, label='KL Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {name}')
    plt.legend()
    loss_plot_path = f'Outputs/VAE_GAT_GM12878/{name}_loss_plot.png'
    plt.savefig(loss_plot_path)  
    print(f'Saved loss plot as {loss_plot_path}')

    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    coords = model.decoder(recon_x, data.edge_index)  
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

    repmod, repspear, repmse = coords, SpRho, total_loss

    print(f'Optimal conversion factor: {conversion}')
    print(f'Optimal dSCC: {repspear}')


    with open(f'Outputs/VAE_GAT_GM12878/{name}_VAE_GAT.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {conversion}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n'])
    
    torch.save(model.state_dict(), f'Outputs/VAE_GAT_GM12878/{name}_VAE_GAT_weights.pt')
    utils.WritePDB(repmod * 100, f'Outputs/VAE_GAT_GM12878/{name}_VAE_GAT_structure.pdb')

    print(f'Saved trained model to Outputs/VAE_GAT_GM12878/{name}_VAE_GAT_weights.pt')
    print(f'Saved optimal structure to Outputs/VAE_GAT_GM12878/{name}_VAE_GAT_structure.pdb')

    contact_counts = normed.flatten()
    raw_variance = np.var(contact_counts)
    log_variance = np.log10(raw_variance)
    
    log_variance_data = {}
    log_variance_path = "Outputs/VAE_GAT_GM12878/log_variances_VAE_GAT.json"
    
    if os.path.exists(log_variance_path):
        with open(log_variance_path, "r") as f:
            log_variance_data = json.load(f)
    
    log_variance_data[chromosome] = log_variance
    
    with open(log_variance_path, "w") as f:
        json.dump(log_variance_data, f)
    
    print(f'Saved log variance for {chromosome}: {log_variance}')
