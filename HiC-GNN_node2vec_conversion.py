import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import Net
from models import  GATNetMoreReduced, GATNet, GATNetConvLayerChanged, GATNetHeadsChanged, GATNetHeadsChangedLeakyReLU, GATNetHeadsChanged4Layers, GATNetHeadsChanged4LayerEmbedding256, GATNetHeadsChanged4LayerEmbedding256Dense, GATNetHeadsChanged4LayerEmbedding512Dense, TwoGATNetHeadsChanged4LayerEmbedding512Dense
import torch
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from scipy.stats import spearmanr
from random import uniform
import ast
import argparse
import json
import matplotlib.pyplot as plt
import random

# dSCC values more sensitive to global structure
# pairwise distances are more sensitive to local structure

# The difference in MSE is very small, but Node2Vec has a slight edge in minimizing the error, meaning that it might have learned the data a bit more precisely
# smaller errors but worse ranking

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_dRMSD(truth_distances, predicted_distances):
    # squared differences between true and predicted distances
    squared_diff = torch.pow(truth_distances - predicted_distances, 2)
    
    # mean of squared differences
    mean_squared_diff = torch.mean(squared_diff)
    dRMSD = torch.sqrt(mean_squared_diff)
    
    return dRMSD.item()

if __name__ == "__main__":
    base_data_dir = 'Data/Data_GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878'
    base_output_dir = 'Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GAT model.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., CH12-LX).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    
    #parser.add_argument('-c', '--conversions', type=str, default='[.1,.1,2]', help='List of conversion constants.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.0009, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-7, help='Loss threshold for training termination.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset  # Sub-dataset name, e.g., CH12-LX
    resolution = args.resolution  # Resolution subfolder, e.g., 1mb
    chromosome = args.chromosome  # Chromosome, e.g., chr12
    
    #conversions = args.conversions
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold
    #conversions = ast.literal_eval(conversions)
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
    print(f"Shape of raw data: {adj.shape}")

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
    normed = np.loadtxt(normed_matrix_path)
    print(f"Shape of normalized data: {normed.shape}")
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    # G = nx.from_numpy_matrix(adj)
    G = nx.from_numpy_array(adj) #changed to numpy_array for running in my desktop !!!!!!!

    # Node2vec model for creating embeddings
    #node2vec = Node2Vec(G, dimensions=512, walk_length=80, num_walks=10, workers=4,seed=42)
    #model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node2vec = Node2Vec(G, dimensions=512, walk_length=100, num_walks=20,p=2,q=0.5, workers=1, seed=42) # num workers and seed 42 together give deterministic results
    model= node2vec.fit(window=15, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    print(f"Shape for node2vec embeddings: {embeddings.shape}")
    embedding_path = f'Data/Data_GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_embeddings_node2vec_GAT.txt' 
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    data = utils.load_input(normed, embeddings)

    loss_history = []

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    # Train the HiC-GNN model using fixed conversion value
    print(f"Training model using conversion value {conversion}")
    model = GATNetHeadsChanged4LayerEmbedding512Dense() 

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    oldloss, lossdiff = 1,1
    truth = utils.cont2dist(data.y, conversion)

    
    # Training loop
    epoch = 0
    #max_epoch = 400
    while lossdiff > thresh:
    #for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.float(), truth.float())
        lossdiff = abs(oldloss - loss)
        loss.backward()
        optimizer.step()
        oldloss = loss
        
        loss_history.append(loss.item())
        epoch +=1 

        #print(f'Epoch {epoch}, Loss: {loss.item()}', end='\r')
        print(f'Loss: {loss}', end='\r')
    
    # Plotting loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss per Epoch', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {name}')
    plt.legend()
    loss_plot_path = f'Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_loss_plot.png'
    plt.savefig(loss_plot_path)  
    print(f'Saved loss plot as {loss_plot_path}')

    idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]]
    coords = model.get_model(data.x, data.edge_index)
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
    SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]
    repmod, repspear, repmse = coords, SpRho, loss

    dRMSD_value = calculate_dRMSD(dist_truth, dist_out)

    print(f'Optimal conversion factor: {conversion}')
    print(f'Optimal dSCC: {repspear}')
    print(f'dRMSD for {name}: {dRMSD_value}')

    with open(f'Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_node2vec_log_GAT.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {conversion}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n', f'dRMSD: {dRMSD_value}\n'])
    
    torch.save(model.state_dict(), f'Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_node2vec_GAT_weights.pt')
    utils.WritePDB(repmod * 100, f'Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_node2vec_GAT_structure.pdb')

    print(f'Saved trained model to Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_node2vec_GAT_weights.pt')
    print(f'Saved optimal structure to Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/{name}_node2vec_GAT_structure.pdb')


    # Calculate and save the variance of contact counts
    contact_counts = normed.flatten()  # Flatten the matrix to a single list of contact values
    raw_variance = np.var(contact_counts)  # Calculate the variance of contact values
    log_variance = np.log10(raw_variance)  # Calculate log variance
    
    # Save log variance in a separate file
    log_variance_data = {}
    log_variance_path = "Outputs/GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_node2vecParamsChanged_GM12878/log_variances_node2vec_GAT.json"
    
    if os.path.exists(log_variance_path):
        with open(log_variance_path, "r") as f:
            log_variance_data = json.load(f)
    
    log_variance_data[chromosome] = log_variance
    
    # Save the updated log variance data back to file
    with open(log_variance_path, "w") as f:
        json.dump(log_variance_data, f)
    
    print(f'Saved log variance for {chromosome}: {log_variance}')
