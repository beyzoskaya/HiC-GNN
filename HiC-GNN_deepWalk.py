import numpy as np
from karateclub import DeepWalk
import sys
import utils
import networkx as nx
import os
from models import Net, SmallerNet, GATSmallerNet
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

if __name__ == "__main__":
    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')
    if not os.path.exists('Data'):
        os.makedirs('Data')

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., CH12-LX).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    
    parser.add_argument('-c', '--conversions', type=str, default='[.1,.1,2]', help='List of conversion constants.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset  # Sub-dataset name, e.g., CH12-LX
    resolution = args.resolution  # Resolution subfolder, e.g., 1mb
    chromosome = args.chromosome  # Chromosome, e.g., chr12
    
    conversions = args.conversions
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold
    conversions = ast.literal_eval(conversions)

    if len(conversions) == 3:
        conversions = list(np.arange(conversions[0], conversions[2], conversions[1]))
    elif len(conversions) == 1:
        conversions = [conversions[0]]
    else:
        raise Exception('Invalid conversion input.')
        sys.exit(2) 

    # Generate file path for the input data
    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, subdataset, resolution, filename)
    
    # Check if the input file exists
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{subdataset}_{chromosome}_{resolution}'

    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0) 
    matrix_path = f'Data/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # Normalize the adjacency matrix using normalize.R script
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'Data/{name}_matrix_KR_normed.txt' 
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_matrix(adj)

    # Print the number of vertices (nodes) in the graph
    num_vertices = G.number_of_nodes()
    print(f'Number of vertices in the graph: {num_vertices}')

    # DeepWalk model for creating embeddings
    deepwalk = DeepWalk(dimensions=512, walk_length=80,workers=4)
    deepwalk.fit(G)  # Fit the DeepWalk model on the graph
    embeddings = deepwalk.get_embedding()

    embedding_path = f'Data/{name}_embeddings_SmallerNet_deepwalk.txt'  # Changed here for SmallerNet
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    # Visualize the DeepWalk embeddings as a heatmap (Matrix of vertex representations)
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(embeddings, cmap="coolwarm", cbar=True)
    #plt.title(f'DeepWalk Embeddings Matrix for {name}')
    #plt.xlabel('Embedding Dimensions')
    #plt.ylabel('Vertices')
    #plt.show()
    #plt.savefig('deepWalk.png')

    # Clustering with KMeans
    n_clusters = 5  # Number of clusters (communities)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    node_clusters = kmeans.fit_predict(embeddings)

    # Reduce the dimensionality of embeddings to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Visualize the clustering
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=node_clusters, cmap='viridis', s=100, alpha=0.8)
    plt.title('2D PCA of DeepWalk Embeddings with K-Means Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.show()
    plt.savefig('deepWalk_clustering_5_clusters.png')


    # Load input data with embeddings
    data = utils.load_input(normed, embeddings)

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    # Train the HiC-GNN model using different conversion factors
    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')
        model = SmallerNet()  # Replace with GATSmallerNet if needed

        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')

        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        oldloss, lossdiff = 1, 1
        truth = utils.cont2dist(data.y, 0.5)

        # Training loop
        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out.float(), truth.float())
            lossdiff = abs(oldloss - loss)
            loss.backward()
            optimizer.step()
            oldloss = loss
            print(f'Loss: {loss}', end='\r')

        # Calculate the Spearman correlation between true and predicted distances
        idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        coords = model.get_model(data.x, data.edge_index)
        dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
        SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

        # Save the results for the current model
        tempspear.append(SpRho)
        tempmodels.append(coords)
        tempmse.append(loss)
        model_list.append(model)

    # Select the best model based on Spearman correlation
    idx = tempspear.index(max(tempspear))
    repmod, repspear, repmse, repconv, repnet = tempmodels[idx], tempspear[idx], tempmse[idx], conversions[idx], model_list[idx]

    print(f'Optimal conversion factor: {repconv}')
    print(f'Optimal dSCC: {repspear}')

    # Save the best model and results
    with open(f'Outputs/{name}_SmallerNet_deepWalk_log.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {repconv}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n'])

    torch.save(repnet.state_dict(), f'Outputs/{name}_SmallerNet_deepWalk_weights.pt')  # Changed here for SmallerNet
    utils.WritePDB(repmod * 100, f'Outputs/{name}_SmallerNet_deepWalk_structure.pdb')  # Changed here for SmallerNet

    print(f'Saved trained model to Outputs/{name}_SmallerNet_deepWalk_weights.pt') # Changed here for SmallerNet
    print(f'Saved optimal structure to Outputs/{name}_SmallerNet_deepWalk_structure.pdb') # Changed here for SmallerNet
