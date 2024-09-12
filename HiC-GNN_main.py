import numpy as np
from ge import LINE  # Import LINE for node embeddings
import sys
import utils
import networkx as nx
import os
from models import Net
from models import SmallerNet
from models import GATSmallerNet
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast
import argparse

# LINE has a slightly higher dSCC of 0.9156, meaning that its predictions are more correlated with the true distances
# better ranking but larger errors
if __name__ == "__main__":
    # Create necessary directories if they do not exist
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
    
    conversions = ast.literal_eval(args.conversions)
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

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
    matrix_path = f'Data/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # Normalize the adjacency matrix using normalize.R script
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'Data/{name}_matrix_KR_normed.txt'
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_matrix(adj)

    # LINE model for creating embeddings
    line = LINE(G, embedding_size=512, order='second')  # Adjust order if needed
    line.train(batch_size=batch_size, epochs=epochs, verbose=1)
    embeddings = np.array([line.get_embeddings()[node] for node in G.nodes()])
    embedding_path = f'Data/{name}_embeddings_GATSmallerNet_LINE.txt'  # Changed here for SmallerNet
    #embedding_path = f'Data/{name}_embeddings_LINE.txt'
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    data = utils.load_input(normed, embeddings)

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    # Train the HiC-GNN model using different conversion factors
    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')
        #model = Net()
        #model = SmallerNet()
        model = GATSmallerNet()

        # Print the total number of parameters in the model
        #total_params = sum(p.numel() for p in model.parameters())
        #print(f'Total number of parameters: {total_params}')

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
    with open(f'Outputs/{name}_GATSmallerNet_LINE_log.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {repconv}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n'])

    #torch.save(repnet.state_dict(), f'Outputs/{name}_LINE_weights.pt') 
    #utils.WritePDB(repmod * 100, f'Outputs/{name}_LINE_structure.pdb')
    #print(f'Saved trained model to Outputs/{name}__LINE_weights.pt')
    #print(f'Saved optimal structure to Outputs/{name}_LINE_structure.pdb') 

    torch.save(repnet.state_dict(), f'Outputs/{name}_GATSmallerNet_LINE_weights.pt')  # Changed here for SmallerNet
    utils.WritePDB(repmod * 100, f'Outputs/{name}_GATSmallerNet_LINE_structure.pdb')  # Changed here for SmallerNet

    print(f'Saved trained model to Outputs/{name}_GATSmallerNet_LINE_weights.pt') # Changed here for SmallerNet
    print(f'Saved optimal structure to Outputs/{name}_GATSmallerNet_LINE_structure.pdb') # Changed here for SmallerNet
