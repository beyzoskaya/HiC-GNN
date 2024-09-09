import numpy as np
from ge import LINE
import sys
import utils
import networkx as nx
import os
from models import Net
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast
import argparse

if __name__ == "__main__":
    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')
    if not os.path.exists('Data'):
        os.makedirs('Data')

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model.')
    
    # New dataset naming RAWobserved
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing RAWobserved files.')
    
    # There are different chr and mb/kb inside the CH12-LX
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr19).')
    parser.add_argument('resolution', type=str, help='Resolution (e.g., 1mb or 100kb).')
    
    parser.add_argument('-c', '--conversions', type=str, default='[.1,.1,2]', help='List of conversion constants.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    chromosome = args.chromosome  # Chromosome, chr19
    resolution = args.resolution  # Resolution, 1mb or 100kb
    
    conversions = ast.literal_eval(args.conversions)
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

    # Take the filename with arguments
    filename = f'{chromosome}_{resolution}.RAWobserved.txt'
    filepath = os.path.join(dataset_folder, filename)
    
    # Control the file
    if not os.path.exists(filepath):
        print(f'File {filepath} not found. Please check the folder and file structure.')
        sys.exit(1)

    name = f'{chromosome}_{resolution}'

    # RAWobserved data in CH12-LX same with coordinate list in GM
    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)
    matrix_path = f'Data/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    # Normalize the matrix 
    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'Data/{name}_matrix_KR_normed.txt'
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_matrix(adj)

    embed = LINE(G, embedding_size=512, order='second')
    embed.train(batch_size=batch_size, epochs=epochs, verbose=1)
    embeddings = np.asarray(list(embed.get_embeddings().values()))

    embedding_path = f'Data/{name}_embeddings.txt'
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')

    data = utils.load_input(normed, embeddings)

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')
        model = Net()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        oldloss, lossdiff = 1, 1
        truth = utils.cont2dist(data.y, 0.5)

        # Training loop until loss stabilizes
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

        idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        coords = model.get_model(data.x, data.edge_index)
        dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
        SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

        tempspear.append(SpRho)
        tempmodels.append(coords)
        tempmse.append(loss)
        model_list.append(model)

    # Select the best model based on Spearman correlation
    idx = tempspear.index(max(tempspear))
    repmod, repspear, repmse, repconv, repnet = tempmodels[idx], tempspear[idx], tempmse[idx], conversions[idx], model_list[idx]

    print(f'Optimal conversion factor: {repconv}')
    print(f'Optimal dSCC: {repspear}')

    with open(f'Outputs/{name}_log.txt', 'w') as f:
        f.writelines([f'Optimal conversion factor: {repconv}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n'])

    torch.save(repnet.state_dict(), f'Outputs/{name}_weights.pt')
    utils.WritePDB(repmod * 100, f'Outputs/{name}_structure.pdb')

    print(f'Saved trained model to Outputs/{name}_weights.pt')
    print(f'Saved optimal structure to Outputs/{name}_structure.pdb')
