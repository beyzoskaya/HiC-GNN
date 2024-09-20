import numpy as np
from ge import LINE 
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import Net
from models import SmallerNet
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

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model with transfer learning.')
    
    parser.add_argument('dataset_folder', type=str, help='Input dataset folder path containing dataset folders.')
    parser.add_argument('subdataset', type=str, help='Sub-dataset name (e.g., CH12-LX).')
    parser.add_argument('resolution', type=str, help='Resolution subfolder name (e.g., 1mb).')
    parser.add_argument('chromosome', type=str, help='Chromosome name (e.g., chr12).')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model weights for transfer learning.')
    
    parser.add_argument('-c', '--conversions', type=str, default='[.1,.1,2]', help='List of conversion constants.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model weights for transfer learning.') 
    
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    subdataset = args.subdataset
    resolution = args.resolution
    chromosome = args.chromosome
    
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
    matrix_path = f'Data/{name}_matrix.txt'
    np.savetxt(matrix_path, adj, delimiter='\t')

    os.system(f'Rscript normalize.R {name}_matrix')
    normed_matrix_path = f'Data/{name}_matrix_KR_normed.txt'
    print(f'Created normalized matrix for {filepath} as {normed_matrix_path}')

    normed = np.loadtxt(normed_matrix_path)

    G = nx.from_numpy_matrix(adj)

    # Node2vec model for creating embeddings
    node2vec = Node2Vec(G, dimensions=512, walk_length=80, num_walks=10, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    embedding_path = f'Data/{name}_embeddings_node2vec.txt'
    np.savetxt(embedding_path, embeddings)
    print(f'Created embeddings corresponding to {filepath} as {embedding_path}')


    data = utils.load_input(normed, embeddings)

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    model = Net()

    # For pretrain (fine-tuning)
    if args.pretrained:
        print(f'Loading pretrained model from {args.pretrained}')
        model.load_state_dict(torch.load(args.pretrained))
    
    # Freeze all params
    for param in model.parameters():
        param.required_grad = False
    
    # Unfreeze last layer
    for param in model.fc.parameters():
        param.required_grad = True
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    criterion = MSELoss()
    oldloss, lossdiff = 1, 1
    truth = utils.cont2dist(data.y, conversion)

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

    repmod, repspear, repmse = coords, SpRho, loss

    print(f'Optimal dSCC: {repspear}')
    
    # Save fine-tuned model results
    with open(f'Outputs/{name}_node2vec_finetuned_log.txt', 'w') as f:
        f.writelines([f'Conversion factor: {conversion}\n', f'Optimal dSCC: {repspear}\n', f'Final MSE loss: {repmse}\n'])
    
    torch.save(model.state_dict(), f'Outputs/{name}_node2vec_weights_finetuned.pt')
    utils.WritePDB(repmod * 100, f'Outputs/{name}_node2vec_structure_finetuned.pdb')

    print(f'Saved fine-tuned model to Outputs/{name}_node2vec_weights_finetuned.pt')
    print(f'Saved optimal structure to Outputs/{name}_node2vec_structure_finetuned.pdb')
