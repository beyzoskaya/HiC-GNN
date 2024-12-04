import numpy as np
from ge import LINE
import utils
import networkx as nx
import os
from models import Net
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if not os.path.exists('NCol_train_test/Outputs_GNN'):
        os.makedirs('NCol_train_test/Outputs_GNN')

    if not os.path.exists('Data_NCol_same_train_test/Data_NCol'):
        os.makedirs('Data_NCol_same_train_test/Data_NCol')

    parser = argparse.ArgumentParser(description='Generalize a trained model to new data using NCol normalized input.')
    parser.add_argument('list_trained', type=str, help='File path for list format of normalized NCol Hi-C data for training.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of normalized NCol Hi-C data for testing.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs for embeddings generation.')
    parser.add_argument('-lr', '--learningrate', type=float, default=.001, help='Learning rate for training GNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

    conversion = 1

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    normed_trained = np.loadtxt(f'Hi-C_dataset/Hindlll_Ncol/Ncol/{name_trained}.txt', delimiter='\t')
    normed_untrained = np.loadtxt(f'Hi-C_dataset/Hindlll_Ncol/Ncol/{name_untrained}.txt', delimiter='\t')

    if not os.path.isfile(f'Data_NCol_same_train_test/Data_NCol/{name_trained}_embeddings.txt'):
        print(f'Generating embeddings for {name_trained}...')
        G = nx.from_numpy_matrix(normed_trained)
        embed_trained = LINE(G, embedding_size=512, order='second')
        embed_trained.train(batch_size=batch_size, epochs=epochs, verbose=1)
        embeddings_trained = embed_trained.get_embeddings()
        embeddings_trained = np.asarray(list(embeddings_trained.values()))
        np.savetxt(f'Data_NCol_same_train_test/Data_NCol/{name_trained}_embeddings.txt', embeddings_trained)
    embeddings_trained = np.loadtxt(f'Data_NCol_same_train_test/Data_NCol/{name_trained}_embeddings.txt')

    if not os.path.isfile(f'Data_NCol_same_train_test/Data_NCol/{name_untrained}_embeddings.txt'):
        print(f'Generating embeddings for {name_untrained}...')
        G = nx.from_numpy_matrix(normed_untrained)
        embed_untrained = LINE(G, embedding_size=512, order='second')
        embed_untrained.train(batch_size=batch_size, epochs=epochs, verbose=1)
        embeddings_untrained = embed_untrained.get_embeddings()
        embeddings_untrained = np.asarray(list(embeddings_untrained.values()))
        np.savetxt(f'Data_NCol_same_train_test/Data_NCol/{name_untrained}_embeddings.txt', embeddings_untrained)
    embeddings_untrained = np.loadtxt(f'Data_NCol_same_train_test/Data_NCol/{name_untrained}_embeddings.txt')

    data_trained = utils.load_input(normed_trained, embeddings_trained)
    data_untrained = utils.load_input(normed_untrained, embeddings_untrained)

    if not os.path.isfile(f'NCol_train_test/Outputs_GNN/{name_trained}_weights.pt'):
        print(f'Training model for {name_trained}...')
        model = Net()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        oldloss = 1
        lossdiff = 1
        truth = utils.cont2dist(data_trained.y, conversion)

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()
            out = model(data_trained.x.float(), data_trained.edge_index)
            loss = criterion(out.float(), truth.float())
            lossdiff = abs(oldloss - loss.item())
            loss.backward()
            optimizer.step()
            oldloss = loss.item()
            print(f'Loss: {loss.item():.6f}', end='\r')

        idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        coords = model.get_model(data_trained.x.float(), data_trained.edge_index)
        dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]
        SpRho = spearmanr(dist_truth.cpu().numpy(), dist_out.detach().cpu().numpy())[0]

        print(f'\nOptimal dSCC for {name_trained}: {SpRho:.6f}')

        torch.save(model.state_dict(), f'NCol_train_test/Outputs_GNN/{name_trained}_weights.pt')
        utils.WritePDB(coords.detach().cpu().numpy() * 100, f'NCol_train_test/Outputs_GNN/{name_trained}_structure.pdb')

    print(f'Evaluating on {name_untrained}...')
    model = Net()
    model.load_state_dict(torch.load(f'NCol_train_test/Outputs_GNN/{name_trained}_weights.pt'))
    model.eval()

    fitembed = utils.domain_alignment(data_trained.y, data_untrained.y, embeddings_trained, embeddings_untrained)
    data_untrained_fit = utils.load_input(normed_untrained, fitembed)

    truth = utils.cont2dist(data_untrained_fit.y, conversion).float()
    idx = torch.triu_indices(data_untrained_fit.y.shape[0], data_untrained_fit.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]].detach().cpu().numpy()
    coords = model.get_model(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
    dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]].detach().cpu().numpy()

    SpRho = spearmanr(dist_truth, dist_out)[0]
    print(f'Optimal dSCC for generalized data {name_untrained}: {SpRho:.6f}')

    utils.WritePDB(coords.detach().cpu().numpy() * 100, f'NCol_train_test/Outputs_GNN/{name_untrained}_generalized_structure.pdb')

    with open(f'NCol_train_test/Outputs_GNN/{name_untrained}_generalized_log.txt', 'w') as f:
        f.write(f'Optimal dSCC: {SpRho:.6f}\n')
