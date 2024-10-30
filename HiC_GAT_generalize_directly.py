import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import GATNetHeadsChanged3LayersLeakyReLUv2, GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim256, GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim128,GATNetHeadsChanged4LayersLeakyReLU, GATNetHeadsChanged4Layers, GATNetHeadsChanged4LayersLeakyReLUHeads4
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr,pearsonr
import argparse
import matplotlib.pyplot as plt

"""
grid search options 
p_values = [0.5, 1, 1.5]
q_values = [0.5, 1, 2]
num_walks_values = [10, 25, 50]
epochs_values = [100, 200, 500]
learning_rates = [0.0005, 0.001, 0.002]

option 1:
p=1.5, q=0.5, num_walks=30, walk length: 80, epochs=500, lr=0.001 

option 2:
p=2, q=0.25, num_walks=25, walk length: 100, epochs=600, lr=0.0008

option 3:
p=1.75, q=0.4, num_walks=35, walk length: 90, epochs=550, lr=0.0012

option 4:
p=2, q=0.5, num_walks=40, walk length: 120, epochs=700, lr=0.0005

option 5:
p=1.6, q=0.3, num_walks=50, walk length: 90, epochs=450, lr=0.001

"""

def calculate_dRMSD(truth_distances, predicted_distances):
    squared_diff = torch.pow(truth_distances - predicted_distances, 2)  # (y_true - y_pred)^2
    mean_squared_diff = torch.mean(squared_diff)  # Mean of squared diff
    dRMSD = torch.sqrt(mean_squared_diff)  # sqrt(mean((y_true - y_pred)^2))
    return dRMSD.item()

if __name__ == "__main__":
    base_data_dir = 'Data/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0001_epoch_1000_threshold_1e-8_p_1.2_q_0.8_embedding_dim_512_GM12878_generalization_aligned'
    base_output_dir = 'Outputs/GATNetHeadsChanged3LayersLeakyReLUv2_lr_0.0001_epoch_1000_threshold_1e-8_p_1.2_q_0.8_embedding_dim_512_GM12878_generalization_aligned'

    if not(os.path.exists(base_data_dir)):
        os.makedirs(base_data_dir)

    if not(os.path.exists(base_output_dir)):
        os.makedirs(base_output_dir)

    parser = argparse.ArgumentParser(description='Generalize a trained model to new data using combined loss.')
    parser.add_argument('list_trained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_trained.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_untrained.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=1000, help='Number of epochs used for model training.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.0001, help='Learning rate for training GCNN.')
    parser.add_argument('-thresh', '--loss_diff_threshold', type=float, default=1e-8, help='Loss difference threshold for early stopping.')
    parser.add_argument('-print_interval', type=int, default=10, help='Interval for printing MSE and dSCC values.')
    #parser.add_argument('-alpha', '--alpha', type=float, default=0.1, help='Weight for dSCC loss component in combined loss.')

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    #alpha = args.alpha
    loss_diff_threshold = args.loss_diff_threshold
    print_interval = args.print_interval
    conversion = 1

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    list_trained = np.loadtxt(filepath_trained)
    list_untrained = np.loadtxt(filepath_untrained)
    print(f"Loaded {len(list_trained)} entries for trained and {len(list_untrained)} for untrained data.")

    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_matrix.txt')):
        print(f'Failed to find matrix form of {filepath_trained} from {base_data_dir}/{name_trained}_matrix.txt.')
        adj_trained = utils.convert_to_matrix(list_trained)
        np.fill_diagonal(adj_trained, 0)
        np.savetxt(f'{base_data_dir}/{name_trained}_matrix.txt', adj_trained, delimiter='\t')
        print(f'Created matrix form of {filepath_trained} as {base_data_dir}/{name_trained}_matrix.txt.')
    matrix_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_matrix.txt')
    print(f"Matrix trained shape: {matrix_trained.shape}")

    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_matrix.txt')):
        print(f'Failed to find matrix form of {filepath_untrained} from {base_data_dir}/{name_untrained}_matrix.txt.')
        adj_untrained = utils.convert_to_matrix(list_untrained)
        np.fill_diagonal(adj_untrained, 0)
        np.savetxt(f'{base_data_dir}/{name_untrained}_matrix.txt', adj_untrained, delimiter='\t')
        print(f'Created matrix form of {filepath_untrained} as {base_data_dir}/{name_untrained}_matrix.txt.')
    matrix_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_matrix.txt')
    print(f"Matrix untrained shape: {matrix_untrained.shape}")

    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_matrix_KR_normed.txt')):
        print(f'Failed to find normalized matrix form of {filepath_trained} from {base_data_dir}/{name_trained}_matrix_KR_normed.txt')
        os.system(f'Rscript normalize.R {name_trained}_matrix')
        print(f'Created normalized matrix form of {filepath_trained} as {base_data_dir}/{name_trained}_matrix_KR_normed.txt')
    normed_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_matrix_KR_normed.txt')
    print(f"Normalized trained matrix stats: mean={np.mean(normed_trained)}, std={np.std(normed_trained)}, min={np.min(normed_trained)}, max={np.max(normed_trained)}")

    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_matrix_KR_normed.txt')):
        print(f'Failed to find normalized matrix form of {filepath_untrained} from {base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
        os.system(f'Rscript normalize.R {name_untrained}_matrix')
        print(f'Created normalized matrix form of {filepath_untrained} as {base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
    normed_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
    print(f"Normalized untrained matrix stats: mean={np.mean(normed_untrained)}, std={np.std(normed_untrained)}, min={np.min(normed_untrained)}, max={np.max(normed_untrained)}")


    # Create node2vec embeddings for the trained data
    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_embeddings.txt')):
        print(f'Failed to find embeddings corresponding to {filepath_trained} from {base_data_dir}/{name_trained}_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_trained)

        node2vec_trained = Node2Vec(G, dimensions=512, walk_length=100, num_walks=25, p=1.2, q=0.8, workers=1, seed=42)
        embeddings_trained = node2vec_trained.fit(window=15, min_count=1, batch_words=4)
        embeddings_trained = np.array([embeddings_trained.wv[str(node)] for node in G.nodes()])
        np.savetxt(f'{base_data_dir}/{name_trained}_embeddings.txt', embeddings_trained)
        print(f'Created embeddings corresponding to {filepath_trained} as {base_data_dir}/{name_trained}_embeddings.txt.')
    embeddings_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_embeddings.txt')
    print(f"Trained embeddings stats: shape={embeddings_trained.shape}, mean={np.mean(embeddings_trained)}, std={np.std(embeddings_trained)}, min={np.min(embeddings_trained)}, max={np.max(embeddings_trained)}")

    # Create node2vec embeddings for the untrained data
    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_embeddings.txt')):
        print(f'Failed to find embeddings corresponding to {filepath_untrained} from {base_data_dir}/{name_untrained}_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_untrained)

        node2vec_untrained = Node2Vec(G, dimensions=512, walk_length=100, num_walks=25, p=1.2, q=0.8, workers=1, seed=42)
        embeddings_untrained = node2vec_untrained.fit(window=15, min_count=1, batch_words=4)
        embeddings_untrained = np.array([embeddings_untrained.wv[str(node)] for node in G.nodes()])
        np.savetxt(f'{base_data_dir}/{name_untrained}_embeddings.txt', embeddings_untrained)
        print(f'Created embeddings corresponding to {filepath_untrained} as {base_data_dir}/{name_untrained}_embeddings.txt.')
    embeddings_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_embeddings.txt')
    print(f"Untrained embeddings stats: shape={embeddings_untrained.shape}, mean={np.mean(embeddings_untrained)}, std={np.std(embeddings_untrained)}, min={np.min(embeddings_untrained)}, max={np.max(embeddings_untrained)}")

    data_trained = utils.load_input(normed_trained, embeddings_trained)
    data_untrained = utils.load_input(normed_untrained, embeddings_untrained)
    print(f"Data shapes: Trained (x)={data_trained.x.shape}, Trained (y)={data_trained.y.shape}")

    # Train the model using a fixed number of epochs and combined loss
    if not(os.path.isfile(f'{base_output_dir}/{name_trained}_weights.pt')):
        print(f'Failed to find model weights corresponding to {filepath_trained} from {base_output_dir}/{name_trained}_weights.pt')
        model = GATNetHeadsChanged3LayersLeakyReLUv2()
        criterion_mse = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        loss_history = []  
        dSCC_history = []
        mse_history = []  
        iteration = 0

        model.eval()  
        out_initial = model(data_trained.x.float(), data_trained.edge_index)
        truth = utils.cont2dist(data_trained.y, conversion)

        idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        dist_truth_np = dist_truth.detach().numpy()  

        coords_initial = model.get_model(data_trained.x.float(), data_trained.edge_index)
        dist_out_initial = torch.cdist(coords_initial, coords_initial)[idx[0, :], idx[1, :]]
        dist_out_np = dist_out_initial.detach().numpy()  

        initial_dSCC, _ = spearmanr(dist_truth_np, dist_out_np)
        alpha = 1 - initial_dSCC
        print(f"Initial dSCC (Spearman Correlation): {initial_dSCC}")
        print(f"Initial adaptive alpha: {alpha}")

        loss_diff = 1
        old_loss = 1
        #for epoch in range(epochs):
        while loss_diff > loss_diff_threshold:
            model.train()
            optimizer.zero_grad()

            out = model(data_trained.x.float(), data_trained.edge_index)
            #print(f"Output shape: {out.shape}, Expected shape: {data_trained.y.shape}")
            truth = utils.cont2dist(data_trained.y, conversion)

            mse_loss = criterion_mse(out.float(), truth.float())

            idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1)
            dist_truth = truth[idx[0, :], idx[1, :]]
            dist_truth_np = dist_truth.detach().numpy()
            coords = model.get_model(data_trained.x.float(), data_trained.edge_index)
            dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]  # Pairwise Euclidean distances


            # min-max scaling 
            #dist_out_min, dist_out_max = dist_out.min(), dist_out.max()
            #dist_out = (dist_out - dist_out_min) / (dist_out_max - dist_out_min)
            #dist_out = dist_out * (dist_truth.max() - dist_truth.min()) + dist_truth.min()

            # mean std dev normalization
            #dist_out = (dist_out - dist_out.mean()) / dist_out.std()
            #dist_out = dist_out * dist_truth.std() + dist_truth.mean()

            dist_out_np = dist_out.detach().numpy() 


            SpRho, _ = spearmanr(dist_truth.detach().numpy(), dist_out.detach().numpy())
            PearsonR, _ = pearsonr(dist_truth.detach().numpy(), dist_out_np)
            #SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]
            if np.isnan(SpRho):
                print(f"Warning: dSCC is NaN")
                SpRho = 0 

            dSCC_loss = (1 - SpRho)  # Minimize 1 - dSCC
            combined_loss = mse_loss + alpha * dSCC_loss  # Combined loss
            loss_diff = abs(old_loss - combined_loss.item())
            old_loss = combined_loss.item()

            alpha = 1 - SpRho
            #mse_loss.backward()
            combined_loss.backward()
            optimizer.step()

            #if epoch % 10 == 0:
            #    print(f"Epoch [{epoch}/{epochs}], MSE Loss: {mse_loss.item()}, dSCC: {SpRho}")
            if iteration % print_interval == 0:
                print(f"Iteration [{iteration}], MSE Loss: {mse_loss.item()}, dSCC: {SpRho}, Loss Diff: {loss_diff}")
            
            #loss_history.append(combined_loss.item())
            mse_history.append(mse_loss.item())
            dSCC_history.append(SpRho)

            #print(f'Epoch [{epoch + 1}/{epochs}], Combined Loss: {combined_loss.item()}', end='\r')
            final_mse_loss = mse_loss.item()
            final_combined_loss = combined_loss.item()
            iteration += 1


        print(f'\nOptimal dSCC after training: {SpRho}')
        print(f"Pearson Correlation: {PearsonR}")

        mse_mean, mse_max = np.mean(mse_history), np.max(mse_history)
        dSCC_mean, dSCC_max = np.mean(dSCC_history), np.max(dSCC_history)

        print(f"\nMSE Loss - Mean: {mse_mean}, Max: {mse_max}")
        print(f"dSCC Loss - Mean: {dSCC_mean}, Max: {dSCC_max}")

        torch.save(model.state_dict(), f'{base_output_dir}/{name_trained}_weights.pt')
        utils.WritePDB(coords * 100, f'{base_output_dir}/{name_trained}_structure.pdb')
        print(f'Saved trained model to {base_output_dir}/{name_trained}_weights.pt')
        print(f'Saved optimal structure to {base_output_dir}/{name_trained}_structure.pdb')

        with open(f'{base_output_dir}/{name_trained}_results.txt', 'w') as f:
            f.writelines([f'Optimal dSCC: {SpRho}\n', f'Final MSE loss: {final_mse_loss}\n', f'Final Combined loss: {final_combined_loss}\n' ])

        # Plot combined loss, dSCC, and MSE over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(mse_history, label='Combined Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Combined Loss Curve for {name_trained}')
        plt.legend()
        plt.savefig(f'{base_output_dir}/{name_trained}_combined_loss_plot.png')

        plt.figure(figsize=(10, 6))
        plt.plot(mse_history, label='MSE Loss', color='red')
        plt.plot(dSCC_history, label='dSCC (Spearman Correlation)', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.title(f'MSE and dSCC Curves for {name_trained}')
        plt.legend()
        plt.savefig(f'{base_output_dir}/{name_trained}_mse_dSCC_plot.png')

    # Generalize to untrained data
    model = GATNetHeadsChanged3LayersLeakyReLUv2()
    model.load_state_dict(torch.load(f'{base_output_dir}/{name_trained}_weights.pt'))
    model.eval()

    fitembed = utils.domain_alignment(list_trained, list_untrained, embeddings_trained, embeddings_untrained)
    data_untrained_fit = utils.load_input(normed_untrained, fitembed)

    truth = utils.cont2dist(data_untrained_fit.y, conversion).float()
    idx = torch.triu_indices(data_untrained_fit.y.shape[0], data_untrained_fit.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]].detach().numpy()
    coords = model.get_model(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
    out = torch.cdist(coords, coords)
    dist_out = out[idx[0, :], idx[1, :]].detach().numpy()

    #dist_out_min, dist_out_max = dist_out.min(), dist_out.max()
    #dist_out = (dist_out - dist_out_min) / (dist_out_max - dist_out_min)
    #dist_out = dist_out * (dist_truth.max() - dist_truth.min()) + dist_truth.min()

    #dist_out = (dist_out - dist_out.mean()) / dist_out.std()
    #dist_out = dist_out * dist_truth.std() + dist_truth.mean()

    SpRho = spearmanr(dist_truth, dist_out)[0]
    print(f'Optimal dSCC for generalized data: {SpRho}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dist_truth)), dist_truth, label='True Distances', color='blue')
    plt.plot(range(len(dist_out)), dist_out, label='Predicted Distances', color='orange')
    plt.xlabel('Pairwise Distances Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_distance_comparison_plot.png')

    # Scatter Plot (True vs Predicted Distances)
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_truth, dist_out, alpha=0.5)
    plt.plot([dist_truth.min(), dist_truth.max()], [dist_truth.min(), dist_truth.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel('True Pairwise Distance')
    plt.ylabel('Predicted Pairwise Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_scatter_distance_comparison_plot.png')

    # Histogram of True and Predicted Distances
    plt.figure(figsize=(10, 6))
    plt.hist(dist_truth, bins=50, alpha=0.5, label='True Distances', color='blue')
    plt.hist(dist_out, bins=50, alpha=0.5, label='Predicted Distances', color='orange')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_histogram_distance_comparison_plot.png')

    # Save the generalized results
    utils.WritePDB(coords * 100, f'{base_output_dir}/{name_untrained}_generalized_structure.pdb')

    with open(f'{base_output_dir}/{name_untrained}_generalized_log.txt', 'w') as f:
        f.writelines([f'Optimal dSCC: {SpRho}\n'])
