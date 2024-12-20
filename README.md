# GAT-HiC Efficient Reconstruction of 3D Chromosome Structure Via Graph Attention Neural Network
------------------------------------------------------------------------------------------------------------------------------------
## Overview
GATHiC is a method developed to predict three-dimensional chromosome structure from Hi-C interaction data. GATHiC can generalize to unseen Hi-C datasets, enabling prediction across various cell populations, restriction enzymes, and Hi-C resolutions. This method combines the unsupervised vertex embedding technique Node2vec with an attention-based graph neural network to predict the 3D coordinates of genomic loci.

## HiC Data
------------------------------------------------------------------------------------------------------------------------------------
The Hi-C data used in this project consists of contact maps for each species, where each contact map represents the interaction frequencies between genomic loci across chromosomes. These interaction matrices are crucial for predicting the three-dimensional (3D) structure of chromosomes, providing a foundation for the GATHiC method to infer the 3D coordinates from these interaction patterns.

## Dependencies
The required libraries for running this project are listed in the `requirements.txt` file. To install all the necessary libraries, follow these steps:

### Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/seferlab/GAT-HiC.git
```
### Set Up the Environment
Navigate to the project directory and install the dependencies from the `requirements.txt` file:
```bash
cd GAT-HiC
pip install -r requirements.txt
```

----------------------------------------------------------------------------------------------------------------------------------	
## Key Scripts 

### 1. `HiC-GAT_generalize_directly.py`

#### Purpose:
This file focuses on the process of training a Graph Attention Network (GAT) model on Hi-C data embeddings, followed by generalizing the trained model to untrained Hi-C data.

#### Inputs:
- **`list_trained`**: A file containing the trained Hi-C data in a list format.
- **`list_untrained`**: A file containing the untrained Hi-C data in a list format.
- **`batchsize`** (optional): The batch size used for embeddings generation (default is 128).
- **`epochs`** (optional): The number of epochs for model training (default is 1000).
- **`learningrate`** (optional): The learning rate for training (default is 0.001).
- **`loss_diff_threshold`** (optional): A threshold for early stopping based on loss difference (default is 1e-8).
  
#### Outputs:
- **Trained Model**: The model weights are saved as `weights.pt` in the output directory.
- **Predicted Structure**: The 3D structure of the predicted distances is saved in a `.pdb` file.
- **Training and Validation Plots**: Graphs showing the MSE loss, Spearman correlation, and training/validation loss curves.
- **Results File**: A text file containing the optimal dSCC (Spearman correlation), final loss, and final combined loss.

#### Workflow:
1. **Preprocessing**: The Hi-C matrices are normalized using the KR normalization, and node2vec embeddings are generated for both trained and untrained data.
2. **Model Training**: A GAT model is trained on the preprocessed Hi-C data using a combined loss function that minimizes MSE and a regularization term (Spearman correlation loss).
3. **Generalization**: Once the model is trained, it is tested on untrained Hi-C data, and the performance is evaluated using Spearman's correlation coefficient.
4. **Outputs**: The optimal model and the predicted structure are saved, and performance metrics (e.g., dSCC) are reported.

#### Usage:
To run the script `HiC-GAT_generalize_directly.py`, use the following command in the terminal:

```bash
python HiC-GAT_generalize_directly.py <list_trained> <list_untrained> [-bs BATCHSIZE] [-ep EPOCHS] [-lr LEARNINGRATE] [-loss_diff_threshold LOSS_DIFF_THRESHOLD]
```

### 2. `train_and_test_on_same_res.py`

#### Purpose:
This script trains a Graph Attention Network (GAT) model on Hi-C data embeddings and then generalizes the model to untrained Hi-C data at the same resolution. It focuses on predicting pairwise genomic distances from the embeddings and adjacency matrices and evaluates the model using Spearman's correlation.

#### Inputs:
- **`list_trained`**: A file containing the trained Hi-C data in list format.
- **`list_untrained`**: A file containing the untrained Hi-C data in list format.
- **`batchsize`** (optional): The batch size used for embeddings generation (default is 128).
- **`epochs`** (optional): The number of epochs for embeddings generation (default is 10).
- **`learningrate`** (optional): The learning rate for training (default is 0.0001).
- **`threshold`** (optional): The loss threshold for early stopping (default is 1e-8).

#### Outputs:
- **Trained Model**: The model weights are saved as `{name_trained}_weights.pt` in the output directory.
- **Training Loss Plot**: A plot of training loss over iterations saved as `{name_trained}_training_loss_plot.png`.
- **Predicted Structure**: The predicted 3D structure of the untrained data is saved in a `.pdb` file (`{name_untrained}_generalized_structure.pdb`).
- **Distance Comparison Plot**: A plot comparing the true vs predicted distances for the untrained data saved as `{name_untrained}_distance_comparison_plot.png`.
- **Results File**: A text file containing the optimal dSCC (Spearman correlation), saved as `{name_untrained}_generalized_log.txt`.

#### Workflow:
1. **Preprocessing**: The Hi-C matrices for both trained and untrained datasets are either loaded from existing files or generated from raw data if not already present. These matrices are normalized using KR normalization, and Node2Vec embeddings are generated for both datasets.
2. **Model Training**: A GAT model is trained on the preprocessed Hi-C data (trained dataset) using a combined loss function that minimizes MSE and a contrastive loss term that penalizes absolute differences in predicted and true distances.
3. **Testing**: After training, the model is tested on the same resolution HiC data, and the performance is evaluated using Spearman's correlation coefficient.
4. **Outputs**: The optimal model weights are saved, and the predicted structure for the untrained Hi-C data is stored in `.pdb` format. Performance metrics (e.g., dSCC) are reported, and training loss plots are saved to visualize the training process.

#### Usage:
To run the script `train_and_test_on_same_res.py`, use the following command in the terminal:
```bash
python train_and_test_on_same_res.py <list_trained> <list_untrained> [-bs BATCHSIZE] [-ep EPOCHS] [-lr LEARNINGRATE] [-th THRESHOLD]
```


* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath```: Path of the input file. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-h, --help  show this help message and exit<br />
	&nbsp;&nbsp;&nbsp;&nbsp;-c, --conversions <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;String of conversion constants of the form '[lowest, interval, highest]' for a set of equally spaced conversion factors, or of the form '[conversion]' for a single conversion factor. Default value: '[.1,.1,2]' <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-bs, --batchsize <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Batch size for embeddings generation. Default value: 128. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-ep, --epochs <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of epochs used for embeddings generation. Default value: 10. <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-lr, --learningrate <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learning rate for training GCNN. Default value: .001. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-th, --threshold <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loss threshold for training termination. Default value: 1e-8. <br />
    
* **Example**: ```python HiC-GNN_main.py Data/GM12878_1mb_chr19_list.txt```

### HiC-GNN_generalize.py
This script takes in two Hi-C maps in coordinate list format. The script generates embeddings for the first input map and then trains a model using the map and the corresponding embeddings. The script then generates embeddings for the second input map and aligns these embeddings to those of the first input map and tests the model generated from the first input using these aligned embeddings. The output is a structure corresponding to the second input generalized from the model trained on the first input. The script searches for files corresponding to the raw matrix format, the normalized matrix format, the embeddings, and a trained model for the inputs in the current working directory. For example, if the input file is ```input.txt```, then the script checks if ```Data/input_matrix.txt```, ```Data/input_matrix_KR_normed.txt```, and ```Data/input_embeddings.txt``` exists. If these files do not exist, then the script generates them automatically.

**Inputs**: 
1. A Hi-C contact map in either matrix format or coordinate list format.

**Outputs**: 
1. A .pdb file of the predicted 3D structure corresponding to the second input file in ```Outputs/input_2_generalized_structure.pdb```.
2. A .txt file depicting the optimal conversion value and the dSCC value of the output structure```Outputs/input_2_generalized_log.txt```.
3. A .pt file of the trained model weights corresponding to the first input file in ```Outputs/input_1_weights.pt```.
4. A .txt of the normalized Hi-C contact map corresponding to the KR normalization of both input files in ```Data/input_matrix_KR_normed.txt``` if these files don't exist already.
5. A .txt file of the embeddings corresponding to the input files in ```Data/input_embeddings.txt``` if these files don't exist already. 
6. A .txt file of the input files in matrix format in ```Data/input_matrix.txt``` if these files don't exist already.

**Usage**: ```python HiC-GNN_generalize.py input_filepath1 input_filepath2```

* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath1```: Path of the input file with which a model will be trained and later generalized on ```input_filepath2```. <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath2```: Path of the input file with which a generalized structure corresponding to a model trained on ```input_filepath1``` will be generated. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp; Same as ```HiC-GNN_main.py```
	
* **Example**: ```python HiC-GNN_generalize.py Data/GM12878_1mb_chr19_list.txt Data/GM12878_500kb_chr19_list.txt```

### HiC-GNN_embed.py
This script takes a single Hi-C contact map as an input and utilizes it to generate node embeddings. 

**Inputs**: 
1. A Hi-C contact map in either matrix format or coordinate list format.

**Outputs**: 
1. A .txt file of the embeddings corresponding to the input files in ```Data/input_embeddings.txt```.
2. (In the case that the input file was in list format) A .txt file of the input file in matrix format in ```Data/input_filename_matrix.txt```.

**Usage**: ```python HiC-GNN_generalize.py input_filepath```

* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath1```: Path of the input file with which a embeddings will be generated. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-h, --help  show this help message and exit<br />
	&nbsp;&nbsp;&nbsp;&nbsp;-bs, --batchsize <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Batch size for embeddings generation. Default value: 128. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-ep, --epochs <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of epochs used for embeddings generation. Default value: 10. <br />	
		
* **Example**: ```python HiC-GNN_embed.py Data/GM12878_1mb_chr19_list.txt```
