import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import cdist
from layers import SAGEConv
from layers import CustomGATConv
from torch_geometric.nn import GATConv

# GraphSAGE based model, builds upon SAGEConv layer 
# Can be used for node classification, link prediction, graph-level prediction
class Net(torch.nn.Module): 
  def __init__(self):
    super(Net, self).__init__()
    self.conv = SAGEConv(512, 512)
    self.densea = Linear(512,256)
    self.dense1 = Linear(256,128)
    self.dense2 = Linear(128,64)
    self.dense3 = Linear(64,3)
  
  def forward(self, x, edge_index):
    #print(f"Input x shape: {x.shape}")
    x = self.conv(x, edge_index)
    #print(f"Shape after conv: {x.shape}")
    x = x.relu() # non-linearity preserved for complex patterns
    x = self.densea(x)
    #print(f"Shape after densea: {x.shape}")
    x = x.relu()
    x = self.dense1(x)
    #print(f"Shape after dense1: {x.shape}")
    x = x.relu()
    x = self.dense2(x)
    #print(f"Shape after dense2: {x.shape}")
    x = x.relu()
    x = self.dense3(x)
    #print(f"Shape after dense3: {x.shape}")
    x = cdist(x, x, p=2)
    #print(f"Shape after cdist: {x.shape}")

    return x

  def get_model(self, x, edge_index):
    x = self.conv(x, edge_index)
    x = x.relu()
    x = self.densea(x)
    x = x.relu()
    x = self.dense1(x)
    x = x.relu()
    x = self.dense2(x)
    x = x.relu()
    x = self.dense3(x)

    return x 

# Total number of parameters in this case: 697475

class SmallerNet(torch.nn.Module): 
  def __init__(self):
    super(SmallerNet, self).__init__()
    self.conv = SAGEConv(512, 256)  # Update input channels to 512
    self.densea = Linear(256, 128)
    self.dense1 = Linear(128, 64)
    self.dense2 = Linear(64, 32)
    self.dense3 = Linear(32, 3)
  
  def forward(self, x, edge_index):
    #print(f"Input x shape: {x.shape}")
    x = self.conv(x, edge_index)
    #print(f"Shape after conv: {x.shape}")
    x = x.relu() # non-linearity preserved for complex patterns
    x = self.densea(x)
    #print(f"Shape after densea: {x.shape}")
    x = x.relu()
    x = self.dense1(x)
    #print(f"Shape after dense1: {x.shape}")
    x = x.relu()
    x = self.dense2(x)
    #print(f"Shape after dense2: {x.shape}")
    x = x.relu()
    x = self.dense3(x)
    #print(f"Shape after dense3: {x.shape}")
    x = cdist(x, x, p=2)
    #print(f"Shape after cdist: {x.shape}")

    return x

  def get_model(self, x, edge_index):
    x = self.conv(x, edge_index)
    x = x.relu()
    x = self.densea(x)
    x = x.relu()
    x = self.dense1(x)
    x = x.relu()
    x = self.dense2(x)
    x = x.relu()
    x = self.dense3(x)

    return x 
# Total number of parameters: 305731


### Graph Attention Network (GAT) Versions ###

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv = GATConv(512, 512, heads=4, concat=False)
        self.densea = Linear(512, 256)
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.densea(x)
        x = x.relu()
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        x = x.relu()
        x = self.dense3(x)
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.densea(x)
        x = x.relu()
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        x = x.relu()
        x = self.dense3(x)
        return x

class GATSmallerNet(torch.nn.Module): 
  def __init__(self):
    super(GATSmallerNet, self).__init__()
    self.conv = GATConv(512, 256) 
    self.densea = Linear(256, 128)
    self.dense1 = Linear(128, 64)
    self.dense2 = Linear(64, 32)
    self.dense3 = Linear(32, 3)
  
  def forward(self, x, edge_index):
    #print(f"Input x shape: {x.shape}")
    x = self.conv(x, edge_index)
    #print(f"Shape after conv: {x.shape}")
    x = x.relu() # non-linearity preserved for complex patterns
    x = self.densea(x)
    #print(f"Shape after densea: {x.shape}")
    x = x.relu()
    x = self.dense1(x)
    #print(f"Shape after dense1: {x.shape}")
    x = x.relu()
    x = self.dense2(x)
    #print(f"Shape after dense2: {x.shape}")
    x = x.relu()
    x = self.dense3(x)
    #print(f"Shape after dense3: {x.shape}")
    x = cdist(x, x, p=2)
    #print(f"Shape after cdist: {x.shape}")

    return x

  def get_model(self, x, edge_index):
    x = self.conv(x, edge_index)
    x = x.relu()
    x = self.densea(x)
    x = x.relu()
    x = self.dense1(x)
    x = x.relu()
    x = self.dense2(x)
    x = x.relu()
    x = self.dense3(x)

    return x 


### Variational Auto Encoder Versions ###
  
class VAE_GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE_GAT, self).__init__()
        self.encoder_conv1 = GATConv(input_dim, hidden_dim)  # GATConv layer
        self.encoder_conv2 = GATConv(hidden_dim, latent_dim)  # Second GATConv layer for latent space

        self.fc_mu = Linear(latent_dim, latent_dim)  # mean of the latent distribution
        self.fc_logvar = Linear(latent_dim, latent_dim)  # log variance

        self.decoder_fc1 = Linear(latent_dim, hidden_dim)  # First decoder layer
        self.decoder_fc2 = Linear(hidden_dim, output_dim)  # Final output layer

    def encode(self, x, edge_index):
        x = F.relu(self.encoder_conv1(x, edge_index))  # GATConv with ReLU activation
        x = F.relu(self.encoder_conv2(x, edge_index))  # Another GATConv 
        mu = self.fc_mu(x)  # Mean of the latent space
        logvar = self.fc_logvar(x)  # Log variance of the latent space
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + eps * std  # Reparameterize

    def decode(self, z):
        x = F.relu(self.decoder_fc1(z))  # Fully connected decoder layer with ReLU
        x = torch.sigmoid(self.decoder_fc2(x))  # Final layer with sigmoid to ensure output in [0,1]
        return x

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)  # Get the mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample latent space vector
        recon_x = self.decode(z)  # Decode from the latent space
        return recon_x, mu, logvar  # Return the reconstructed input and the latent distribution






