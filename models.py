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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, heads=4):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=heads, concat=False)
        self.mu_layer = Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = Linear(hidden_dim // 2, latent_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

# Decoder using Linear layers to reconstruct embeddings
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, heads=4):
        super(Decoder, self).__init__()
        self.dense1 = Linear(latent_dim, hidden_dim // 2)
        self.dense2 = Linear(hidden_dim // 2, hidden_dim)
        self.conv1 = GATConv(hidden_dim, output_dim, heads=heads, concat=False)
    
    def forward(self, z, edge_index):
        x = F.relu(self.dense1(z))
        x = F.relu(self.dense2(x))
        x = self.conv1(x, edge_index)
        return x

# VAE + GAT model
class VAE_GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, heads=4):
        super(VAE_GAT, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, heads=heads)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, heads=heads)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, edge_index)
        
        pairwise_distances = torch.cdist(recon_x, recon_x, p=2)
        
        return recon_x, mu, logvar, pairwise_distances

# Variational Autoencoder loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss






