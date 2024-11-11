import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LeakyReLU, LayerNorm
from torch import cdist
from layers import SAGEConv
from layers import CustomGATConv
from torch_geometric.nn import GATConv, GCNConv
from torch.nn import BatchNorm1d
from torch_sparse import SparseTensor


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


### Graph Attention Network (GAT) Versions ###

class GATNet(torch.nn.Module): # Total number of parameters: 436355
    def __init__(self):
        super(GATNet, self).__init__()
        # number of heads= 4 is overfitted the HiC dataset (I tried with GM12878)
        self.conv = GATConv(512, 512)
        self.densea = Linear(512, 256)
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 3)

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.dense1(x)
        x = x.relu()
        x = self.dropout(x)
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
    
class GATNetConvLayerChanged(torch.nn.Module): # Total number of parameters: 175171
    def __init__(self):
        super(GATNetConvLayerChanged, self).__init__()
        self.conv = GATConv(512, 256)  # Reduce hidden dim to 256 instead of 512
        self.densea = Linear(256, 128) 
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 32)
        self.dense3 = Linear(32, 3)

        self.dropout = Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
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

# Total number of parameters: 333571
class GATNetHeadsChanged(torch.nn.Module):  # Updated to use 2 heads in GATConv
    def __init__(self):
        super(GATNetHeadsChanged, self).__init__()
        self.conv = GATConv(512, 128, heads=2, concat=True)  # Two heads, concat=True  output becomes 256
        self.densea = Linear(256, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 32)
        self.dense3 = Linear(32, 3)

        self.dropout = Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        
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

class GATNetHeadsChangedLeakyReLU(torch.nn.Module):  # Updated to use 2 heads in GATConv
    def __init__(self):
        super(GATNetHeadsChangedLeakyReLU, self).__init__()
        self.conv = GATConv(512, 128, heads=2, concat=True)  # Two heads, concat=True  output becomes 256
        self.densea = Linear(256, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 32)
        self.dense3 = Linear(32, 3)

        #self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU() 

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.densea(x)
        x = self.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dense3(x)
        
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.leaky_relu(x)
    
        x = self.densea(x)
        x = self.leaky_relu(x)
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dense3(x)
        return x

# Total number of parameters: 
class GATNetHeadsChanged4Layers(torch.nn.Module):  # Updated with 4 linear layers
    def __init__(self):
        super(GATNetHeadsChanged4Layers, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 3) 

        #self.dropout = Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        #x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        #x = self.dropout(x)
        
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

# Total number of parameters: 436355

class GATNetHeadsChanged4LayersLeakyReLU(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersLeakyReLU, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 256)
        self.bn_a = BatchNorm1d(256) 
        self.dense1 = Linear(256, 128)
        self.bn1 = BatchNorm1d(128)
        self.dense2 = Linear(128, 64)
        self.bn2 = BatchNorm1d(64)
        self.dense3 = Linear(64, 3)
        self.dropout = Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.densea(x)
        x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.densea(x)
        x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        return x
    
class GATNetHeadsChanged4LayersReLU_LayerNorm(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersReLU_LayerNorm, self).__init__()
        self.conv = GATConv(256, 128, heads=2, concat=True)
        
        self.densea = Linear(256, 128)
        self.norm_a = LayerNorm(128)
        
        self.dense1 = Linear(128, 64)
        self.norm1 = LayerNorm(64)
        
        self.dense2 = Linear(64, 32)
        self.norm2 = LayerNorm(32)
        
        self.dense3 = Linear(32, 3)
        

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = x.relu()

        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = x.relu()
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = x.relu()
        
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = x.relu()
        
        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = x.relu()
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = x.relu()
        
        x = self.dense3(x)

        return x
    
class GATNetHeadsChanged4LayersReLU_LayerNormEmbed512(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersReLU_LayerNormEmbed512, self).__init__()
        # Updated input dimension to 512
        self.conv = GATConv(512, 256, heads=2, concat=True)
        
        self.densea = Linear(512, 256)
        self.norm_a = LayerNorm(256)
        
        self.dense1 = Linear(256, 128)
        self.norm1 = LayerNorm(128)
        
        self.dense2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)
        
        self.dense3 = Linear(64, 3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = F.relu(x)

        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = F.relu(x)
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = F.relu(x)
        
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = F.relu(x)
        
        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = F.relu(x)
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = F.relu(x)
        
        x = self.dense3(x)

        return x

class GATNetHeadsChanged4LayersReLU_LayerNormEmbed512WithResiduals(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersReLU_LayerNormEmbed512WithResiduals, self).__init__()
        # Updated input dimension to 512
        self.conv1 = GATConv(512, 256, heads=2, concat=True)  # Output will be 256 * 2 = 512
        self.norm1 = LayerNorm(512)
        
        self.densea = Linear(512, 256)
        self.norm_a = LayerNorm(256)
        
        self.dense1 = Linear(256, 128)
        self.norm1_2 = LayerNorm(128)
        
        self.dense2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)
        
        self.dense3 = Linear(64, 3)

        # Corrected linear transformations for dimension alignment in residuals
        self.align1 = Linear(512, 512)  # For GATConv layer output
        self.align_a = Linear(512, 256)  # For densea layer, matching input to densea
        self.align1_2 = Linear(256, 128)  # For dense1 layer
        self.align2 = Linear(128, 64)  # For dense2 layer

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        x_initial = self.align1(x)  # Align x_initial to 512 for residual
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # First dense layer with residual connection
        x_initial = self.align_a(x)  # Align to 256
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Second dense layer with residual connection
        x_initial = self.align1_2(x)  # Align to 128
        x = self.dense1(x)
        x = self.norm1_2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Third dense layer with residual connection
        x_initial = self.align2(x)  # Align to 64
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Final dense layer
        x = self.dense3(x)
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        # Forward pass for evaluation
        x_initial = self.align1(x)  # Align x_initial to 512 for residual
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # First dense layer with residual connection
        x_initial = self.align_a(x)  # Align to 256
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Second dense layer with residual connection
        x_initial = self.align1_2(x)  # Align to 128
        x = self.dense1(x)
        x = self.norm1_2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Third dense layer with residual connection
        x_initial = self.align2(x)  # Align to 64
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Final dense layer
        x = self.dense3(x)
        return x

class GATNetSelectiveResiduals(torch.nn.Module):
    def __init__(self):
        super(GATNetSelectiveResiduals, self).__init__()
        # GATConv layer: input 512, output 256 * 2 = 512 (after heads concatenation)
        self.conv = GATConv(512, 256, heads=2, concat=True)
        
        self.densea = Linear(512, 256)
        self.norm_a = LayerNorm(256)
        # Residual alignment layer for `densea`
        self.align_densea = Linear(512, 256)

        self.dense1 = Linear(256, 128)
        self.norm1 = LayerNorm(128)
        # Residual alignment layer for `dense1`
        self.align_dense1 = Linear(256, 128)

        self.dense2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)
        # Residual alignment layer for `dense2`
        self.align_dense2 = Linear(128, 64)

        self.dense3 = Linear(64, 3)

    def forward(self, x, edge_index):
        # Initial layer
        x = self.conv(x, edge_index)
        x = F.relu(x)
        
        # First dense layer with residual connection
        x_initial = self.align_densea(x)  # Align to 256 for residual
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection
        
        # Second dense layer with residual connection
        x_initial = self.align_dense1(x)  # Align to 128 for residual
        x = self.dense1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection
        
        # Third dense layer with residual connection
        x_initial = self.align_dense2(x)  # Align to 64 for residual
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Final layer without residual connection
        x = self.dense3(x)
        
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        # Initial layer
        x = self.conv(x, edge_index)
        x = F.relu(x)
        
        # First dense layer with residual connection
        x_initial = self.align_densea(x)  # Align to 256 for residual
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection
        
        # Second dense layer with residual connection
        x_initial = self.align_dense1(x)  # Align to 128 for residual
        x = self.dense1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection
        
        # Third dense layer with residual connection
        x_initial = self.align_dense2(x)  # Align to 64 for residual
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Final layer without residual connection
        x = self.dense3(x)
        
        return x

class GATNetSelectiveResidualsUpdated(torch.nn.Module):
    def __init__(self):
        super(GATNetSelectiveResidualsUpdated, self).__init__()
        
        # GATConv layer: input 512, output 256 * 2 = 512 (after heads concatenation)
        self.conv = GATConv(512, 256, heads=2, concat=True)
        
        self.densea = Linear(512, 256)
        self.norm_a = LayerNorm(256)
        self.align_densea = Linear(512, 256)  # Align to 256 for residual

        self.dense1 = Linear(256, 128)
        self.norm1 = LayerNorm(128)
        self.align_dense1 = Linear(256, 128)  # Align to 128 for residual with `dense1`

        self.dense2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)

        self.dense3 = Linear(64, 3)

    def forward(self, x, edge_index):
        # Initial GATConv layer
        x = self.conv(x, edge_index)
        x = F.relu(x)

        # First dense layer with residual connection
        x_initial = self.align_densea(x)  # Align to 256 for residual
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Second dense layer with residual connection
        x_initial = self.align_dense1(x)  # Align to 128 for residual
        x = self.dense1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Third dense layer without residual connection
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # Final layer without residual connection
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        # Initial layer
        x = self.conv(x, edge_index)
        x = F.relu(x)

        # First dense layer with residual connection
        x_initial = self.align_densea(x)  # Align to 256 for residual
        x = self.densea(x)
        x = self.norm_a(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Second dense layer with residual connection
        x_initial = self.align_dense1(x)  # Align to 128 for residual
        x = self.dense1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + x_initial  # Residual connection

        # Third dense layer without residual connection
        x = self.dense2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # Final layer without residual connection
        x = self.dense3(x)
        
        return x

class GATNetSelectiveResidualsUpdatedLayerNorm(torch.nn.Module):
    def __init__(self):
        super(GATNetSelectiveResidualsUpdatedLayerNorm, self).__init__()
        
        self.conv = GATConv(512, 256, heads=2, concat=True)
        
        self.densea = Linear(512, 256)
        self.norm_a = LayerNorm(256)
        self.align_densea = Linear(512, 256) 
        self.norm_residual_a = LayerNorm(256)  

        self.dense1 = Linear(256, 128)
        self.norm1 = LayerNorm(128)
        self.align_dense1 = Linear(256, 128)  
        self.norm_residual_1 = LayerNorm(128)  

        self.dense2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)

        self.dense3 = Linear(64, 3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        #x = F.relu(x)
        x = F.gelu(x)

        x_initial = self.align_densea(x)  
        x = self.densea(x)
        x = self.norm_a(x)
        #x = F.relu(x)
        x = F.gelu(x)
        x = x + x_initial 
        x = self.norm_residual_a(x)  

        x_initial = self.align_dense1(x) 
        x = self.dense1(x)
        x = self.norm1(x)
        #x = F.relu(x)
        x = F.gelu(x)
        x = x + x_initial 
        x = self.norm_residual_1(x) 

        x = self.dense2(x)
        x = self.norm2(x)
        #x = F.relu(x)
        x = F.gelu(x)

        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        #x = F.relu(x)
        x = F.gelu(x)

        x_initial = self.align_densea(x)  
        x = self.densea(x)
        x = self.norm_a(x)
        #x = F.relu(x)
        x = F.gelu(x)
        x = x + x_initial  
        x = self.norm_residual_a(x) 

        x_initial = self.align_dense1(x)  
        x = self.dense1(x)
        x = self.norm1(x)
        #x = F.relu(x)
        x = F.gelu(x)
        x = x + x_initial  
        x = self.norm_residual_1(x)  

        x = self.dense2(x)
        x = self.norm2(x)
        #x = F.relu(x)
        x = F.gelu(x)

        x = self.dense3(x)
        
        return x


class GATNetHeadsChanged4LayersLeakyReLU_LayerNorm(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersLeakyReLU_LayerNorm, self).__init__()
        self.conv = GATConv(256, 128, heads=2, concat=True)
        
        self.densea = Linear(256, 128)
        self.norm_a = LayerNorm(128)
        
        self.dense1 = Linear(128, 64)
        self.norm1 = LayerNorm(64)
        
        self.dense2 = Linear(64, 32)
        self.norm2 = LayerNorm(32)
        
        self.dense3 = Linear(32, 3)
        

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = F.leaky_relu(x)

        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = F.leaky_relu(x)
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = F.leaky_relu(x)
        
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.densea(x)
        x = self.norm_a(x)  # Apply layer normalization
        x = F.leaky_relu(x)
        
        x = self.dense1(x)
        x = self.norm1(x)  # Apply layer normalization
        x = F.leaky_relu(x)
        
        x = self.dense2(x)
        x = self.norm2(x)  # Apply layer normalization
        x = F.leaky_relu(x)
        
        x = self.dense3(x)

        return x

class GATNetHeadsChanged4LayersLeakyReLUHeads4(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersLeakyReLUHeads4, self).__init__()
        self.conv = GATConv(256, 256, heads=4, concat=True)
        self.densea = Linear(1024, 256)
        self.bn_a = BatchNorm1d(256) 
        self.dense1 = Linear(256, 128)
        self.bn1 = BatchNorm1d(128)
        self.dense2 = Linear(128, 64)
        self.bn2 = BatchNorm1d(64)
        self.dense3 = Linear(64, 3)
        self.dropout = Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.densea(x)
        x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.densea(x)
        x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        return x

class GATNetHeadsChanged4LayersLeakyReLUEmbed128(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged4LayersLeakyReLUEmbed128, self).__init__()
        self.conv = GATConv(128, 128, heads=2, concat=True)
        self.densea = Linear(256, 128)
        #self.bn_a = BatchNorm1d(256) 
        self.dense1 = Linear(128, 64)
        #self.bn1 = BatchNorm1d(128)
        self.dense2 = Linear(64, 32)
        #self.bn2 = BatchNorm1d(64)
        self.dense3 = Linear(32, 3)
        self.dropout = Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.densea(x)
        #x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.dense1(x)
        #x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        #x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.densea(x)
        #x = self.bn_a(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense1(x)
        #x = self.bn1(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense2(x)
        #x = self.bn2(x)  # Apply batch normalization
        x = F.leaky_relu(x)
        x = self.dense3(x)

        return x


# Total number of parameters: 148483
class GATNetReduced2LayersLeakyReLU(torch.nn.Module):
    def __init__(self):
        super(GATNetReduced2LayersLeakyReLU, self).__init__()
        self.conv = GATConv(512, 128, heads=2, concat=True)  
        
        self.dense1 = Linear(256, 64)  
        self.dense2 = Linear(64, 3)    
        
        self.dropout = Dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)  
        x = F.leaky_relu(x)         
        x = self.dropout(x)           #

        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        
        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        
        return x

# Total number of parameters: 428291
class GATNetHeadsChanged3LayersLeakyReLU(torch.nn.Module):  # Updated with 3 linear layers
    def __init__(self):
        super(GATNetHeadsChanged3LayersLeakyReLU, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 3)

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
  
        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        
        return x

# Total number of parameters: 411651
class GATNetHeadsChanged3LayersLeakyReLUv2(torch.nn.Module):  # Updated number of neurons
    def __init__(self):
        super(GATNetHeadsChanged3LayersLeakyReLUv2, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 64)
        self.dense2 = Linear(64, 3)

        #self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.densea(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
  
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        #print("Final Layer Output Mean:", x.mean().item(), "Std:", x.std().item())
        
        return x

class GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim256(torch.nn.Module):  # Updated number of neurons with different embedding dimension which is 256
    def __init__(self, scaling_factor=1.0):
        super(GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim256, self).__init__()
        self.conv = GATConv(256, 128, heads=2, concat=True)
        self.norm1 = LayerNorm(128 * 2) 

        self.densea = Linear(256, 128)  
        self.norm2 = LayerNorm(128) 
        #self.bn1 = BatchNorm1d(128)  

        self.dense1 = Linear(128, 64)
        self.norm3 = LayerNorm(64)  
        #self.bn2 = BatchNorm1d(64) 

        self.dense2 = Linear(64, 3)

        #self.dropout = Dropout(p=0.2)
        #self.dropout_scale = 1 - self.dropout.p 
        #self.scaling_factor = scaling_factor

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.densea(x)
        x = self.norm2(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.dense1(x)
        x = self.norm3(x)
        #x = self.bn2(x) 
        x = F.leaky_relu(x)
        
        x = self.dense2(x)
     
  
        x = cdist(x, x, p=2) 
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = self.norm2(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x)
        

        x = self.dense1(x)
        x = self.norm3(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = self.dense2(x)
        
        return x 

class GATNetHeadsChanged3LayersEmbeddingDim256Entropy(torch.nn.Module):
    def __init__(self):
        super(GATNetHeadsChanged3LayersEmbeddingDim256Entropy, self).__init__()
        self.conv = GATConv(256, 128, heads=2, concat=True)
        self.densea = Linear(256, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 3)
        self.alpha = 0.01

    def forward(self, x, edge_index):
        x, attn_weights = self.conv(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        
        x = self.densea(x)
        x = F.relu(x)
        
        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        
        x = torch.cdist(x, x, p=2)  
        return x, attn_weights

    def get_model(self, x, edge_index):
        x, _ = self.conv(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        
        x = self.densea(x)
        x = F.relu(x)
        
        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        
        return x

class GATNetHeadsChanged3LayersLeakyReLUv3EmbeddingDim256(torch.nn.Module):  # Updated number of neurons with different embedding dimension which is 256
    def __init__(self):
        super(GATNetHeadsChanged3LayersLeakyReLUv3EmbeddingDim256, self).__init__()
        self.conv = GATConv(256, 128, heads=2, concat=True)
        self.densea = Linear(256, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 32)
        self.dense3 = Linear(32, 3)

        #self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.densea(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.dense1(x)
        x = F.leaky_relu(x)

        x = self.dense2(x)
        x = F.leaky_relu(x)
        x = self.dense3(x)

        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        x = self.dense3(x)
        
        return x


class GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim128(torch.nn.Module):  
    def __init__(self):
        super(GATNetHeadsChanged3LayersLeakyReLUv2EmbeddingDim128, self).__init__()
        self.conv = GATConv(128, 128, heads=2, concat=True)
        self.densea = Linear(256, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 3)

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
  
        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        
        return x

# Total number of parameters: 337795
class GATNetHeadsChanged3LayersLeakyReLUv3(torch.nn.Module):  # Updated number of neurons
    def __init__(self):
        super(GATNetHeadsChanged3LayersLeakyReLUv3, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 128)  
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 3)

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
  
        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
    
        x = self.densea(x)
        x = F.leaky_relu(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        
        return x

# Total number of parameters: 305283
class GATNetHeadsChanged4LayerEmbedding256(torch.nn.Module):  # Updated node2vec embeddings 256 rather than 512
    def __init__(self):
        super(GATNetHeadsChanged4LayerEmbedding256, self).__init__()
        self.conv = GATConv(256, 256, heads=2, concat=True)  # Two heads, concat=True, output becomes 512
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 3) 

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        
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
# Total number of parameters: 307267
class GATNetHeadsChanged4LayerEmbedding256Dense(torch.nn.Module):  # Updated node2vec embeddings 256 rather than 512 and one more Linear layer added
    def __init__(self):
        super(GATNetHeadsChanged4LayerEmbedding256Dense, self).__init__()
        self.conv = GATConv(256, 256, heads=2, concat=True)  # Two heads, concat=True, output becomes 512
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 32)
        self.dense4 = Linear(32,3) 

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        x = x.relu()
        x = self.dense3(x)
        x = x.relu()
        x = self.dense4(x)
  
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
        x = x.relu()
        x = self.dense4(x)
        
        return x

# Total number of parameters: 438339
class GATNetHeadsChanged4LayerEmbedding512Dense(torch.nn.Module):  # Updated node2vec embeddings 256 rather than 512 and one more Linear layer added
    def __init__(self):
        super(GATNetHeadsChanged4LayerEmbedding512Dense, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)  # Two heads, concat=True, output becomes 512
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 32)
        self.dense4 = Linear(32,3) 

        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        x = x.relu()
        x = self.dense3(x)
        x = x.relu()
        x = self.dense4(x)
  
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
        x = x.relu()
        x = self.dense4(x)
        
        return x

  
class GATNetReduced(torch.nn.Module):
    def __init__(self):
        super(GATNetReduced, self).__init__()
        self.conv = GATConv(512, 512, heads=2, concat=True)  # Output size becomes 1024 with concat=True
        #self.batch_norm1 = BatchNorm1d(1024)
        #self.densea = Linear(1024, 128)
        self.densea = Linear(1024,256)
        #self.batch_norm2 = BatchNorm1d(128) 
        self.dense1 = Linear(256, 64)
        #self.batch_norm3 = BatchNorm1d(64)
        self.dense2 = Linear(64, 3)
        self.dropout = Dropout(p=0.4)  

    def forward(self, x, edge_index):
        # GAT Layer
        x = self.conv(x, edge_index)
        x = x.relu()
        #x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Dense layers
        x = self.densea(x)
        x = x.relu()
        #x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = x.relu()
        #x = self.batch_norm3(x)
        x = self.dense2(x)
        
        x = cdist(x, x, p=2)
        return x

    def get_model(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        #x = self.batch_norm1(x)
        x = self.densea(x)
        x = x.relu()
        #x = self.batch_norm2(x)
        x = self.dense1(x)
        x = x.relu()
        #x = self.batch_norm3(x)
        x = self.dense2(x)
        return x
  
class GATNetMoreReduced(torch.nn.Module):
    def __init__(self):
        super(GATNetMoreReduced, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True) 
        self.densea = Linear(512, 128) 
        self.dense1 = Linear(128, 32)   
        self.dense2 = Linear(32, 3)     
        self.dropout = Dropout(p=0.4)  

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.densea(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        
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
        return x
    
### Variational Auto Encoder Versions ###

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(512, 512, heads=2, concat=True)  # Output becomes 1024
        self.conv2 = GCNConv(1024, 512)  # Reduces output size back to 512
        self.densea = Linear(512, 128)  # Further reduces to 128
        self.dense_latent = Linear(128, 64)  # Latent space of 64
        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index):
        #print(f"Shape of x: {x.shape}")
        # GATConv layer
        x = self.conv1(x, edge_index)
        #print(f"After GATConv1, x.shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCNConv layer
        x = self.conv2(x, edge_index)
        #print(f"After GCNConv2, x.shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout(x)

        # Dense layers to latent space
        x = self.densea(x)
        #print(f"After densea, x.shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout(x)
        latent = self.dense_latent(x)
        #print(f"Latent space, latent.shape: {latent.shape}")
        return latent

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Dense layers to go back to larger dimensions
        self.dense1 = Linear(64, 128)
        self.dense2 = Linear(128, 512)
        self.conv1 = GCNConv(512, 512)  # GCNConv layer
        self.conv2 = GATConv(512, 512, heads=2, concat=True)  # Final GATConv for reconstruction
        self.dropout = Dropout(p=0.3)

    def forward(self, z, edge_index):
        # Dense layers from latent space
        x = F.relu(self.dense1(z))
        #print(f"After dense1 in Decoder, x.shape: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        #print(f"After dense2 in Decoder, x.shape: {x.shape}")
        x = self.dropout(x)
        
        # GCNConv layer
        x = self.conv1(x, edge_index)
        #print(f"After GCNConv in Decoder, x.shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout(x)
        
        # GATConv layer for final reconstruction
        x = self.conv2(x, edge_index)
        #print(f"After GATConv in Decoder, x.shape: {x.shape}")
        return x

class AutoencoderGAT_GCN(torch.nn.Module):
    def __init__(self):
        super(AutoencoderGAT_GCN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, edge_index):
        latent = self.encoder(x, edge_index)
        #print(f"Latent representation from Encoder, latent.shape: {latent.shape}")
        recon_x = self.decoder(latent, edge_index)
        #print(f"Reconstructed x, recon_x.shape: {recon_x.shape}")
        pairwise_distances = torch.cdist(recon_x, recon_x, p=2)
        #print(f"Pairwise distances, pairwise_distances.shape: {pairwise_distances.shape}")
        return pairwise_distances


