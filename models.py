import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LeakyReLU
from torch import cdist
from layers import SAGEConv
from layers import CustomGATConv
from torch_geometric.nn import GATConv, GCNConv

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

        #self.dropout = Dropout(p=0.4)
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU() 

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.densea(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
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

# Total number of parameters: 436355
class GATNetHeadsChanged4LayersLeakyReLU(torch.nn.Module):  # Updated with 4 linear layers
    def __init__(self):
        super(GATNetHeadsChanged4LayersLeakyReLU, self).__init__()
        self.conv = GATConv(512, 256, heads=2, concat=True)
        self.densea = Linear(512, 256)  
        self.dense1 = Linear(256, 128)
        self.dense2 = Linear(128, 64)
        self.dense3 = Linear(64, 3) 

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

# Total number of parameters: 165385
class OnlyGATNetLeakyReLU(torch.nn.Module):
    def __init__(self):
        super(OnlyGATNetLeakyReLU, self).__init__()
        self.conv1 = GATConv(512, 128, heads=2, concat=True)
        self.conv2 = GATConv(256, 64, heads=2, concat=True)
        self.conv3 = GATConv(128, 3, heads=1, concat=False) 

        self.dropout = Dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)

        x = cdist(x, x, p=2)  
        return x

    def get_model(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)

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


