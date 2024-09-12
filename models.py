import torch
from torch.nn import Linear
from torch import cdist
from layers import SAGEConv
from laers import CustomGATConv
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
    print(f"Input x shape: {x.shape}")
    x = self.conv(x, edge_index)
    print(f"Shape after conv: {x.shape}")
    x = x.relu() # non-linearity preserved for complex patterns
    x = self.densea(x)
    print(f"Shape after densea: {x.shape}")
    x = x.relu()
    x = self.dense1(x)
    print(f"Shape after dense1: {x.shape}")
    x = x.relu()
    x = self.dense2(x)
    print(f"Shape after dense2: {x.shape}")
    x = x.relu()
    x = self.dense3(x)
    print(f"Shape after dense3: {x.shape}")
    x = cdist(x, x, p=2)
    print(f"Shape after cdist: {x.shape}")

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
        self.conv = GATConv(512, 512, heads=4)
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
        self.conv = GATConv(512, 256, heads=4, concat=False)
        self.densea = Linear(256, 128)
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 32)
        self.dense3 = Linear(32, 3)

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
