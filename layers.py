import torch
from torch import Tensor
from typing import Union, Tuple
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, GATConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
import torch.nn.functional as F

# information is aggregated from a node’s neighbors and combined with its own features

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        #print(f"Number of input channels: {in_channels}")
        self.out_channels = out_channels
        #print(f"Number of output channels: {out_channels}")
        self.normalize = normalize
        self.root_weight = root_weight
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False) # fully connected layer

        self.reset_parameters
         
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
          self.lin_r.reset_parameters()

    def adjust_weights(self, adj_t):
      #device = torch.device('cuda')
      sum_vec = adj_t.sum(dim=0)
      numerator = torch.ones(len(sum_vec))
      inverse = torch.divide(numerator, sum_vec)
      size = len(sum_vec)

      row = torch.arange(size, dtype=torch.long)
      index = torch.stack([row, row], dim=0)

      value = inverse.float()
      normalized = SparseTensor(row=index[0], col=index[1], value=value)
      norm_mat = matmul(normalized, adj_t)
      return norm_mat


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
          x: OptPairTensor = (x, x)
        
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out.float())
        x_r = x[1].long()
      
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r.float())

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        

        return out

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:

        norm_mat = self.adjust_weights(adj_t)
        return  matmul(norm_mat, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.6, bias=True, **kwargs):
        super(CustomGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout, bias=bias, **kwargs)
        
        self.lin_l = Linear(out_channels * heads if concat else out_channels, out_channels, bias=bias)
        
        # custom normalization added specific to HiC-data
        self.apply_custom_norm = True

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()

    def adjust_attention(self, edge_index, attention_scores):
        # Custom normalization for exp: L1 normalization 
        attention_sum = torch.sum(attention_scores, dim=1, keepdim=True)
        normalized_attention = attention_scores / (attention_sum + 1e-8) 
        return normalized_attention

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
       
        # computing attention scores
        x, edge_index, edge_attr = self.__distribute__(x, edge_index, edge_attr)
        # apply atttention mechanism from GATConv
        attention_scores, out = super().message_and_aggregate(edge_index, x)

        if self.apply_custom_norm:
            attention_scores = self.adjust_attention(edge_index, attention_scores)

        # normalized aggregation
        out = matmul(attention_scores, x)
        # after aggregation, linear transformed applied
        out = self.lin_l(out)

        return out

    def message_and_aggregate(self, edge_index: Tensor, x: Tensor) -> Tensor:
        attention_scores, out = super().message_and_aggregate(edge_index, x)
        
        if self.apply_custom_norm:
            attention_scores = self.adjust_attention(edge_index, attention_scores)
        
        return attention_scores, out