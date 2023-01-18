from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import torch

class GraphEvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads, dropout=0.0, edge_dim=1):
        super().__init__()
        
        #we want the channels to be (x, y, vx, vy)
        assert(in_channels == out_channels)
        
        self.conv  = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim, fill_value=-1.)
            
        self.post = torch.nn.Linear(heads*hidden_channels, hidden_channels)
        self.post2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.post3 = torch.nn.Linear(out_channels, hidden_channels)
        self.post4 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        #x is a tensor of shape (T, N, in_channels)
        #for each time step, we will predict the next time step
        #the last one is empty and will be filled with the prediction
        
        #the edge_attr is a tensor of shape between (1, 1) and (T, N)

        #edge_index is a tensor of shape between (2, 1) and (2, T*N)
        
        xshape = x.shape
        x = x.reshape(xshape[0]*xshape[1], 4)

        #here T is treated as a batch dimension
        # x[i] is a tensor of shape (N*T, in_channels)
        y = self.conv(x, edge_index, edge_attr)
        # y is a tensor of shape (N*T, heads*hidden_channels)
        y = F.elu(y)
        # different nodes will be considered as batches
        y = self.post(y)
        # y is a tensor of shape (N*T, out_channels)
        y = F.elu(y)
        y = (self.post2(y).reshape(xshape) + x.reshape(xshape)).reshape(xshape[0]*xshape[1], 4)
        
        y = F.elu(y)
        y = self.post3(y)
        
        y = F.elu(y)
        y = self.post4(y)
            
        return y.reshape(xshape) 