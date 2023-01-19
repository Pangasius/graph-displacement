from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import torch

class GraphEvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads, dropout=0.0, edge_dim=1):
        super().__init__()
        
        #we want the channels to be (x, y, vx, vy)
        assert(in_channels == out_channels)
        
        assert((edge_dim // heads) * heads == edge_dim)
        
        self.edge_dim = edge_dim
        self.edge_dim_embed = 6
        self.channels = in_channels
        self.gat_heads = 8
        
        #first we want to embed the edge_attr from edge_dim to self.edge_dim_embed
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=edge_dim, nhead=heads, dropout=dropout, batch_first=True, activation=F.elu, dim_feedforward=hidden_channels)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.encoder_resize = torch.nn.Linear(edge_dim, self.edge_dim_embed)
        
        #then we pass it through a GATv2Conv
        self.conv  = GATv2Conv(in_channels, hidden_channels, heads=self.gat_heads, dropout=dropout, edge_dim=self.edge_dim_embed, fill_value=-1.)
            
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.gat_heads*hidden_channels, nhead=self.gat_heads, dropout=dropout, batch_first=True, activation=F.elu, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize = torch.nn.Linear(self.gat_heads*hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        #x is a tensor of shape (T, N, in_channels)
        #for each time step, we will predict the next time step
        #the last one is empty and will be filled with the prediction
        
        #the edge_attr is a tensor of shape between (1, 1, edge_dim) and (T, N, edge_dim)

        #edge_index is a tensor of shape between (2, 1) and (2, T*N)
        
        xshape = x.shape
        x = x.reshape(xshape[0]*xshape[1], self.channels)
        
        #we embed the edge_attr
        edge_attr = edge_attr.reshape(-1, 1, self.edge_dim)
        edge_attr = self.transformer_encoder(edge_attr)
        edge_attr = self.encoder_resize(edge_attr)
        edge_attr = F.elu(edge_attr)

        #here T is treated as a batch dimension
        # x[i] is a tensor of shape (N*T, in_channels)
        y = self.conv(x, edge_index, edge_attr)
        # y is a tensor of shape (N*T, heads*hidden_channels)
        
        
        # different nodes will be considered as batches
        y = y.reshape(xshape[0]*xshape[1], 1, -1)
        # y is a tensor of shape (T, N, heads*hidden_channels)
        y = self.transformer_decoder(y, y)
        y = self.decoder_resize(y).reshape(xshape) + x.reshape(xshape) 
            
        return y