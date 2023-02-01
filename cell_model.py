from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import torch

class GraphEvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1):
        super().__init__()
        
        #we want the channels to be (x, y, vx, vy)
        assert(in_channels == out_channels)
        
        self.tf_heads_attr = edge_dim // 2
        self.tf_heads_enc = in_channels//2
        
        assert(self.tf_heads_attr * 2 == edge_dim)
        assert(self.tf_heads_enc * 2 == in_channels)
        
        self.edge_dim = edge_dim
        self.edge_dim_embed = 10
        self.channels = in_channels
        self.gat_heads = 8
        
        #first we want to embed the edge_attr from edge_dim to self.edge_dim_embed
        encoder_layer_attr = torch.nn.TransformerEncoderLayer(d_model=self.edge_dim, nhead=self.tf_heads_attr, dropout=dropout, batch_first=True, activation=F.elu, dim_feedforward=hidden_channels)
        self.transformer_encoder_attr = torch.nn.TransformerEncoder(encoder_layer_attr, num_layers=2)
        
        encoder_layer_enc = torch.nn.TransformerEncoderLayer(d_model=self.channels, nhead=self.tf_heads_enc, dropout=dropout, batch_first=True, activation=F.elu, dim_feedforward=hidden_channels)
        self.transformer_encoder_enc = torch.nn.TransformerEncoder(encoder_layer_enc, num_layers=2)
        
        #then we pass it through a GATv2Conv
        self.conv  = GATv2Conv(self.channels, self.channels, heads=self.gat_heads, dropout=dropout, edge_dim=self.edge_dim, fill_value=-1.)
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.channels, nhead=self.tf_heads_enc, dropout=dropout, batch_first=True, activation=F.elu, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize_1 = torch.nn.Linear(self.channels*self.gat_heads, self.channels)
        self.decoder_resize_2 = torch.nn.Linear(self.channels, self.channels)

    def forward(self, x, edge_index, edge_attr):
        #x is a tensor of shape (T, N, in_channels)
        #for each time step, we will predict the next time step
        #the last one is empty and will be filled with the prediction
        
        #the edge_attr is a tensor of shape between (1, 1, edge_dim) and (T, N, edge_dim)

        #edge_index is a tensor of shape between (2, 1) and (2, T*N)
        
        xshape = x.shape
        
        #we embed the edge_attr
        edge_attr = edge_attr.reshape(-1, 1, self.edge_dim)
        edge_attr = self.transformer_encoder_attr(edge_attr)
        
        #we embed the positions
        x = x.reshape(-1, 1, self.channels)
        y_enc = self.transformer_encoder_enc(x)

        #here T is treated as a batch dimension
        y = y_enc.reshape(-1, self.channels)
        y = self.conv(y, edge_index, edge_attr)
        
        # y is a tensor of shape (N*T, heads*hidden_channels)
        y = y.reshape(xshape[0]*xshape[1], self.gat_heads, self.channels)

        # different nodes will be considered as batches
        y = self.transformer_decoder(y, y)
        y = y.reshape(xshape[0]*xshape[1], 1, self.gat_heads*self.channels)
        y = self.decoder_resize_1(y).reshape(xshape) + x.reshape(xshape) 
        
        y = F.elu(y).reshape(xshape[0]*xshape[1], 1, -1)
        y = self.decoder_resize_2(y).reshape(xshape)
            
        return y