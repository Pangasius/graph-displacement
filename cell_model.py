from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import torch

class GraphEvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1, messages=3, wrap=True):
        super().__init__()
        
        #we want the channels to be (x, y, vx, vy)
        assert(in_channels == out_channels)
        
        self.edge_dim = edge_dim
        self.channels = in_channels
        self.gat_heads = 8
        
        self.hidden_channels = hidden_channels
        
        self.wrap = wrap
        
        #first we want to embed the edge_attr from edge_dim to self.edge_dim_embed
        #encoder_layer_attr = torch.nn.TransformerEncoderLayer(d_model=self.edge_dim, nhead=self.tf_heads_attr, dropout=dropout, batch_first=True,dim_feedforward=hidden_channels)
        #self.transformer_encoder_attr = torch.nn.TransformerEncoder(encoder_layer_attr, num_layers=2)
        
        #encoder_layer_enc = torch.nn.TransformerEncoderLayer(d_model=self.channels, nhead=self.tf_heads_enc, dropout=dropout, batch_first=True, dim_feedforward=hidden_channels)
        #self.transformer_encoder_enc = torch.nn.TransformerEncoder(encoder_layer_enc, num_layers=2)
        
        #make "messages" layers of GATv2Conv with "heads" heads and "hidden_channels" hidden channels
        
        assert(messages > 1)
        
        self.gatv2s = []
        for i in range(messages):
            if i == 0:
                self.gatv2s.append(GATv2Conv(self.channels, hidden_channels, heads=self.gat_heads, dropout=dropout, concat=True, edge_dim=self.edge_dim))
            elif i == messages - 1 :
                self.gatv2s.append(GATv2Conv(hidden_channels * self.gat_heads, hidden_channels, heads=1, dropout=dropout, concat=True))
            else :
                self.gatv2s.append(GATv2Conv(hidden_channels * self.gat_heads, hidden_channels, heads=self.gat_heads, dropout=dropout, concat=True))
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=self.gat_heads // 2, dropout=dropout, batch_first=True, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize_1 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.decoder_resize_1_1 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.decoder_resize_2 = torch.nn.Linear(self.hidden_channels, self.channels)

    def forward(self, x, edge_index, edge_attr, params):
        #x is a tensor of shape (T, N, in_channels)
        #for each time step, we will predict the next time step
        #the last one is empty and will be filled with the prediction
        
        #the edge_attr is a tensor of shape between (1, 1, edge_dim) and (T, N, edge_dim)

        #edge_index is a tensor of shape between (2, 1) and (2, T*N)
        
        xshape = x.shape
        
        #we embed the edge_attr and the positions and the params
        edge_attr = edge_attr.reshape(edge_index.shape[1], 1, -1)
        
        #add the params to each edge
        edge_attr = torch.cat([edge_attr, params.reshape(1, 1, 4).repeat(edge_index.shape[1], 1, 1).to(x.device)], dim=-1)
        
        #add the positions and velocities to each edge
        #for this we identify the edge_index with the positions
        #we will have to reshape the positions to (T*N, 1, in_channels)
        x = x.reshape(-1, 1, self.channels)
        
        #we will have to reshape the edge_index to (T*N, 1, 2)
        edge_attr = torch.cat([edge_attr, x[edge_index[0]].reshape(-1, 1, self.channels), x[edge_index[1]].reshape(-1, 1, self.channels)], dim=-1)
        
        #finally pass it through the transformer
        #edge_attr = self.transformer_encoder_attr(edge_attr)
        #edge_attr = torch.tanh(edge_attr)
        
        
        #we embed the positions
        #y = self.transformer_encoder_enc(x)
        #y = torch.tanh(y)

        #here T is treated as a batch dimension
        y = x.reshape(-1, self.channels)
        for i in range(len(self.gatv2s)):
            y = self.gatv2s[i](y, edge_index, edge_attr)
            edge_attr = None
            y = F.elu(y)
        
        # y is a tensor of shape (N*T, heads*hidden_channels)
        y = y.reshape(xshape[0]*xshape[1], 1, self.hidden_channels)
        
        #make x so that it can be added to the output
        z =  torch.cat((x, torch.zeros(xshape[0]*xshape[1], 1, self.hidden_channels - xshape[2]).to(x.device)), dim=2).to(x.device)
        y = y + z

        # different nodes will be considered as batches
        y = self.transformer_decoder(y, y)
        y = torch.tanh(y)
        
        y = y.reshape(xshape[0]*xshape[1], 1, self.hidden_channels)
        y = self.decoder_resize_1(y) + z
        y = F.leaky_relu(y)
        y = self.decoder_resize_1_1(y)
        y = F.leaky_relu(y)
        y = self.decoder_resize_2(y)
        
        if self.wrap :
            y[:,:,:2] = torch.where(y[:,:,:2] < 0, y[:,:,:2] + 1, y[:,:,:2])
            y[:,:,:2] = torch.where(y[:,:,:2] > 1, y[:,:,:2] - 1, y[:,:,:2])
            
        return y.reshape(xshape)
    
    def to(self, device):
        self = super().to(device)
        for i in range(len(self.gatv2s)):
            self.gatv2s[i] = self.gatv2s[i].to(device)
        return self