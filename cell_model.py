from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import torch

class GraphEvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1, messages=3, wrap=True):
        super().__init__()
        
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gat_heads = 8
        
        self.hidden_channels = hidden_channels
        
        self.wrap = wrap
        
        assert(messages > 1)
        
        self.gatv2s = []
        for i in range(messages):
            if i == 0:
                self.gatv2s.append(GATv2Conv(self.in_channels, hidden_channels, heads=self.gat_heads, dropout=dropout, concat=True, edge_dim=self.edge_dim))
            elif i == messages - 1 :
                self.gatv2s.append(GATv2Conv(hidden_channels * self.gat_heads, hidden_channels, heads=self.gat_heads, dropout=dropout, concat=False))
            else :
                self.gatv2s.append(GATv2Conv(hidden_channels * self.gat_heads, hidden_channels, heads=self.gat_heads, dropout=dropout, concat=True))
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize_1 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.decoder_resize_1_1 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.decoder_resize_2 = torch.nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x, edge_index, edge_attr, params):
        #x is a tensor of shape (T, N, in_channels)
        #for each time step, we will predict the next time step
        #the last one is empty and will be filled with the prediction
        
        #the edge_attr is a tensor of shape between (1, 1, edge_dim) and (T, N, edge_dim)

        #edge_index is a tensor of shape between (2, 1) and (2, T*N)
        
        xshape = x.shape
        
        #add to x the params 
        x = torch.cat((x, params.reshape(1, 1, params.shape[0]).repeat(xshape[0], xshape[1], 1)), dim=2)
        
        xshape = x.shape

        #here T is treated as a batch dimension
        y = x.reshape(-1, self.in_channels)
        for i in range(len(self.gatv2s)):
            y = self.gatv2s[i](y, edge_index, edge_attr)
            edge_attr = None
            y = F.elu(y)
        
        # y is a tensor of shape (N*T, hidden_channels)
        y = y.reshape(xshape[0]*xshape[1], 1, self.hidden_channels)
        
        #make x so that it can be added to the output
        z =  torch.cat((x.reshape(xshape[0]*xshape[1], 1, self.in_channels), torch.zeros(xshape[0]*xshape[1], 1, self.hidden_channels - xshape[2]).to(x.device)), dim=2).to(x.device)
        
        y = y + z

        # different nodes will be considered as batches
        y = self.transformer_decoder(y, y)
        y = torch.tanh(y)
        
        y = self.decoder_resize_1(y)
        y = F.leaky_relu(y)
        y = self.decoder_resize_1_1(y)
        y = F.leaky_relu(y)
        y = self.decoder_resize_2(y)
            
        #::2 is the mean
        #1::2 is the log std

        #the output is a gaussian distribution for each dimension
        return y.reshape(xshape[0], xshape[1], self.out_channels)
    
    def draw(self, pred) :
        pred = pred.clone()
        pred[:,:,1::2] = torch.exp(pred[:,:,1::2])
        sample = torch.normal(pred[:,:,::2], pred[:,:,1::2])
        if self.wrap :
            sample[:,:,:2] = sample[:,:,:2] % 1
            sample[:,:,2:] = sample[:,:,2:] - (sample[:,:,2:] > 0.5).float() + (sample[:,:,2:] < -0.5).float()
        return sample
    
    def loss_direct(self, pred, target):
        #pred is a tensor of shape (T, N, 4)
        #target is a tensor of shape (T, N, 2)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        pred = pred.clone()
        pred[:,:,1::2] = torch.exp(pred[:,:,1::2])
        sample = torch.normal(pred[:,:,::2], pred[:,:,1::2]).to(pred.device)
        target = target.to(pred.device)
        
        if self.wrap :
            sample[:,:,:2] = sample[:,:,:2] % 1
            sample[:,:,2:] = sample[:,:,2:] - (sample[:,:,2:] > 0.5).float() + (sample[:,:,2:] < -0.5).float()
         
        #see https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf
        #slide 16
        loss = torch.mean(((sample - target)**2) / (2 * pred[:,:,1::2]**2) + torch.log(pred[:,:,1::2]))
        
        #we will also try to ensure temporal coherence
        #we will try to ensure that the next time step is close to the current time step
        
        loss = loss + 0.05 * torch.mean((((sample[1:,:,:] - sample[:-1,:,:])**2) / (2 * pred[:-1,:,1::2]**2)) + torch.log(pred[:-1,:,1::2]))
        
        return loss.requires_grad_(), sample
         
    
    def to(self, device):
        self = super().to(device)
        for i in range(len(self.gatv2s)):
            self.gatv2s[i] = self.gatv2s[i].to(device)
        return self