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
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=128)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize = torch.nn.Conv1d(self.hidden_channels, self.out_channels, 1)

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
            y = F.gelu(y)
        
        # y is a tensor of shape (N*T, hidden_channels)
        y = y.reshape(xshape[0]*xshape[1], 1, self.hidden_channels)

    
        # different nodes will be considered as batches
        y = self.transformer_decoder(y, y)
        y = torch.tanh(y).reshape(xshape[0]*xshape[1], self.hidden_channels, 1)
        y = self.decoder_resize(y).reshape(xshape[0], xshape[1], self.out_channels)
        
        means = torch.tanh(y[:,:,:2]) + x.reshape(xshape)[:,:,:2]
        log_std = y[:,:,2:]
        
        out = torch.cat((means[:,:,0].unsqueeze(-1),\
                         log_std[:,:,0].unsqueeze(-1),\
                         means[:,:,1].unsqueeze(-1),\
                         log_std[:,:,1].unsqueeze(-1)), dim=2)
        
        #::2 is the mean
        #1::2 is the log std

        #the output is a gaussian distribution for each dimension
        return y
    
    def loss_direct(self, pred, target):
        #pred is a tensor of shape (T, N, 8)
        #target is a tensor of shape (T, N, 4)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        pred = pred.clone()
        std = torch.exp(pred[:,:,1::2])
        
        sample = torch.normal(pred[:,:,::2], std).to(pred.device)
        
        return self.diff(sample, pred, std, target)

    def diff(self, sample, pred, std, target) :
        target = target.to(pred.device)
        
        #see https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf
        #slide 16
        diff = (sample - target)**2
        
        #https://www.geogebra.org/m/fvsyepzd
        special = torch.sin(diff * torch.pi) + torch.square(diff) / 10
        
        loss = torch.mean(torch.sum((special / (2 * std**2)) + pred[:,:,1::2], dim=2))
        
        return loss.requires_grad_(), sample.detach() % 1
    
    def loss_recursive(self, pred, target):
        # we will compute aggregate statistics for the whole trajectory
        pred = pred.clone()
        std = torch.exp(pred[:,:,1::2])
        target = target.to(pred.device)
        
        sample = torch.normal(pred[:,:,::2], std).to(pred.device)
             
        average_speed_diff = torch.mean(torch.square(sample[1:, :, :] - sample[-1, :, :]) - \
                                        torch.square(target[1:, :, :] - target[-1, :, :]))
        
        center_of_mass_diff = torch.mean(torch.square(torch.mean(sample, dim=0) - torch.mean(target, dim=0)))
        
        spread_diff = torch.mean(torch.square(torch.std(sample, dim=0) - torch.std(target, dim=0)))
        
        loss = average_speed_diff + center_of_mass_diff + spread_diff + self.diff(sample, pred, std, target)[0]
        
        return loss.requires_grad_(), sample.detach() % 1
    
    def draw(self, pred) :
        std = torch.exp(pred[:,:,1::2])
        
        return torch.normal(pred[:,:,::2], std).to(pred.device) % 1
         
    def to(self, device):
        self = super().to(device)
        for i in range(len(self.gatv2s)):
            self.gatv2s[i] = self.gatv2s[i].to(device)
        return self


    
class Discriminator(torch.nn.Module) :
    """This class is a simple discriminator that will try to distinguish between real and fake trajectories for the recursive case"""
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super(Discriminator, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.gatv2s = torch.nn.ModuleList()
        self.gatv2s.append(GATv2Conv(in_channels, hidden_channels, dropout=dropout, heads=4, concat=False))
        self.gatv2s.append(GATv2Conv(hidden_channels, hidden_channels, dropout=dropout, heads=4, concat=False))
        self.gatv2s.append(GATv2Conv(hidden_channels, hidden_channels, dropout=dropout, heads=4, concat=False))
        self.gatv2s.append(GATv2Conv(hidden_channels, hidden_channels, dropout=dropout, heads=4, concat=False))
        
        #reduce the number of information per node to 1
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=hidden_channels//2, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.conv = torch.nn.Conv1d(hidden_channels // 2, 1, 1)
        
        self.pool = torch.nn.AdaptiveAvgPool1d(hidden_channels // 2)
        
        #have another decoder 
        self.transformer_decoder_2 = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.conv_2 = torch.nn.Conv1d(hidden_channels // 2, 1, 1)

    def forward(self, to_score, edge):
        #this function assigns a score to a trajectory such that the score is high for real trajectories and low for fake trajectories
        #to_score is a tensor of shape (T, N, 4)
        #edge is a tensor of shape (2, E)
        
        #we will reshape the tensor to be able to use the GATv2
        xshape = to_score.shape
        y = to_score.reshape(-1, self.in_channels)
        
        #we will use the GATv2 to compute the hidden representation
        for i in range(len(self.gatv2s)):
            y = self.gatv2s[i](y, edge)
            y = F.elu(y)
            
        y = y.reshape(xshape[0], xshape[1], self.hidden_channels // 2)
        
        y = self.transformer_decoder(y, y)
        
        #we will reduce the number of nodes information to 1
        y = y.reshape(xshape[0], self.hidden_channels // 2, xshape[1])
        y = self.conv(y).reshape(xshape[0], xshape[1])
        
        #we will reduce the amount of nodes to hidden_channels
        y = F.elu(y)
        y = self.pool(y)
        
        y = y.reshape(1, xshape[0], self.hidden_channels // 2)
        
        y = self.transformer_decoder_2(y, y)
        
        y = y.reshape(xshape[0], self.hidden_channels // 2, 1)
        
        y = self.conv_2(y).reshape(xshape[0])
        
        y = torch.tanh(y)
        
        return y.mean()

class GraphEvolutionDiscr(GraphEvolution):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1, messages=3, wrap=True) :
        super(GraphEvolutionDiscr, self).__init__(in_channels, out_channels, hidden_channels, dropout=dropout, edge_dim=edge_dim, messages=messages, wrap=wrap)
        self.discr = Discriminator(out_channels, hidden_channels, dropout=dropout)
        
    def loss_recursive(self, pred, target, all_edges, grad=True):
        """Uses the discriminator to compute the loss"""
        #pred is a tensor of shape (T, N, 8)
        #target is a tensor of shape (T, N, 4)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        pred = pred.clone()
        std = torch.exp(pred[:,:,1::2])
        
        sample = torch.normal(pred[:,:,::2], std).to(pred.device) % 1
        target = target.to(pred.device)
         
        #pass the sample through the discriminator
         
        disc_out = self.discr(sample, all_edges)
        disc_true = self.discr(target, all_edges)
        
        #enforce a gradient penalty
        #see https://arxiv.org/pdf/1704.00028.pdf
       
        if grad :
            #we want x_hat to be for each node either the sample or the target
            x_hat = torch.zeros_like(sample)
            random_index = torch.randint(0, 2, (sample.shape[0], sample.shape[1], 1)).to(sample.device)
            x_hat = torch.where(random_index == 0, sample, target)
            
            gradients_disc = torch.autograd.grad(outputs=self.discr(x_hat, all_edges), inputs=x_hat, grad_outputs=torch.ones_like(self.discr(x_hat, all_edges)), create_graph=True, retain_graph=True)[0]
            norm_grad = (torch.norm(gradients_disc, p=2) - 1)**2 #type: ignore
        else : 
            norm_grad = 0

        #compute the loss of the discriminator
        loss_critic = torch.mean(disc_out) - torch.mean(disc_true) + 10 * norm_grad
        
        loss_gen = -torch.mean(disc_out) + self.diff(sample,\
                                                     pred,\
                                                     std,\
                                                     target)[0]
    
        loss = loss_gen + 5 * loss_critic
        
        return loss.requires_grad_(), sample.detach()