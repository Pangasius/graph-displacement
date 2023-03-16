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
        
        assert(messages > 0)
        
        #expand input so encoder is the size of the hidden channels
        self.encoder_resize = torch.nn.Linear(self.in_channels, self.hidden_channels)
        self.encoder_resize2 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        
        #we need an encoder to encode the messages before sending them
        encode_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=32)
        self.transformer_encoder = torch.nn.TransformerEncoder(encode_layer, num_layers=2)
        
        self.gat_resize = torch.nn.Linear(self.hidden_channels, self.hidden_channels * self.gat_heads)
        
        self.gatv2s = torch.nn.ModuleList()
        for i in range(messages):
            self.gatv2s.append(GATv2Conv(self.hidden_channels * self.gat_heads, self.hidden_channels, heads=self.gat_heads, dropout=dropout, concat=True, edge_dim=self.edge_dim))
                
        self.gat_resize2 = torch.nn.Linear(self.hidden_channels * self.gat_heads, self.hidden_channels)
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=32)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=2)
        
        self.decoder_resize = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.decoder_resize2 = torch.nn.Linear(self.hidden_channels, self.out_channels)

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

        #encode
        encoded = self.encoder_resize(x.reshape(-1, self.in_channels))
        encoded = F.gelu(encoded)
        encoded = self.encoder_resize2(encoded).reshape(-1, 1, self.hidden_channels)
        encoded = self.transformer_encoder(encoded)
        
        y = self.gat_resize(encoded).reshape(-1, self.hidden_channels * self.gat_heads)

        #here T is treated as a batch dimension
        for i in range(len(self.gatv2s)):
            y = self.gatv2s[i](y, edge_index, edge_attr)
            y = F.gelu(y)
            
        y = self.gat_resize2(y).reshape(xshape[0]*xshape[1], 1, self.hidden_channels)

        # different nodes will be considered as batches
        y = self.transformer_decoder(y, encoded).reshape(xshape[0]*xshape[1], self.hidden_channels)

        y = self.decoder_resize(y)
        y = F.gelu(y)
        y = self.decoder_resize2(y).reshape(xshape[0], xshape[1], self.out_channels)
        
        means = y[:,:,:self.out_channels//2] + x.reshape(xshape)[:,:,:self.out_channels//2]
        log_std = y[:,:,self.out_channels//2:]
        
        out = torch.cat((means, log_std), dim=2)

        #the output is a gaussian distribution for each dimension
        return out
    
    def show_gradients(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad != None :
                    print(name, param.grad.mean())
    
    def loss_direct(self, pred, target):
        #pred is a tensor of shape (T, N, 8)
        #target is a tensor of shape (T, N, 4)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        pred = pred.clone()
        std = torch.exp(pred[:,:,self.out_channels//2:])

        return self.diff(pred, std, target)

    def diff(self, pred, std, target) :
        target = target.to(pred.device)
        
        #see https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf
        #slide 16
        diff = (pred[:,:,:self.out_channels//2] - target)**2
        
        sample = torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device)
        
        if self.wrap :
            #https://www.geogebra.org/m/fvsyepzd
            special = torch.sin(diff * torch.pi) + torch.square(diff) / 20
        
            loss = torch.mean(special / (2 * std**2) + pred[:,:,self.out_channels//2:])
            
            return loss.requires_grad_(), sample % 1
        else : 
            loss = torch.mean(diff / (2 * std**2) + pred[:,:,self.out_channels//2:])
        
            return loss.requires_grad_(), sample
        
        
    def in_depth_parameters(self, x, normal = False) :
        
        all_params = {}
        
        if normal :
            x_from_normal = torch.normal(x[:,:,:self.out_channels//2], torch.exp(x[:,:,self.out_channels//2:]))[:,:,:2]
            
            x = x[:,:,:2].to(x.device)

        else :
            x_from_normal = x[:,:,:2]
        
        speed_diff_normal = torch.cat((torch.zeros(1).to(x.device), torch.mean(torch.square(x_from_normal[1:, :, :] - x_from_normal[:-1, :, :]), dim=(1,2))), dim=0)
        all_params['speed_normal'] = speed_diff_normal * 100 #to make it comparable to the other parameters
        
        center_of_mass = torch.mean(x, dim=(1,2))
        all_params['center_of_mass'] = center_of_mass
        
        spread = torch.std(x, dim=(1,2))
        all_params['spread'] = spread * 2
        
        return all_params
        
    def aggregate_parameters(self, pred, target):
        pred = pred.clone()
        target = target.to(pred.device)
        
        all_params_pred = self.in_depth_parameters(pred, normal=True)
        all_params_target = self.in_depth_parameters(target)
        
        loss = torch.tensor(0.).to(pred.device)
        for key in all_params_pred :
            loss = loss + torch.mean(torch.square(all_params_pred[key] - all_params_target[key])) * 50
            
        std = torch.exp(pred[:,:,self.out_channels//2:]).to(pred.device)
        sample = torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device)

        if self.wrap :
            return loss.requires_grad_(), sample % 1
        else : 
            return loss.requires_grad_(), sample
        
    def loss_recursive(self, pred, target):
        # we will compute aggregate statistics for the whole trajectory
        
        loss, sample = self.aggregate_parameters(pred, target) 

        return loss.requires_grad_(), sample
    
    def draw(self, pred) :
        std = torch.exp(pred[:,:,self.out_channels//2:])
        
        if self.wrap :
            return torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device) % 1
        else : 
            return torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device)

    
class Discriminator(torch.nn.Module) :
    """This class is a simple discriminator that will try to distinguish between real and fake trajectories for the recursive case"""
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super(Discriminator, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.heads = 4

        self.gatv2s = torch.nn.ModuleList()
        self.gatv2s.append(GATv2Conv(in_channels, hidden_channels, dropout=dropout, heads=self.heads, concat=True))
        self.gatv2s.append(GATv2Conv(hidden_channels * self.heads, hidden_channels, dropout=dropout, heads=self.heads, concat=True))
        self.gatv2s.append(GATv2Conv(hidden_channels * self.heads, hidden_channels, dropout=dropout, heads=self.heads, concat=True))
        self.gatv2s.append(GATv2Conv(hidden_channels * self.heads, hidden_channels // 2, dropout=dropout, heads=self.heads, concat=False))
        
        #reduce the number of information per node to 1
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=hidden_channels//2, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=hidden_channels)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=1)
        
        self.conv = torch.nn.Conv1d(hidden_channels // 2, 1, 1)
        
        self.pool = torch.nn.AdaptiveAvgPool1d(hidden_channels // 2)
        
        #have another decoder 
        self.transformer_decoder_2 = torch.nn.TransformerDecoder(decode_layer, num_layers=1)
        
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
            y = F.gelu(y)
            
        y = y.reshape(xshape[0], xshape[1], self.hidden_channels // 2)
        
        y = self.transformer_decoder(y, y)
        
        #we will reduce the number of nodes information to 1
        y = y.reshape(xshape[0], self.hidden_channels // 2, xshape[1])
        y = self.conv(y).reshape(xshape[0], xshape[1])
        
        #we will reduce the amount of nodes to hidden_channels
        y = F.gelu(y)
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
        self.discr = Discriminator(out_channels // 2, hidden_channels, dropout=dropout)
        
    def loss_recursive(self, pred, target, all_edges, grad=True):
        """Uses the discriminator to compute the loss"""
        #pred is a tensor of shape (T, N, 8)
        #target is a tensor of shape (T, N, 4)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        pred = pred.clone()

        target = target.to(pred.device)
         
        #pass the sample through the discriminator
         
        disc_out = self.discr(pred[:,:,:self.out_channels//2], all_edges)
        disc_true = self.discr(target, all_edges)
        
        #enforce a gradient penalty
        #see https://arxiv.org/pdf/1704.00028.pdf
       
        if grad :
            #we want x_hat to be for each node either the sample or the target
            x_hat = torch.zeros_like(pred[:,:,:self.out_channels//2])
            random_index = torch.randint(0, 2, (pred[:,:,:self.out_channels//2].shape[0], pred[:,:,:self.out_channels//2].shape[1], 1)).to(pred[:,:,:self.out_channels//2].device)
            x_hat = torch.where(random_index == 0, pred[:,:,:self.out_channels//2], target)
            
            gradients_disc = torch.autograd.grad(outputs=self.discr(x_hat, all_edges), inputs=x_hat, grad_outputs=torch.ones_like(self.discr(x_hat, all_edges)), create_graph=True, retain_graph=True)[0] #type: ignore
            norm_grad = (torch.norm(gradients_disc, p=2) - 1)**2 #type: ignore
        else : 
            norm_grad = 0

        #compute the loss of the discriminator
        loss_critic = torch.mean(disc_out) - torch.mean(disc_true) + 10 * norm_grad
        
        std = torch.exp(pred[:,:,self.out_channels//2:])
        
        loss_gen = -torch.mean(disc_out) + self.diff( pred,\
                                                     std,\
                                                     target)[0]
    
        loss = loss_gen + 5 * loss_critic

        return loss.requires_grad_(), pred[:,:,:self.out_channels//2].detach()