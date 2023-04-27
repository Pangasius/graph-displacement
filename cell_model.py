from torch_geometric.nn import GATv2Conv, GeneralConv
import torch.nn.functional as F

import torch

class Gatv2Predictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1, messages=3, wrap=True, absolute=0, heads=8):
        super().__init__()
        
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_passers_heads = heads
        
        self.hidden_channels = hidden_channels
        
        self.wrap = wrap
        
        self.messages = messages
        
        self.dropout = dropout
        
        assert(messages > 0)
        
        #expand input so encoder is the size of the hidden channels
        self.encoder_resize = torch.nn.Linear(self.in_channels, self.hidden_channels)
        
        #we need an encoder to encode the messages before sending them
        encode_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=self.hidden_channels)
        self.transformer_encoder = torch.nn.TransformerEncoder(encode_layer, num_layers=6)
        
        self.gat_resize = torch.nn.Linear(self.hidden_channels, self.hidden_channels * self.message_passers_heads)
        
        self.message_passers_constr()
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.message_passers_heads * self.hidden_channels, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=self.hidden_channels * self.message_passers_heads)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=6)
        
        self.decoder_resize = torch.nn.Linear(self.hidden_channels * self.message_passers_heads, self.out_channels)
        
        self.absolute = absolute
        
    def message_passers_constr(self) :
        self.message_passers = torch.nn.ModuleList()
        for i in range(self.messages):
            self.message_passers.append(GATv2Conv(self.hidden_channels * self.message_passers_heads, self.hidden_channels, heads=self.message_passers_heads, dropout=self.dropout, concat=True, edge_dim=self.edge_dim))
        
    def forward_encode(self, x) :
        #encode
        x_extended = self.encoder_resize(x.reshape(-1, self.in_channels)).reshape(-1, 1, self.hidden_channels)
        encoded = F.gelu(x_extended)
        encoded = self.transformer_encoder(encoded) + x_extended
        
        y = self.gat_resize(encoded).reshape(-1, self.hidden_channels * self.message_passers_heads)
        
        encoder_extended = y.clone().reshape(-1, 1, self.hidden_channels * self.message_passers_heads)
        
        return y, encoder_extended
    
    def forward_message(self, y, edge_index, edge_attr) :
        for i in range(self.messages):
            y = self.message_passers[i](y, edge_index, edge_attr)
            y = F.gelu(y)
        
        return y
    
    def forward_message_and_return_attention(self, y, edge_index, edge_attr) :
        attentions = []
        for i in range(self.messages):
            y, att = self.message_passers[i](y, edge_index, edge_attr, return_attention_weights=True)
            attentions.append(att)
            y = F.gelu(y)
        
        return y, attentions
    
    def forward_decode(self, xshape, y, encoder_extended) :
        y = y.reshape(xshape[0]*xshape[1], 1, self.message_passers_heads* self.hidden_channels)

        # different nodes will be considered as batches
        y = self.transformer_decoder(y, encoder_extended).reshape(xshape[0]*xshape[1], self.message_passers_heads* self.hidden_channels)

        y = self.decoder_resize(y).reshape(xshape[0], xshape[1], self.out_channels)
        
        return y
    
    def forward_postprocess(self, x, y, xshape) :
        if not self.absolute :
            means = y[:,:,:self.out_channels//2]
        else :
            means = y[:,:,:self.out_channels//2] + x.reshape(xshape)[:,:,:self.out_channels//2]
            
        log_std = y[:,:,self.out_channels//2:]
        
        out = torch.cat((means, log_std), dim=2)
        return out
        
    def forward(self, x, edge_index, edge_attr, params, only_mean=False):
        if len(x.shape) == 2 :
            x = x.reshape(1, x.shape[0], x.shape[1])
        #add to x the params 
        xshape = x.shape
        x = torch.cat((x, params.reshape(1, 1, params.shape[0]).repeat(xshape[0], xshape[1], 1)), dim=2)
        
        xshape = x.shape

        #encode
        y, encoder_extended = self.forward_encode(x)

        #message passing
        y = self.forward_message(y, edge_index, edge_attr)
            
        #decode
        y = self.forward_decode(xshape, y, encoder_extended)
        
        #post processing
        out = self.forward_postprocess(x, y, xshape)
        
        if only_mean :
            out = out[:,:,:self.out_channels//2]
        
        return out
    
    def forward_return_attention(self, x, edge_index, edge_attr, params):
        #add to x the params 
        xshape = x.shape
        x = torch.cat((x, params.reshape(1, 1, params.shape[0]).repeat(xshape[0], xshape[1], 1)), dim=2)
        
        xshape = x.shape

        #encode
        y, encoder_extended = self.forward_encode(x)

        #message passing
        y, attentions = self.forward_message_and_return_attention(y, edge_index, edge_attr)
            
        #decode
        y = self.forward_decode(xshape, y, encoder_extended)
        
        #post processing
        out = self.forward_postprocess(x, y, xshape)
        
        return out, attentions
    
    def show_gradients(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad != None :
                    print(name, param.grad.mean())
    
    def loss_direct(self, pred, target, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal', add_back=None):
        #pred is a tensor of shape (T, N, 8)
        #target is a tensor of shape (T, N, 4)
        
        #we will draw a sample from the normal distribution
        #and compute the loss
        std = torch.exp(pred[:,:,self.out_channels//2:])

        return self.diff(pred, std, pred[:,:,self.out_channels//2:], target, loss_history=loss_history, distrib=distrib, add_back=add_back)
    
    def loss_relative_direct(self, pred, target, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal'):
        if target.shape[0] != pred.shape[0] + 1 :
            raise Exception("target.shape[0] != pred.shape[0] + 1")
        
        previous = target[:-1].to(pred.device)
        now = target[1:].to(pred.device)
        
        targ = now - previous
        
        return self.loss_direct(pred, targ, loss_history=loss_history, distrib=distrib, add_back=previous)
    
    def loss_recursive(self, pred_distr, pred, target, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal'):
        std = torch.exp(pred_distr[:,:,self.out_channels//2:])
        
        return self.diff(pred, std, pred_distr[:,:,self.out_channels//2:], target, loss_history=loss_history, distrib=distrib)
    
    def loss_relative_recursive(self, pred_distr, pred, target, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal'):
        pass

    def diff(self, pred, std, log_std, target, distrib='normal', loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, add_back = None) :
        target = target.to(pred.device)

        #see https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf
        #slide 16
        
        std = std + 1e-6
            
        if distrib == 'normal' :
            diff = (pred[:,:,:self.out_channels//2] - target)**2

            if self.wrap :
                #https://www.geogebra.org/m/fvsyepzd
                diff = torch.sin(diff * torch.pi) + torch.square(diff) / 20

            loss_mean = diff / (2 * std**2) 
            loss_log = log_std
        elif distrib == 'laplace' :
            diff = torch.abs((pred[:,:,:self.out_channels//2] - target))

            if self.wrap :
                #https://www.geogebra.org/m/fvsyepzd
                diff = torch.sin(diff * torch.pi) + torch.square(diff) / 20
            
            loss_mean = diff / (std)
            loss_log = log_std
        else :
            raise ValueError('distrib must be normal or laplace')
        
        loss = torch.max(loss_mean + loss_log)
        
        loss_history["loss_mean"].append(loss_mean.mean().item())
        loss_history["loss_log"].append(loss_log.mean().item())
        loss_history["loss"].append(torch.mean(loss_mean + loss_log).item())
        
        if add_back != None :
            pred[:,:,:self.out_channels//2] = pred[:,:,:self.out_channels//2] + add_back

        if self.wrap :
            return loss.requires_grad_(), pred[:,:,:self.out_channels//2] % 1
    
        return loss.requires_grad_(), pred[:,:,:self.out_channels//2]
        
        
    def in_depth_parameters(self, x) :
        
        all_params = {}
        
        x = x[:,:,:4].to(x.device)
        
        speed_diff_x = torch.mean(x[:,:,2], dim=1)
        speed_diff_y = torch.mean(x[:,:,3], dim=1)
        all_params['speed_mu_x'] = speed_diff_x
        all_params['speed_mu_y'] = speed_diff_y
        
        speed_spread_x = torch.log(torch.std(x[:,:,2], dim=1))
        speed_spread_y = torch.log(torch.std(x[:,:,3], dim=1))
        all_params['log(speed_std_x)'] = speed_spread_x
        all_params['log(speed_std_y)'] = speed_spread_y
        
        mean_mu_x =   torch.mean(x[:,:,0], dim=1)
        mean_mu_y =  torch.mean(x[:,:,1], dim=1)
        all_params['Mean_mu_x'] =  mean_mu_x
        all_params['Mean_mu_y'] =  mean_mu_y
        
        std_mu_x =  torch.log(torch.std(x[:,:,0], dim=1))
        std_mu_y =  torch.log(torch.std(x[:,:,1], dim=1))
        all_params['log(std_mu_x)'] =  std_mu_x
        all_params['log(std_mu_y)'] =  std_mu_y

        return all_params
        
    
    def draw(self, pred, distrib='normal') :
        std = torch.exp(pred[:,:,self.out_channels//2:])
        
        if distrib == 'normal' :
            if self.wrap :
                return torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device) % 1
            else : 
                return torch.normal(pred[:,:,:self.out_channels//2], std).to(pred.device)
        elif distrib == 'laplace' :
            if self.wrap :
                return torch.distributions.laplace.Laplace(pred[:,:,:self.out_channels//2], std).sample().to(pred.device) % 1
            else : 
                return torch.distributions.laplace.Laplace(pred[:,:,:self.out_channels//2], std).sample().to(pred.device)
        else :
            raise ValueError('distrib must be normal or laplace')
        
class ConvPredictor(Gatv2Predictor) :
    def message_passers_constr(self):
        self.message_passers = torch.nn.ModuleList()
        for i in range(self.messages) :
            self.message_passers.append(GeneralConv(in_channels=self.hidden_channels * self.message_passers_heads, out_channels=self.hidden_channels* self.message_passers_heads, in_edge_channels=self.edge_dim, heads=self.message_passers_heads,attention=True, attention_type='dot_product', l2_normalize=True))

    
class Gatv2PredictorDiscr(torch.nn.Module) :
    """This class is a simple discriminator that will try to distinguish between real and fake trajectories for the recursive case"""
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super(Gatv2PredictorDiscr, self).__init__()
        
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

class GraphEvolutionDiscr(Gatv2Predictor):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0, edge_dim=1, messages=3, wrap=True) :
        super(GraphEvolutionDiscr, self).__init__(in_channels, out_channels, hidden_channels, dropout=dropout, edge_dim=edge_dim, messages=messages, wrap=wrap)
        self.discr = Gatv2PredictorDiscr(out_channels // 2, hidden_channels, dropout=dropout)
        
    def diff_loss(self, pred, std, log_std, target, all_edges, grad=True):
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
        
        loss_gen = -torch.mean(disc_out) + self.diff(pred,\
                                                     std, log_std,\
                                                     target)[0]
    
        loss = loss_gen + 5 * loss_critic

        return loss.requires_grad_(), pred[:,:,:self.out_channels//2]
    
    def loss_direct(self, pred, target, all_edges, grad=True):
        std = torch.exp(pred[:,:,self.out_channels//2:])
        
        return self.diff_loss(pred, std, pred[:,:,self.out_channels//2:], target, all_edges, grad=grad)
    
    def loss_recursive(self, pred_distr, pred, target, all_edges, grad=True):
        std = torch.exp(pred_distr[:,:,self.out_channels//2:])
        
        return self.diff_loss(pred, std, pred[:,:,self.out_channels//2:], target, all_edges, grad=grad)