from torch_geometric.nn import GATv2Conv, GeneralConv
import torch.nn.functional as F

import torch

class Gatv2Predictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.1, edge_dim=1, messages=3, wrap=True, absolute=0, heads=8, horizon=1):
        super().__init__()
        
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_passers_heads = heads
        
        self.hidden_channels = hidden_channels
        
        self.wrap = wrap
        
        self.messages = messages
        
        self.dropout = dropout
        
        self.horizon = horizon
        
        #expand input so encoder is the size of the hidden channels
        self.encoder_resize = torch.nn.Linear(self.in_channels, self.hidden_channels)
        
        #needed because norm first is set to true in the transformer encoder
        self.norm_encoder = torch.nn.LayerNorm(self.hidden_channels)
        
        #we need an encoder to encode the messages before sending them
        encode_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=False, dim_feedforward=self.hidden_channels, activation=torch.nn.functional.leaky_relu, norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encode_layer, num_layers=6, enable_nested_tensor=True)
    
        self.message_passers_constr()
        
        #needed because norm first is set to true in the transformer encoder
        self.norm_decoder = torch.nn.LayerNorm(self.hidden_channels)
        
        #we take the output and convert it to the desired output
        decode_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=4, dropout=dropout, batch_first=False, dim_feedforward=self.hidden_channels, activation=torch.nn.functional.leaky_relu, norm_first=True)
        self.transformer_decoder = torch.nn.TransformerDecoder(decode_layer, num_layers=6)
        
        self.decoder_resize = torch.nn.Linear(self.hidden_channels, self.out_channels)
        
    def message_passers_constr(self) :
        self.message_passers = torch.nn.ModuleList()
        for i in range(self.messages):
            self.message_passers.append(GATv2Conv(self.hidden_channels, self.hidden_channels, heads=self.message_passers_heads, dropout=self.dropout, concat=False, edge_dim=self.edge_dim, add_self_loops=False, fill_value=0.0))
        
    def forward_encode(self, x, many_many=False) :
        #encode
        x = self.encoder_resize(x.view(-1, self.in_channels)).view(self.horizon, -1, self.hidden_channels)
        
        x = self.transformer_encoder(x)
        
        if many_many == False:
            x = x.mean(dim=0, keepdim=True)
        
        x = self.norm_encoder(x)
        
        return x
    
    def forward_message(self, x, edge_index, edge_attr) :
        to_return = torch.tensor([], device=x.device)
        
        for i in range(x.shape[0]) :
            y = x[i].clone()
            for j in range(self.messages):
                y = y + F.leaky_relu(self.message_passers[j](y, edge_index, edge_attr))
                
            to_return = torch.cat((to_return, y.unsqueeze(0)), dim=0)
        
        return to_return
    
    def forward_decode(self, xshape, y, encoder_extended) :
        y = y.view(-1, xshape[1], self.hidden_channels)
        
        # different nodes will be considered as batches
        y = self.transformer_decoder(y, encoder_extended)
        
        y = self.norm_decoder(y)

        y = self.decoder_resize(y).view(-1, xshape[1], self.out_channels)
        
        return y
        
    def forward(self, x, edge_index, edge_attr, params=None, only_mean=False, many_many=False):
        if len(x.shape) == 2 :
            x = x.unsqueeze(0)
        
        if self.horizon != x.shape[0]:
            #pad the missing time until horizon
            x = torch.cat((torch.zeros((self.horizon - x.shape[0], x.shape[1], x.shape[2]), device=x.device), x), dim=0)
        
        #add to x the params 
        xshape = x.shape
        
        if params is not None :
            x = torch.cat((x, params.reshape(1, 1, params.shape[0]).repeat(xshape[0], xshape[1], 1)), dim=2)

            xshape = x.shape

        #encode
        encoded = self.forward_encode(x, many_many=many_many)

        #message passing
        y = self.forward_message(encoded, edge_index, edge_attr)
            
        #decode
        y = self.forward_decode(xshape, y, encoded)
        
        if only_mean :
            y = y[:,:,:self.out_channels//2]
        
        return y

    def show_gradients(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad != None :
                    print(name, param.grad.mean())

    def loss_relative_direct(self, pred, target, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal', aggr = 'mean', masks=None, wrapped_columns=[]) :
        
        if target.shape[0] != pred.shape[0] + 1 :
            print(target.shape[0], pred.shape[0] + 1)
            raise Exception("target.shape[0] != pred.shape[0] + 1")
        
        previous = target[:-1].to(pred.device)
        now = target[1:].to(pred.device)
        
        targ = now - previous
        
        """
        #sort the masks and target in the same order
        perms = torch.argsort(targ, dim=1)
        targ = torch.gather(targ, 1, perms)
        
        if masks is not None :
            masks = torch.gather(masks[1:], 1, perms[:,:,0].unsqueeze(2))
             
        pred = torch.sort(pred, dim=1)[0]
        """
        
        if masks is not None :
            masks = masks[1:].to(pred.device)
        
        log_std = pred[:,:,self.out_channels//2:]
        mu = pred[:,:,:self.out_channels//2]
        
        std = torch.exp(log_std)
    
        return self.diff(mu, std, log_std, targ, loss_history=loss_history, distrib=distrib, aggr=aggr, masks=masks)
    
    def diff(self, mu, std, log_std, target, distrib='normal', loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, aggr = 'mean', masks=None, wrap_columns=[]) :
        target = target.to(mu.device)

        #see https://glouppe.github.io/info8010-deep-learning/pdf/lec10.pdf
        #slide 16
        
        not_wrap_columns = [i for i in range(self.out_channels//2) if i not in wrap_columns]
        
        if distrib == 'normal' :
            diff_not_wrapped = (mu[:, :, not_wrap_columns] - target[:, :, not_wrap_columns]) ** 2
            
            wrapped_diff = torch.abs((mu[:, :, wrap_columns] - target[:, :, wrap_columns]))
            diff_wrapped = torch.sin(wrapped_diff) + 0.01 * wrapped_diff
            
            diff = torch.cat((diff_not_wrapped, diff_wrapped), dim=2)

            if self.wrap :
                diff = torch.sin(diff * torch.pi)

            loss_mean = diff / (2 * std**2) 
            loss_log = log_std
            
        elif distrib == 'laplace' :
            diff_not_wrapped = torch.abs((mu[:, :, not_wrap_columns] - target[:, :, not_wrap_columns]))
            
            wrapped_diff = torch.abs((mu[:, :, wrap_columns] - target[:, :, wrap_columns]))
            diff_wrapped = torch.sin(wrapped_diff) + 0.01 * wrapped_diff
            
            diff = torch.cat((diff_not_wrapped, diff_wrapped), dim=2)

            if self.wrap :
                diff = torch.sin(diff * torch.pi) 
            
            loss_mean = diff / (std)
            loss_log = log_std
        else :
            raise ValueError('distrib must be normal or laplace')

        if masks is not None :
            diff = diff * masks
            loss_log = loss_log * masks
            loss_mean = loss_mean * masks
        
        if aggr == 'mean' :
            loss = torch.mean(loss_mean + loss_log)
        else :
            raise ValueError('aggr must be mean or max')
        
        loss_history["loss_mean"].append(diff.mean().item())
        loss_history["loss_log"].append(loss_log.mean().item())
        loss_history["loss"].append(torch.mean(loss_mean + loss_log).item())
    
        return loss.requires_grad_()
        
        
    def in_depth_parameters(self, x) :
        
        all_params = {}
        
        x = x[:,:,:2].to(x.device)
        
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