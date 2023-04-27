import torch        

from cell_dataset import CellGraphDataset
from cell_model import Gatv2Predictor, GraphEvolutionDiscr

import psutil

def denorm(x, out, duration, border, device) :
    factor = torch.tensor([border[1] - border[0], border[3] - border[2]]).to(device)
    min_ = torch.tensor([border[0], border[2]]).to(device)
    
    if out.shape[-1] == 4 :
        out = torch.cat((out[:, :, :2] * factor + min_, out[:, :, 2:] * factor), dim=-1)
        x = torch.cat((x[2:duration + 2,:,:2].to(device) * factor + min_, x[2:duration + 2,:, 2:] * factor), dim=-1)

    else :
        out = (out * factor + min_)
        x = (x[2:duration + 2,:,:2].to(device) * factor + min_)
    
    return x.detach().cpu(), out.detach().cpu()
    
def compute_parameters_draw(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, duration : int = -1, distrib='normal') :
    
    model.eval()
    all_times_out = []
    all_times_true = []
    
    with torch.no_grad() :

        for i in range(data.len()) :
            x, edge_index, edge_attr, border, params = data.get(i)
    
            xshape = x.shape
            
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            params = params.to(device)

            #we don't want to predict the last step since we wouldn't have the data for the loss
            #and for the first point we don't have the velocity
            input_x = x[1].unsqueeze(dim=0).to(device)
            out = torch.tensor([]).to(device)
            mask = torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') == 0
            edge_index = edge_index[:, mask].to(device)
            edge_attr = edge_attr[mask].to(device)

            if duration > xshape[0] - 2 or duration < 1 :
                duration = xshape[0] - 2
                
            for current_time in range(1, 1 + duration) : 
                input_x = model(input_x.to(device), edge_index.to(device), edge_attr.to(device), params)
                #input_x_clone = input_x.clone()
                
                #from the output we need to rebuild the edge_index and edge_attr
                #since the number of points changes
                input_x_drawn = model.draw(input_x[:,:,:model.out_channels], distrib=distrib)
                
                out = torch.cat((out, input_x_drawn.clone()), dim=0)
                
                input_x, edge_index, edge_attr = data.get_edges(input_x_drawn[:,:,:2].cpu(), data.max_degree, wrap=data.wrap, T=1, N=xshape[1])
                input_x[:,:,:model.out_channels // 2] = input_x_drawn[:,:,:model.out_channels // 2]
            
            #get the dictionary of parameters
            all_params_out = model.in_depth_parameters(out)
            all_params_true = model.in_depth_parameters(x[2:duration+2,:, :4])
            
            #extract everything as item and detach it from the graph
            for key in list(all_params_out) :
                all_params_out[key] = all_params_out[key].detach().cpu()#type: ignore
                
            for key in list(all_params_true) :
                all_params_true[key] = all_params_true[key].detach().cpu()#type: ignore
            
            all_times_out.append(all_params_out)
            all_times_true.append(all_params_true)
            
        return all_times_out, all_times_true
        
def compute_parameters(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, duration : int = -1, distrib='normal', relative=False) :
    
    model.eval()
    all_times_out = []
    all_times_true = []
    
    with torch.no_grad() :

        for i in range(data.len()) :
            x, edge_index, edge_attr, border, params = data.get(i)
    
            xshape = x.shape
            
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            params = params.to(device)

            #we don't want to predict the last step since we wouldn't have the data for the loss
            #and for the first point we don't have the velocity
            input_x = x[1].unsqueeze(dim=0).to(device)
            out = torch.tensor([]).to(device)
            mask = torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') == 0
            edge_index = edge_index[:, mask].to(device)
            edge_attr = edge_attr[mask].to(device)

            if duration > xshape[0] - 2 or duration < 1 :
                duration = xshape[0] - 2
                
            for current_time in range(1, 1 + duration) : 
                input_x_before = input_x.clone()
                input_x = model(input_x.to(device), edge_index.to(device), edge_attr.to(device), params)
                input_x_clone = input_x.clone()
                
                if relative :
                    input_x_clone[:,:,:model.out_channels // 2] = input_x[:,:,:model.out_channels // 2] + input_x_before[:, :, :model.out_channels // 2].to(device)
                
                out = torch.cat((out, input_x_clone.clone()), dim=0)
                

                input_x, edge_index, edge_attr = data.get_edges(input_x_clone[:,:,:2].cpu(), data.max_degree, wrap=data.wrap, T=1, N=xshape[1])
                input_x[:,:,:model.out_channels // 2] = input_x_clone[:,:,:model.out_channels // 2]
            
            #get the dictionary of parameters
            all_params_out = model.in_depth_parameters(out)
            all_params_true = model.in_depth_parameters(x[2:duration+2,:, :4])
            
            #extract everything as item and detach it from the graph
            for key in list(all_params_out) :
                all_params_out[key] = all_params_out[key].detach().cpu()#type: ignore
                
            for key in list(all_params_true) :
                all_params_true[key] = all_params_true[key].detach().cpu()#type: ignore
            
            all_times_out.append(all_params_out)
            all_times_true.append(all_params_true)
            
        return all_times_out, all_times_true

#this runs a single batch through the model and returns the loss and the output
def run_single(model : Gatv2Predictor, data , i : int, device : torch.device, duration : int = -1, output = False, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal') :
    x, edge_index, edge_attr, border, params = data.get(i)

    xshape = x.shape

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    params = params.to(device)
    x = x.to(device)
    
    if duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2
        start_time = 0
    else : 
        start_time = int(torch.randint(0, xshape[0] - 2 - duration, (1,)).item())

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    mask = torch.logical_and(torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') > start_time, \
                            torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') < start_time + duration + 1)
    
    edge_index = (edge_index[:, mask] - xshape[1] * (start_time + 1)).to(device)
    
    out = model(x[start_time + 1:start_time + duration + 1], edge_index, edge_attr[mask], params)
    
    loss, out = model.loss_direct(out, x[start_time + 2:start_time + duration + 2,:, :model.out_channels // 2], loss_history = loss_history, distrib=distrib)
    
    if output :
        if data.wrap :
            x, out = denorm(x, out, duration, border, device)
        
        return loss, out, x
    else :
        return loss, None, None

#this runs a single batch through the model and returns the loss and the output
#however, it recursively predicts the next step using the previous prediction
def run_single_recursive_draw(model : Gatv2Predictor, data, i : int, device : torch.device, duration : int = -1, output=False, grad=True, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal') :
    x, edge_index, edge_attr, border, params = data.get(i)

    xshape = x.shape

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    params = params.to(device)
    
    if duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2
        start_time = 0
    else : 
        start_time = int(torch.randint(0, xshape[0] - 2 - duration, (1,)).item())

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    input_x = x[start_time + 1].unsqueeze(dim=0).to(device)
    out = torch.tensor([]).to(device)
    out_pred = torch.tensor([]).to(device)
    mask = torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') == start_time + 1

    edge_index = edge_index[:, mask].to(device) - xshape[1] * (start_time + 1)
    edge_attr = edge_attr[mask].to(device)
    
    if isinstance(model, GraphEvolutionDiscr) :
        all_edges = edge_index.clone()
    else : 
        all_edges = None

    for current_time in range(start_time + 1, start_time + 1 + duration) :
        input_x = model(input_x.to(device), edge_index.to(device), edge_attr.to(device), params)
        #input_x_clone = input_x.clone()
        
        #from the output we need to rebuild the edge_index and edge_attr
        #since the number of points changes
        input_x_drawn = model.draw(input_x[:,:,:model.out_channels], distrib=distrib)
        
        out = torch.cat((out, input_x_drawn.clone()), dim=0)
        out_pred = torch.cat((out_pred, input_x.clone()), dim=0)
        
        input_x, edge_index, edge_attr = data.get_edges(input_x_drawn[:,:,:2].cpu(), data.max_degree, wrap=data.wrap, T=1, N=xshape[1])
        input_x[:,:,:model.out_channels // 2] = input_x_drawn[:,:,:model.out_channels // 2]
        
    if isinstance(model, GraphEvolutionDiscr) :
        loss, out = model.loss_recursive(out_pred[:,:,:model.out_channels], out[:,:,:model.out_channels // 2], x[start_time + 2:start_time + duration + 2,:, :model.out_channels // 2], all_edges=all_edges, grad=grad)
    else :
        loss, out = model.loss_recursive(out_pred[:,:,:model.out_channels], out[:,:,:model.out_channels // 2], x[start_time + 2:start_time + duration + 2,:, :model.out_channels // 2], loss_history=loss_history, distrib=distrib)
    
    if output :
        if data.wrap :
            x, out = denorm(x, out, duration, border, device)
        
        return loss, out, x
    else :
        return loss, None, None
    
def run_single_recursive(model : Gatv2Predictor, data, i : int, device : torch.device, duration : int = -1, output=False, grad=True, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal', relative=False) :
    x, edge_index, edge_attr, border, params = data.get(i)

    xshape = x.shape

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    params = params.to(device)
    
    if duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2
        start_time = 0
    else : 
        start_time = int(torch.randint(0, xshape[0] - 2 - duration, (1,)).item())

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    input_x = x[start_time + 1].unsqueeze(dim=0).to(device)
    out = torch.tensor([]).to(device)
    mask = torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') == start_time + 1

    edge_index = edge_index[:, mask].to(device) - xshape[1] * (start_time + 1)
    edge_attr = edge_attr[mask].to(device)
    
    if isinstance(model, GraphEvolutionDiscr) :
        all_edges = edge_index.clone()
    else : 
        all_edges = None

    for current_time in range(start_time + 1, start_time + 1 + duration) :
        input_x_before = input_x.clone()
        input_x = model(input_x.to(device), edge_index.to(device), edge_attr.to(device), params)
        input_x_clone = input_x.clone()
        
        out = torch.cat((out, input_x.clone()), dim=0)
        
        if relative :
            input_x_clone[:,:,:model.out_channels // 2] = input_x[:,:,:model.out_channels // 2] + input_x_before[:, :, :model.out_channels // 2].to(device)
        

        input_x, edge_index, edge_attr = data.get_edges(input_x_clone[:,:,:2].cpu(), data.max_degree, wrap=data.wrap, T=1, N=xshape[1])
        input_x[:,:,:model.out_channels // 2] = input_x_clone[:,:,:model.out_channels // 2]
        

    if not relative :
        if isinstance(model, GraphEvolutionDiscr) :
            loss, out = model.loss_direct(out[:,:,:model.out_channels], x[start_time + 2:start_time + duration + 2,:, :model.out_channels // 2], all_edges=all_edges, grad=grad)
        else :
            loss, out = model.loss_direct(out[:,:,:model.out_channels], x[start_time + 2:start_time + duration + 2,:, :model.out_channels // 2], loss_history=loss_history, distrib=distrib)
    else : 
        if isinstance(model, GraphEvolutionDiscr) :
            raise Exception("Relative loss not implemented for GraphEvolutionDiscr")
        else :
            loss, out = model.loss_relative_direct(out[:,:,:model.out_channels], x[start_time + 2 - 1:start_time + duration + 2,:, :model.out_channels // 2], loss_history=loss_history, distrib=distrib)
            
    if output :
        if data.wrap :
            x, out = denorm(x, out, duration, border, device)
        
        return loss, out, x
    else :
        return loss, None, None    


def test_single(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, loss_history : dict[str, list[float]], duration : int = -1, recursive = False, distrib='normal', relative=False) :
    model.eval()
     
    with torch.no_grad():
        loss_sum = 0
        for i in range(min(data.len(), 50)):
            if not recursive :
                loss, _, _ = run_single(model, data, i, device, duration=duration, loss_history=loss_history)
            else :
                loss, _, _ = run_single_recursive(model, data, i, device, duration=duration, grad=False, loss_history=loss_history, distrib=distrib, relative=relative)
            
            loss_sum = loss_sum + loss.item()
            
        return loss_sum / data.len()      
    
def train(model : Gatv2Predictor, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler._LRScheduler , data : CellGraphDataset, device : torch.device, epoch : int, process : psutil.Process | None, max_epoch : int, recursive = False, distrib='normal', relative=False) :
    model.train()
    
    for i in range(data.len()):
        
        optimizer.zero_grad()

        if isinstance(model, GraphEvolutionDiscr) or recursive :
            duration = epoch + 1
            loss, _, _ = run_single_recursive(model, data, i, device, duration=duration, grad=True, distrib=distrib, relative=relative) 

        else :
            duration = 25
            loss, _, _ = run_single(model, data, i, device, duration=duration, distrib=distrib) 

        if process != None :
            print("Current loss : {:.2f}, ... {}, / {}, Current memory usage : {} MB, loaded {}    ".format(loss.item(), i, data.len(), process.memory_info().rss // 1000000, len(data.memory)), end="\r")  # in megabytes

        loss.backward()
        
        #Prints the gradient norm for debugging
        #print("Gradient norm : {}\n".format(torch.norm(model.decoder_resize.weight.grad)), end="\n")
        #raise Exception("Stop here")
        
        optimizer.step()

        scheduler.step((epoch + i) / data.len()) #type: ignore