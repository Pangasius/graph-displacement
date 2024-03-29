import torch        

from cell_dataset import CellGraphDataset, RealCellGraphDataset
from cell_model import Gatv2Predictor, GraphEvolutionDiscr

import psutil

def iterate(data, x, params, masks, model, duration, device, draw, distrib, many_many):
    out = x[0].detach().clone().unsqueeze(dim=0).to(device)
    predicted_values = torch.tensor([]).to(device)

    _, edge_index, edge_attr = data.get_edges(out[0,:,:2], data.max_degree, wrap=data.wrap, masks=masks)
    
    horizon = model.horizon
        
    for current_time in range(0, duration-1) :
        #time embedding
        recent = out[-horizon:].detach().clone().to(device)
        
        time = torch.arange(1, recent.shape[0] + 1, device=device).unsqueeze(1).unsqueeze(2).repeat(1, recent.shape[1], 1).float() / (recent.shape[0] + 1)
        
        input_model = torch.cat((recent, time), dim=2)
        
        output = model(input_model, edge_index.to(device), edge_attr.to(device), params, many_many=many_many)
        
        if draw : 
            values = model.draw(output, distrib=distrib).to(device) + out[current_time,:,:model.out_channels//2]
        else :
            values = output[:,:,:model.out_channels//2] + out[current_time,:,:model.out_channels//2]
        
        returned, edge_index, edge_attr = data.get_edges(values[:,:,:2], data.max_degree, wrap=data.wrap, previous=out[current_time,:,:2], masks=masks)
        
        if values.shape[0] > 1 and values.shape[0] < duration - 1:
            #very slow thing 
            _, edge_index, edge_attr = data.get_edges(values[-1,:,:2], data.max_degree, wrap=data.wrap, previous=None, masks=masks)
                
        if model.out_channels == 4 :
            values = returned
        else :
            #add back the degree of the node
            values = torch.cat((values, returned[:,:,-1].unsqueeze(2)), dim=2)
        
        out = torch.cat((out, values), dim=0)
        predicted_values = torch.cat((predicted_values, output), dim=0)
          
    return out[1:], predicted_values
    
def compute_parameters(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, duration : int = -1, distrib='normal', many_many = False) :
    
    model.eval()
    all_times_out = []
    all_times_true = []
    
    with torch.no_grad() :

        for i in range(data.len()) :
            if isinstance(data, RealCellGraphDataset) :
                duration, x, edge_index, edge_attr, masks = data.get(i, duration=duration)
                
                params = None
                masks = masks.to(device)
            elif isinstance(data, CellGraphDataset) :
                duration, x, edge_index, edge_attr, border, params = data.get(i, duration=duration)
                
                params = params.to(device)
                
                masks = None
            else :
                raise ValueError("The data type is not supported")

            #we don't want to predict the last step since we wouldn't have the data for the loss
            #and for the first point we don't have the velocity
            out, _ = iterate(data, x, params, masks, model, duration, device, True, distrib, many_many)

            #get the dictionary of parameters
            all_params_out = model.in_depth_parameters(out)
            all_params_true = model.in_depth_parameters(x[1:,:, :4])
            
            #extract everything as item and detach it from the graph
            for key in list(all_params_out) :
                all_params_out[key] = all_params_out[key].detach().cpu()#type: ignore
                
            for key in list(all_params_true) :
                all_params_true[key] = all_params_true[key].detach().cpu()#type: ignore
            
            all_times_out.append(all_params_out)
            all_times_true.append(all_params_true)
            
        return all_times_out, all_times_true
    
def run_single_recursive(model : Gatv2Predictor, data, i : int, device : torch.device, duration : int = -1, loss_history = {"loss_mean" : [], "loss_log" : [], "loss" : []}, distrib='normal', aggr = 'mean', draw=False, many_many = False) :
    if isinstance(data, RealCellGraphDataset) :
        duration, x, edge_index, edge_attr, masks = data.get(i, duration=duration)
        
        params = None
        masks = masks.to(device)
    elif isinstance(data, CellGraphDataset) :
        duration, x, edge_index, edge_attr, border, params = data.get(i, duration=duration)
        
        params = params.to(device)
        
        masks = None
    else :
        raise ValueError("The data type is not supported")
    
    out, diffs = iterate(data, x, params, masks, model, duration, device, draw, distrib, many_many)
    
    if isinstance(model, GraphEvolutionDiscr) :
        raise Exception("Relative loss not implemented for GraphEvolutionDiscr")
    else :
        if isinstance(data, RealCellGraphDataset) :
            loss = model.loss_relative_direct(diffs, x[:,:, :model.out_channels // 2], loss_history=loss_history, distrib=distrib, aggr=aggr, masks=masks, wrapped_columns=[4])
        else :
            loss = model.loss_relative_direct(diffs, x[:,:, :model.out_channels // 2], loss_history=loss_history, distrib=distrib, aggr=aggr, masks=None)
                        
    return loss, out.detach(), x[1:, :, :model.out_channels // 2].detach()


def test_single(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, loss_history : dict[str, list[float]], duration : int = -1, distrib='normal', aggr = 'mean', many_many = False) :
    model.eval()
     
    with torch.no_grad():
        loss_sum = 0
        for i in range(min(data.len(), 50)):
            loss, _, _ = run_single_recursive(model, data, i, device, duration=duration,  loss_history=loss_history, distrib=distrib, aggr=aggr, draw=False, many_many=many_many)
            
            loss_sum = loss_sum + loss.item()
            
        return loss_sum / data.len()      
    
def predict(model : Gatv2Predictor, data : CellGraphDataset, device : torch.device, duration : int = -1, distrib='normal', aggr = 'mean', many_many = False) :
    model.eval()
    
    with torch.no_grad():  
        out = torch.tensor([]).to(device)
        x = torch.tensor([]).to(device)
        
        for i in range(data.len()):

            loss, out_i, x_i = run_single_recursive(model, data, i, device, duration=duration, distrib=distrib, aggr=aggr, draw=True, many_many=many_many)

            if out_i is None or x_i is None :
                continue
                
            out_i = out_i.unsqueeze(dim=0).detach().to(device)
            x_i = x_i.unsqueeze(dim=0).detach().to(device)
            
            out = torch.cat((out, out_i), dim=0)
            x = torch.cat((x, x_i), dim=0)
    
        return out, x
     
    
def train(model : Gatv2Predictor, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler._LRScheduler , data : CellGraphDataset, device : torch.device, epoch : int, process : psutil.Process | None, max_epoch : int, distrib='normal', aggr = 'mean', many_many = False) :
    model.train()
    
    loss_all = torch.zeros((data.len()))
    
    for i in range(data.len()):
        
        optimizer.zero_grad()

        #duration is a random number between 2 and the maximum duration
        if isinstance(data, RealCellGraphDataset) :
            duration =  min(epoch//32+2,6)
        elif isinstance(data, CellGraphDataset) :
            duration =  min(epoch + 2, 8)
        else :
            raise ValueError("The data type is not supported")

        loss, _, _ = run_single_recursive(model, data, i, device, duration=duration, distrib=distrib, aggr=aggr, many_many=many_many) 

        if process != None :
            print("Current loss : {:.2f}, ... {}, / {}, Current memory usage : {} MB, loaded {}    ".format(loss.item(), i, data.len(), process.memory_info().rss // 1000000, len(data.memory)), end="\r")  # in megabytes
        
        loss.backward()
        
        #Prints the gradient norm for debugging
        #print("Gradient norm : {}\n".format(torch.norm(model.decoder_resize.weight.grad)), end="\n")
        #raise Exception("Stop here")
        
        optimizer.step()

        scheduler.step((epoch + i) / data.len()) #type: ignore
        
        loss_all[i] = loss.item()
        
    return loss_all.tolist()