import torch        

import torch.nn.functional as F

from cell_dataset import CellGraphDataset
from cell_model import GraphEvolution

import psutil

#this runs a single batch through the model and returns the loss and the output
def run_single(model : GraphEvolution, data , i : int, device : torch.device, duration : int = -1, output = False) :
    x, edge_index, edge_attr, border, params, cutoff = data.get(i)

    xshape = x.shape
    
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    x = x.to(device)
    params = params.to(device)

    if duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    mask = torch.logical_and(torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') > 0, \
                             torch.div(edge_index[0, :], xshape[1], rounding_mode='floor') < duration + 1)
    
    edge_index = (edge_index[:, mask] - xshape[1])
    
    out = model(x[1:duration+1], edge_index, edge_attr[mask], params)
    
    loss, out = model.loss_direct(out, x[2:duration + 2,:, :4])
    
    if output :
        factor = torch.stack([border[1] - border[0], border[3] - border[2]]).repeat(2).to(device)
        min_ = torch.cat((torch.stack([border[0], border[2]]), torch.zeros(2))).to(device)

        out = (out * factor + min_).detach().cpu()
        x = (x[2:duration + 2,:,:4] * factor + min_).detach().cpu()
        
        return loss, out, x
    else :
        return loss, None, None

#this runs a single batch through the model and returns the loss and the output
#however, it recursively predicts the next step using the previous prediction
def run_single_recursive(model : GraphEvolution, data, i : int, device : torch.device,  duration : int = -1, output=False) :
    x, edge_index, edge_attr, border, params, cutoff = data.get(i)

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
        
        out = torch.cat((out, input_x), dim=0)
        
        #skip the following if on the last step
        if current_time == duration :
            break
        
        #from the output we need to rebuild the edge_index and edge_attr
        #since the number of points changes
        sample = model.draw(input_x.cpu())
        input_x, edge_index, edge_attr = data.get_edges(sample[:,:,:2].cpu(), data.max_degree, wrap=True, T=1, N=xshape[1], cutoff=cutoff)
        
        input_x[:,:,2:4] = sample[:,:,2:4]
        
    #the loss
    loss, out = model.loss_direct(out, x[2:duration + 2,:, :4])
    
    if output :
        factor = torch.stack([border[1] - border[0], border[3] - border[2]]).repeat(2).to(device)
        min_ = torch.cat((torch.stack([border[0], border[2]]), torch.zeros(2))).to(device)
    
        out = (out.to(device) * factor + min_).detach().cpu()
        x = (x[2:duration + 2,:,:4].to(device) * factor + min_).detach().cpu()
        
        return loss, out, x
    else :
        return loss, None, None


def test(model : GraphEvolution, data : CellGraphDataset, device : torch.device, method = run_single, duration : int = -1) :
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        loss_sum = 0
        for i in range(data.len()):
            loss, _, _ = method(model, data, i, device, duration=duration)
            
            loss_sum = loss_sum + loss.item()
            
        return loss_sum / data.len()
    
def evaluate(model : GraphEvolution, data : CellGraphDataset, device : torch.device, method = run_single, duration : int = -1) :
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        all_losses = []
        similarity = []
        for i in range(data.len()):
            loss, out, x = method(model, data, i, device, duration=duration)
            
            inertia_prediction = torch.cat((out[:,:,:2], out[:,:,:2] + out[:,:,2:]) , dim=2) #type: ignore
            inertia_actual = torch.cat((x[:,:,:2], x[:,:,:2] + x[:,:,2:]), dim=2) #type: ignore
            
            similarity.append(F.cosine_similarity(inertia_prediction, inertia_actual, dim=2).mean().item())
            
            all_losses.append(loss.item())
            
        #compute a few metrics
        all_losses = torch.tensor(all_losses)
        mean = all_losses.mean()
        std = all_losses.std()
        
        similarity = torch.tensor(similarity)
        mean_similarity = similarity.mean()
        std_similarity = similarity.std()
        
        return mean, std, mean_similarity, std_similarity
    
    
def train(model : GraphEvolution, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler._LRScheduler , data : CellGraphDataset, device : torch.device, epoch : int, process : psutil.Process, max_epoch : int, recursive = False) :
    model.train()
    model = model.to(device)
    
    probability = torch.sigmoid(torch.tensor((2 * epoch - 2 * max_epoch - 70) / max_epoch))
    distribution = torch.distributions.Bernoulli(torch.tensor([probability]))
    
    if epoch % 10 == 0:
        print("Current probability of recursive training : ", 0 if not recursive else probability)
    
    for i in range(data.len()):
        optimizer.zero_grad()
        
        condition = distribution.sample().item() and recursive
        
        if condition :
            duration = int(torch.randint(low=1, high=int(torch.ceil(probability * 10).item()), size = (1,)).item())
            method = run_single_recursive
        else :
            duration = -1
            method = run_single

        loss, _, _ = method(model, data, i, device, duration=duration) 
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step((epoch + i) / data.len()) #type: ignore
        
        print("Current loss : " , loss.item(), " ... ", i, "/", data.len(), ". Current memory usage : ", process.memory_info().rss // 1000000, " MB, loaded ", len(data.memory), "    ", end="\r")  # in megabytes 
        
    return model