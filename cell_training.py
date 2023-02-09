import torch        

import torch.nn.functional as F

from cell_dataset import CellGraphDataset
from cell_model import GraphEvolution

import psutil

#this runs a single batch through the model and returns the loss and the output
def run_single(model : GraphEvolution, data : CellGraphDataset , i : int, device : torch.device, duration : int | None = None) :
    x, edge_index, edge_attr, batch_edges, border, params = data.get(i)

    xshape = x.shape

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    batch_edges = batch_edges.to(device)
    
    if duration is None or duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    mask = torch.logical_and(batch_edges > 0, \
                             batch_edges < duration + 1)
    
    out = model(x[1:duration+1], edge_index[:, mask] - xshape[1], edge_attr[mask], params)
    
    factor = torch.stack([border[1] - border[0], border[3] - border[2]]).repeat(2).to(device)
    min_ = torch.cat((torch.stack([border[0], border[2]]), torch.zeros(2))).to(device)

    out = out * factor + min_
    x = x * factor + min_

    loss = F.mse_loss(out, x[2:duration + 2])
        
    return loss, out.detach().cpu(), x[2:duration + 2].detach().cpu()

#this runs a single batch through the model and returns the loss and the output
#however, it recursively predicts the next step using the previous prediction
def run_single_recursive(model : GraphEvolution, data : CellGraphDataset , i : int, device : torch.device, duration : int | None  = None) :
    x, edge_index, edge_attr, batch_edges, border, params = data.get(i)

    xshape = x.shape

    x = x.to(device)

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    input_x = x[1].unsqueeze(dim=0).to(device)
    out = torch.tensor([]).to(device)
    mask = batch_edges == 0
    edge_index = edge_index[:, mask].to(device)
    edge_attr = edge_attr[mask].to(device)
    
    attributes = data.attributes

    r = edge_attr[0, attributes.index("avg_radius")].cpu().item()
    
    #radius is a TxN tensor that we can rebuild from the edge_attr but we can use a 1xN tensor
    radius = torch.zeros((xshape[1])).to(device)
    counter = 0
    #using the fact that the tensor is complete
    for j in range (0,edge_attr.shape[0]) :
        if edge_index[0,j] == counter :
            radius[counter] = edge_attr[j, params[3].long()] * r
            counter = counter + 1
        else :
            continue

    radius = (radius.unsqueeze(0).cpu()).numpy()
    
    if duration is None or duration > xshape[0] - 2 or duration < 1 :
        duration = xshape[0] - 2
    
    for current_time in range(1, 1 + duration) :
        input_x = model(input_x.to(device), edge_index.to(device), edge_attr.to(device), params.to(device))
        
        out = torch.cat((out, input_x), dim=0)
        
        #skip the following if on the last step
        if current_time == duration :
            break
        
        #from the output we need to rebuild the edge_index and edge_attr
        #since the number of points changes
        next_val, edge_index, edge_attr = data.get_edges(input_x[:,:,:2].cpu(), data.max_degree, wrap=True, T=1, N=xshape[1])
        edge_attr, batch_edge = data.get_edges_attributes(edge_index, edge_attr, data.max_degree, r, radius, T=1, N=xshape[1])
        
        #get_edges_attributes will return the actual velocities and we do not want them
        input_x[:,:,:2] = next_val[:,:,:2]
        
    factor = torch.stack([border[1] - border[0], border[3] - border[2]]).repeat(2).to(device)
    min_ = torch.cat((torch.stack([border[0], border[2]]), torch.zeros(2))).to(device)

    out = out * factor + min_
    x = x * factor + min_
        
    #the loss
    loss = F.mse_loss(out, x[2:duration + 2])
    
    return loss, out.detach().cpu(), x[2:duration + 2].detach().cpu()


def test(model : GraphEvolution, data : CellGraphDataset, device : torch.device, method = run_single, duration : int | None = None) :
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        loss_sum = 0
        for i in range(data.len()):
            loss, _, _ = method(model, data, i, device, duration=duration)
            
            loss_sum = loss_sum + loss.item()
            
        return loss_sum / data.len()
    
def evaluate(model : GraphEvolution, data : CellGraphDataset, device : torch.device, method = run_single, duration : int | None = None) :
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        all_losses = []
        similarity = []
        for i in range(data.len()):
            loss, out, x = method(model, data, i, device, duration=duration)
            
            inertia_prediction = torch.cat((out[:,:,:2], out[:,:,:2] + out[:,:,2:]) , dim=2)
            inertia_actual = torch.cat((x[:,:,:2], x[:,:,:2] + x[:,:,2:]), dim=2)
            
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
            duration = None
            method = run_single

        loss, _, _ = method(model, data, i, device, duration=duration) 
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step((epoch + i) / data.len()) #type: ignore
        
        print("Current loss : " , loss.item(), " ... ", i, "/", data.len(), ". Current memory usage : ", process.memory_info().rss // 1000000, " MB, loaded ", len(data.memory), "    ", end="\r")  # in megabytes 
        
    return model