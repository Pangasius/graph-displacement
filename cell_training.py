import torch        

import torch.nn.functional as F

from torch_geometric.nn import knn_graph

#this runs a single batch through the model and returns the loss and the output
def run_single(model, data, i, device, denormalize = False) :
    x, edge_index, edge_attr, batch_edges, norm_and_var = data.get(i)

    xshape = x.shape

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    batch_edges = batch_edges.to(device)

    edge_index = edge_index - batch_edges * xshape[1]

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    mask = torch.logical_and(batch_edges > 0, \
                             batch_edges < xshape[0] - 1)
    out = model(x[1:-1], edge_index[:, mask] , edge_attr[mask])

    loss = F.mse_loss(out, x[2:])
    
    #denormalize the data
    if denormalize :
        out_ = out.clone().detach().cpu()
        x_ = x[2:].clone().detach().cpu()
        out_[:,:,:2] = out_[:,:,:2] * norm_and_var[1] + norm_and_var[0]
        x_[:,:,:2] = x_[:,:,:2] * norm_and_var[1] + norm_and_var[0]
        
        return loss, out_, x_
    
    return loss, None, None

#this runs a single batch through the model and returns the loss and the output
#however, it recursively predicts the next step using the previous prediction
def run_single_recursive(model, data, i, device, denormalize = False) :
    x, edge_index, edge_attr, batch_edges, norm_and_var = data.get(i)

    xshape = x.shape

    x = x.to(device)
    edge_index = edge_index
    edge_attr = edge_attr
    batch_edges = batch_edges

    edge_index = edge_index - batch_edges * xshape[1]

    #we don't want to predict the last step since we wouldn't have the data for the loss
    #and for the first point we don't have the velocity
    input_x = x[1].unsqueeze(dim=0).to(device)
    out = torch.tensor([]).to(device)
    mask = batch_edges == 0
    edge_index = edge_index[:, mask].to(device)
    edge_attr = edge_attr[mask].to(device)
    
    #["distance_x", "distance_y", "degree", "velocity_x", "velocity_y", "epsilon", "tau", "v0", \
    #                       "border_xl", "border_xr", "border_yd", "border_yu", \
    #                       "avg_radius", "radius"]
    attributes = data.attributes
    
    border = [edge_attr[0, attributes.index("border_xl")], edge_attr[0, attributes.index("border_xr")], \
              edge_attr[0, attributes.index("border_yd")], edge_attr[0, attributes.index("border_yu")]]
    tau = edge_attr[0, attributes.index("tau")].cpu()
    epsilon = edge_attr[0, attributes.index("epsilon")].cpu()
    v0 = edge_attr[0, attributes.index("v0")].cpu()
    r = edge_attr[0, attributes.index("avg_radius")].cpu()
    
    #radius is a TxN tensor that we can rebuild from the edge_attr but we can use a 1xN tensor
    radius = torch.zeros((xshape[1])).to(device)
    counter = 0
    #using the fact that the tensor is complete
    for j in range (0,edge_attr.shape[0]) :
        if batch_edges[j] == 0 :
            if edge_index[0,j] == counter :
                radius[counter] = edge_attr[j, attributes.index("radius")] + r
                counter = counter + 1
            else :
                continue
        else :
            break
    radius = radius.unsqueeze(0).cpu().numpy()
    
    for current_time in range(1, xshape[0] - 1) :
        out_time = model(input_x.to(device), edge_index.to(device) , edge_attr.to(device))

        input_x = out_time
        
        #from the output we need to rebuild the edge_index and edge_attr
        #since the number of points changes
        edge_index, edge_attr, batch_edge = data.get_edges(input_x.cpu(), torch.zeros(input_x.shape[1]).cpu(), border, tau, epsilon, v0, r, radius)
        
        out = torch.cat((out, out_time), dim=0)
        
    #the loss is only computed for the last step
    loss = F.mse_loss(out_time, x[-1].unsqueeze(0)) #type: ignore
    
    if denormalize :
        out_ = out.clone().detach().cpu()
        x_ = x[2:].clone().detach().cpu()    
        out_[:,:,:2] = out_[:,:,:2] * norm_and_var[1] + norm_and_var[0]
        x_[:,:,:2] = x_[:,:,:2] * norm_and_var[1] + norm_and_var[0]
        
        return loss, out_, x_
    
    return loss, None, None


def test(model, data, device) :
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        loss_sum = 0
        for i in range(data.len()):
            loss, _, _ = run_single(model, data, i, device)
            
            loss_sum = loss_sum + loss.item()
            
        return loss_sum / data.len()
    
def train(model, optimizer, scheduler, data, device, epoch, process) :
    model.train()
    model = model.to(device)
    for i in range(data.len()):
        optimizer.zero_grad()
        
        loss, _, _ = run_single(model, data, i, device)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step(epoch + i / data.len())
        
        print("Current loss : " , loss.item(), " ... ", i, "/", data.len(), ". Current memory usage : ", process.memory_info().rss // 1000000, " MB, loaded ", len(data.memory), "    ", end="\r")  # in megabytes 
        
    return model