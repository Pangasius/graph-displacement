import torch        

import torch.nn.functional as F

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
    out_ = out.clone().detach()
    x_ = x.clone().detach()
    if denormalize :
        out_[:,:,:2] = out_[:,:,:2] * norm_and_var[1] + norm_and_var[0]
        x_[:,:,:2] = x_[:,:,:2] * norm_and_var[1] + norm_and_var[0]

    return loss, out_.cpu(), x_.cpu()

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