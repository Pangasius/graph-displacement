from genericpath import exists
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pickle


import os

from cell_dataset import CellGraphDataset, loadDataset
from cell_model import Gatv2Predictor
from cell_utils import GraphingLoss
from cell_training import train, test_single, compute_parameters, predict

from cell_dataset import loadDataset

import os, psutil
process = psutil.Process(os.getpid())

#https://github.com/clovaai/AdamP
from adamp import AdamP

from ss_stats import create_stats
    
import argparse
    
def run(load_all, pre_separated, override, extension, number_of_messages, size_of_messages, epochs, distrib, out, horizon, validation = False, many_many = False) :
    
    aggr = "mean"

    name_complete = extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + distrib + "_" + str(out) + "_h" + str(horizon)

    print(name_complete)
    
    path_name = "models/new_model/out_" + str(out) + "_eps_-4/" + distrib + "/" + extension[6:] + "/h" + str(horizon) + "/"
    
    #make the directory if it does not exist
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    model_path = path_name + "model" + name_complete
    
    print(model_path)

    data_train, data_test, data_val = loadDataset(load_all, extension, pre_separated, override)
    
    if validation :
        print("\n VALIDATION MODE \n")
        data_test = data_val

    print("\nData loaded\n")

    def start(model : Gatv2Predictor, optimizer : torch.optim.Optimizer, scheduler  : torch.optim.lr_scheduler._LRScheduler,\
          data_train : CellGraphDataset, data_test : CellGraphDataset, device : torch.device, epoch : int, offset : int, grapher : GraphingLoss, save=0):
    
        loss_history_test_recursive = {"loss_mean" : [], "loss_log" : [], "loss" : []}
        loss_history_train = {"loss" : []}
        for e in range(offset, offset + epoch):

            train_loss = train(model, optimizer, scheduler, data_train, device, e, process, max_epoch=offset+epoch, distrib=distrib, aggr=aggr, many_many=many_many)

            #model.show_gradients()
            
            loss_history_train["loss"] += train_loss
            grapher.plot_losses(title="Training", data=loss_history_train, length=len(data_train), extension=name_complete + "_")

            test_loss_r = test_single(model, data_test, device, loss_history_test_recursive, duration=0, distrib=distrib, aggr=aggr, many_many=many_many)

            print("Epoch : ", e, "Test loss recursive : ", test_loss_r)

            grapher.plot_losses(title="Testing recursive", data=loss_history_test_recursive, length=min(50, len(data_test)), extension=name_complete + "_") 
            
            if (e != 0 and e%save == 0) :
                all_params_out, all_params_true = compute_parameters(model, data_test, device, duration=-1, distrib=distrib, many_many=many_many)
                grapher.plot_params(all_params_out, all_params_true, e, extension=name_complete)
            
            if (save and (e%save == 0 or e == epoch-1)) :
                torch.save(model.state_dict(), model_path + str(e) + ".pt")
                
        if epoch > 0 :
            #save the losses
            with open(path_name + "losses" + name_complete + ".pkl", "wb") as f :
                pickle.dump(loss_history_test_recursive, f)
                pickle.dump(loss_history_train, f)
                
    model = Gatv2Predictor(in_channels=13, out_channels=out, hidden_channels=size_of_messages, dropout=0.05, edge_dim=2, messages=number_of_messages, wrap=data_train.wrap, absolute=0, horizon=horizon)
    
    #Load model
    """
    epoch_to_load = 50
    print("Loading model : ", model_path + str(epoch_to_load) + ".pt")
    
    if exists(model_path + str(epoch_to_load) + ".pt") :
        model.load_state_dict(torch.load(model_path + str(epoch_to_load) + ".pt"))
        print("Loaded model")
    """
    
    #might want to investigate AdamP 
    optimizer = AdamP(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-4, weight_decay=5e-3, delta=0.1, wd_ratio=0.1, nesterov=True)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grapher = GraphingLoss()

    model = model.to(device)

    start(model, optimizer, scheduler, data_train, data_test, device, \
            epochs, 0, grapher=grapher, save=50)

    print("\nFinished training\n")

    data_y, data_x = predict(model, data_test, device, -1, distrib=distrib, aggr=aggr, many_many=many_many)

    create_stats(data_y, data_x, path_name=path_name, name_complete=name_complete, extension=extension)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--load_all', type=bool, default=True, help='load directly from a pickle')
    parser.add_argument('--pre_separated', type=bool, default=False, help='if three subfolders already exist for train test and val')
    parser.add_argument('--override', type=bool, default=False, help='make this true to always use the same ones')
    parser.add_argument('--extension', type=str, default="_open_ht_hv", help='extension of the dataset')
    parser.add_argument('--number_of_messages', type=int, default=2, help='number of messages')
    parser.add_argument('--size_of_messages', type=int, default=64, help='size of messages')
    parser.add_argument('--epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--distrib', type=str, default="laplace", help='distribution')
    parser.add_argument('--out', type=int, default=8, help='number of out channels')
    parser.add_argument('--horizon', type=int, default=96, help='horizon')
    parser.add_argument('--many_many', type=bool, default=False, help='Type of sequence, many to many or many to one')

    args = parser.parse_args()

    load_all = args.load_all
    pre_separated = args.pre_separated
    override = args.override
    extension = args.extension
    number_of_messages = args.number_of_messages
    size_of_messages = args.size_of_messages
    epochs = args.epochs
    distrib = args.distrib
    out = args.out
    horizon = args.horizon
    many_many = args.many_many
    
    run(load_all, pre_separated, override, extension, number_of_messages, size_of_messages, epochs, distrib, out, horizon, many_many=many_many)
