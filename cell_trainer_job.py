# %% [markdown]
# In this notebook, I will use torch_geometric to predict the developpement of a graph of positions through time

# %%
"""
I realized I am leaning towards this approach https://doi.org/10.1016/j.trc.2020.102635
"""

# %%
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pickle

import sys
import os
from genericpath import exists

from cell_dataset import CellGraphDataset, load, single_overfit_dataset
from cell_model import GraphEvolution, GraphEvolutionDiscr
from cell_utils import GraphingLoss, make_animation
from cell_training import train, test_single, compute_parameters, run_single_recursive

import matplotlib.pyplot as plt

import os
process = None

print(torch.cuda.is_available())

import argparse

#https://github.com/clovaai/AdamP
from adamp import AdamP

sys.path.append('/home/nstillman/1_sbi_activematter/cpp_model')
try :
    import allium
except :
    print("Could not import allium")

# %% [markdown]
# The data is a graph of cells having their own positions and velocity.
# 
# In the graph, we will first start by connecting all the edges, then maybe later make radius_graphs to reduce the cost of the pass through the model

# %%
load_all =  True #load directly from a pickle
pre_separated = False #if three subfolders already exist for train test and val

override = False #make this true to always use the same ones

#check against the sysargs
parser = argparse.ArgumentParser(description='Data choice, message size and number choice.', exit_on_error=False)
parser.add_argument('--extension', type=str, default="_random_sample")
parser.add_argument('--number_of_messages', type=int, default=2)
parser.add_argument('--size_of_messages', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--absolute', type=int, default=0)

args = parser.parse_args()

extension = args.extension
number_of_messages = args.number_of_messages
size_of_messages = args.size_of_messages
epochs = args.epochs
absolute = args.absolute

model_path = "models/model" + extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative")
loss_path = "models/loss" + extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative")

print("Model path : ", model_path)

data_train, data_test, data_val = load(load_all, extension, pre_separated, override)

# %%
#INFO : if bg_load is True, this starts the loading, if skipped, bg_loading will take place as soon as a get is called
rval, edge_index, edge_attr, border, params = data_train.get(0)
rval, edge_index, edge_attr, border, params = data_test.get(0)

print("Is data wrapped ? ", data_train.wrap)

# %% [markdown]
# Next we need to define the model that will be used :
#     > input 
#         (1) Graph at a particular time t (nodes having x,y,dx,dy as attributes)
#         (2) Graphs up to a particular time [t-a, t] (nodes having x,y as attributes)
#     > output
#         (a) Graph at the immediate next time step t+1
#         (b) Graph [t, t+b]
#         (c) Graph at t+b
#     > graph size
#         (x) Fixed graph size to the most nodes possible (or above)
#         (y) Unbounded graph size
#             >> idea : graph walks
#             >> idea : sampler

# %% [markdown]
# The following model will do (1ax)

# %%
def start(model : GraphEvolution, optimizer : torch.optim.Optimizer, scheduler  : torch.optim.lr_scheduler._LRScheduler,\
          data_train : CellGraphDataset, data_test : CellGraphDataset, device : torch.device, epoch : int, offset : int, grapher : GraphingLoss, save=0, save_datasets=True):
    
    loss_history_train = {"loss_mean" : [], "loss_log" : [], "loss" : []}
    loss_history_test_recursive = {"loss_mean" : [], "loss_log" : [], "loss" : []}
    loss_history_test_single = {"loss_mean" : [], "loss_log" : [], "loss" : []}
    for e in range(offset, offset + epoch):
        
        recursive = epoch > 10

        train(model, optimizer, scheduler, data_train, device, e, process, max_epoch=offset+epoch, loss_history = loss_history_train, recursive=recursive)

        #model.show_gradients()
        
        if(e == 0 and save_datasets) :
            data_train.thread = None
            data_test.thread = None
            with open("data/training" + extension + ".pkl", 'wb') as f:
                pickle.dump(data_train, f)
            with open("data/testing " + extension + ".pkl", 'wb') as f:
                pickle.dump(data_test, f)
            print("Saved datasets")
        

        test_loss_s = test_single(model, data_test, device, loss_history_test_recursive, duration=16, recursive=False)
        test_loss_r = test_single(model, data_test, device, loss_history_test_single, duration=16, recursive=True)

        print("Epoch : ", e, "Test loss : ", test_loss_s, "Test loss recursive : ", test_loss_r)


        grapher.plot_losses(title="Training", data=loss_history_train, extension=extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative") + "_") 
        grapher.plot_losses(title="Testing", data=loss_history_test_recursive, extension=extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative") + "_") 
        grapher.plot_losses(title="Testing recursive", data=loss_history_test_single, extension=extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative") + "_") 
        
        if (e%save == 0) :      
            all_params_out, all_params_true = compute_parameters(model, data_test, device, duration=-1)
            grapher.plot_params(all_params_out, all_params_true, e, extension=extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative"))
        
        if (save and (e%save == 0 or e == epoch-1)) :
            torch.save(model.state_dict(), model_path + str(e) + ".pt")

# %%

model = GraphEvolution(in_channels=14, out_channels=4, hidden_channels=size_of_messages, dropout=0.05, edge_dim=2, messages=number_of_messages, wrap=data_train.wrap, absolute=absolute)

# %%
#might want to investigate AdamP 
optimizer = AdamP(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-2, weight_decay=5e-3, delta=0.1, wd_ratio=0.1, nesterov=True)
scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
grapher = GraphingLoss()

model = model.to(device)

#all_params_out, all_params_true = compute_parameters(model.to(device), data_test, device, duration=0)
#grapher.plot_params(all_params_out, all_params_true, epoch_to_load, extension=extension)

# %%
start(model, optimizer, scheduler, data_train, data_test, device, \
        epochs, 0, grapher=grapher, save=20, save_datasets=False)

# %%
all_params_out, all_params_true = compute_parameters(model, data_test, device, duration=-1)
grapher.plot_params(all_params_out, all_params_true, epochs - 1, extension=extension)