#successively reads every models in a folder and evaluates them
import os
import torch
import pickle

from cell_training import compute_parameters
from cell_model import GraphEvolution

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('data/testing_random_sample.pkl', 'rb') as f:
    data = pickle.load(f)
    

means = {}
stds = {}

for filename in os.listdir('models/imported'):
    if filename.endswith(".pt"):
        print("Loading " + filename)
        #parse the filename to get the parameters #filename = "model_random_sample_1_200_0.pt"

        filename_split = filename.split("_")
        number_of_messages = int(filename_split[3]) 
        size_of_messages = int(filename_split[4])
        
        model = GraphEvolution(in_channels=14, out_channels=4, hidden_channels=size_of_messages, dropout=0.05, edge_dim=2, messages=number_of_messages, wrap=data.wrap)
        model.load_state_dict(torch.load('models/imported/' + filename))
        model.to(device)
    else:
        continue
    
    params_out, params_true = compute_parameters(model, data, device, duration=0)
    
    keys = list(params_out[0])
        
    #all lengths should be the same
    length_inside_keys = params_out[0].get(keys[0]).size()[0] #type: ignore
    
    values_out = torch.zeros((len(params_out), len(keys), length_inside_keys))
    for i, param in enumerate(params_out) :
        for j, key in enumerate(keys) :
            values_out[i, j] = param[key]
                
    values_true = torch.zeros((len(params_true), len(keys), length_inside_keys))      
    for i, param in enumerate(params_true) :
        for j, key in enumerate(keys) :
            values_true[i, j] = param[key]
            
    
    #compute the average loss for each parameter
    loss_means = torch.mean(torch.mean(torch.abs(values_out - values_true), dim=2), dim=0)
    loss_stds = torch.std(torch.mean(torch.abs(values_out - values_true), dim=2), dim=0)
    
    means[filename] = loss_means
    stds[filename] = loss_stds

x = torch.arange(len(keys))  # the label locations
width = 1/(len(means) + 1)  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=stds[attribute])
    ax.bar_label(rects, padding=3)
    multiplier += 1

#format the numbers above the bars to be in scientific notation with 2 decimals
ax.yaxis.set_major_formatter('{x:.2e}')

ax.set_title('Absolute error for each parameter')
#the x axis are the keys of the parameters
ax.set_xticks(x + width * multiplier / 2, keys)
#rescale so that the labels are visible
ax.autoscale_view()
    
ax.legend(loc='upper left', ncols=2)
plt.show()
    