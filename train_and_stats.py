import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pickle

import sys
import os
from genericpath import exists

from cell_dataset import CellGraphDataset, load, single_overfit_dataset
from cell_model import Gatv2Predictor, Gatv2PredictorDiscr, ConvPredictor
from cell_utils import GraphingLoss, make_animation
from cell_training import train, test_single, compute_parameters_draw, run_single_recursive, run_single_recursive_draw, predict

import matplotlib.pyplot as plt

import os, psutil
process = psutil.Process(os.getpid())
print("Using : ", process.memory_info().rss // 1000000)  # in megabytes 
print("Available : ", process.memory_info().vms  // 1000000)  # in megabytes 

print(torch.cuda.is_available())

#https://github.com/clovaai/AdamP
from adamp import AdamP

sys.path.append('/home/nstillman/1_sbi_activematter/cpp_model')
try :
    import allium
except :
    print("Could not import allium")
    
from allium import SimData
import numpy as np
import json 
import allium.summstats as ss

import argparse

    
def run(load_all, pre_separated, override, extension, number_of_messages, size_of_messages, absolute, epochs, distrib, aggr) :

    name_complete = extension + "_" + str(number_of_messages) + "_" + str(size_of_messages) + "_" + ("absolute" if absolute else "relative") + "_" + distrib + "_" + aggr

    print(name_complete)

    model_path = "models/model" + name_complete

    data_train, data_test, data_val = load(load_all, extension, pre_separated, override)

    print("\nData loaded\n")

    def start(model : Gatv2Predictor, optimizer : torch.optim.Optimizer, scheduler  : torch.optim.lr_scheduler._LRScheduler,\
            data_train : CellGraphDataset, data_test : CellGraphDataset, device : torch.device, epoch : int, offset : int, grapher : GraphingLoss, save=0, save_datasets=True):
        
        loss_history_train = {"loss_mean" : [], "loss_log" : [], "loss" : []}
        loss_history_test_recursive = {"loss_mean" : [], "loss_log" : [], "loss" : []}
        for e in range(offset, offset + epoch):
            
            recursive = True

            train(model, optimizer, scheduler, data_train, device, e, process, max_epoch=offset+epoch, recursive=recursive, distrib=distrib, aggr=aggr)

            #model.show_gradients()
            
            if(e == 0 and save_datasets) :
                data_train.thread = None
                data_test.thread = None
                with open("data/training" + extension + ".pkl", 'wb') as f:
                    pickle.dump(data_train, f)
                with open("data/testing" + extension + ".pkl", 'wb') as f:
                    pickle.dump(data_test, f)
                print("Saved datasets")
            

            train_loss = test_single(model, data_train, device, loss_history_train, duration=0, recursive=False, distrib=distrib, aggr=aggr)
            test_loss_r = test_single(model, data_test, device, loss_history_test_recursive, duration=0, recursive=True, distrib=distrib, aggr=aggr)

            print("Epoch : ", e, "Train Loss", train_loss, "Test loss recursive : ", test_loss_r)

            grapher.plot_losses(title="Training", data=loss_history_train, length=min(50, len(data_train)), extension=name_complete + "_") 
            grapher.plot_losses(title="Testing recursive", data=loss_history_test_recursive, length=min(50, len(data_test)), extension=name_complete + "_") 
            
            if (e != 0 and e%save == 0) :      
                all_params_out, all_params_true = compute_parameters_draw(model, data_test, device, duration=-1, distrib=distrib)
                grapher.plot_params(all_params_out, all_params_true, e, extension=name_complete)
            
            if (save and (e%save == 0 or e == epoch-1)) :
                torch.save(model.state_dict(), model_path + str(e) + ".pt")
                
    model = Gatv2Predictor(in_channels=12, out_channels=4, hidden_channels=size_of_messages, dropout=0.05, edge_dim=2, messages=number_of_messages, wrap=data_train.wrap, absolute=absolute)

    #might want to investigate AdamP 
    optimizer = AdamP(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-2, weight_decay=5e-3, delta=0.1, wd_ratio=0.1, nesterov=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grapher = GraphingLoss()

    model = model.to(device)

    start(model, optimizer, scheduler, data_train, data_test, device, \
            epochs, 0, grapher=grapher, save=50, save_datasets=False)

    print("\nFinished training\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_y, data_x = predict(model, data_test, device, -1, recursive=True, distrib=distrib)
    
    data_y = torch.cat((data_y[:,1:], data_y[:,1:,:,:] - data_y[:,:-1,:,:]), dim=3)
    data_x = torch.cat((data_x[:,1:], data_x[:,1:,:,:] - data_x[:,:-1,:,:]), dim=3)
    
    #here we skip the first of the data because it has 0 speed and it makes the plot crash (1/0)
    data_y = data_y.cpu().numpy() #model
    data_x = data_x.cpu().numpy() #data

    class Parameters(object):
        def __init__(self, p):
            for key, values in p.items():
                setattr(self, key, values)

    class SyntheticData:
        @staticmethod
        def checkTypes(readtypes,data):
            #check which particles to load 
            if len(readtypes) > 0:
                usetypes = np.isin(data[:,-1],readtypes)
            else:
                usetypes = [True]*len(data)
            return usetypes

        # Data object for summary statistics
        def __init__(self,**kwargs):
            # check for debugging
            try:
                self.debug = kwargs['debug']
                if self.debug:
                    print('kwargs: ', kwargs)
            except:
                self.debug = False
            # check for specific loadtimes
            try:    
                self.start = kwargs["loadtimes"][0]
                self.end = kwargs["loadtimes"][1]
                self.multiopt = True
            except:
                self.multiopt = False
            # check for specific types
            try:
                self.readtypes = kwargs["readtypes"]
            except:
                self.readtypes = []
                
            try:
                self.dt = kwargs["dt"]
            except:
                self.dt = 1
            # load parameters
            try:    
                self.param = Parameters(kwargs['params'])
            except Exception as e:
                print('Error! Parameters must be a dictionary', e)
                return 1
            # load multiple simulation snapshots
            self.Nsnap = self.end - self.start + 1
            #get maximum number of particles
            self.N = sum(SimData.checkTypes(self.readtypes, kwargs['data'][0]))
            self.Nvals = []
            self.Nvariable =  False
            for t in range(self.start,self.end):
                self.Nvals.append(sum(SimData.checkTypes(self.readtypes, kwargs['data'][t])))
                if self.Nvals[t] > self.N:
                    self.N = self.Nvals[t] 
                    self.Nvariable = True
            if kwargs['trackAll'] and self.Nvariable == False:
                self.Ntracers = self.Nvals[0]
            else:
                print('Error! Currently assuming tracking all cells')
            self.data = kwargs['data']
            self.flag = np.zeros((self.Nsnap,self.N))
            if self.data.shape[-1] == 5:
                self.Z = self.data[:,:,4]
            else:
                np.zeros((self.Nsnap,self.N))
                
            self.rval = self.data[:,:,:2]
            self.vval = self.data[:,:,2:4]
            self.theta = np.zeros((self.Nsnap,self.N))
            self.nval = np.zeros((self.Nsnap,self.N,2))
            self.radius = np.ones((self.Nsnap,self.N))
            self.ptype = np.ones((self.Nsnap,self.N))
            self.sigma = 0.
            
            return None

        def gettypes(self, readtypes, frames):
            return np.isin(self.ptype[frames],readtypes)

        def truncateto(self,start, endtime):
            self.Nsnap = endtime - start
            self.flag =  self.flag[start:endtime]
            self.rval = self.rval[start:endtime]
            self.vval = self.vval[start:endtime]
            self.theta = self.theta[start:endtime]
            self.nval = self.nval[start:endtime]
            self.radius = self.radius[start:endtime]
            self.ptype = self.ptype[start:endtime]
            self.Nvals = self.Nvals[start:endtime]
            self.Ntracers = self.Ntracers[start:endtime]


    configfile = "simconfig_open.json"

    with open(configfile) as jsonFile:
        parameters = json.load(jsonFile)
        
    vav_x = []
    vdist_x = []
    vdist2_x = []

    vav_y = []
    vdist_y = []
    vdist2_y = []

    msd_x = []
    msd_y = []
        
    velbins=np.linspace(0,5,100)
    velbins2=np.linspace(-2,2,100)

        
    for i in range(data_x.shape[0]):
        data_x_i = data_x[i]
        data_y_i = data_y[i]
        
        data_x_i_s = SyntheticData(loadtimes = [0,96], types = [0,1], debug = False, data = data_x_i, params = parameters, trackAll=True, dt = 1)
        data_y_i_s = SyntheticData(loadtimes = [0,96], types = [0,1], debug = False, data = data_y_i, params = parameters, trackAll=True, dt = 1)
        
        Nsnap  = data_x_i_s.Nsnap
        dt = data_x_i_s.param.dt
        output_time = data_x_i_s.param.output_time

        vav, vdist,vdist2 = ss.getVelDist(data_x_i_s, velbins,velbins2, usetype=[0,1],verbose=False)

        vdist = vdist[1:]
        vdist2 = vdist2[vdist2 != max(vdist2)]
        
        vav_x.append(vav)
        vdist_x.append(vdist)
        vdist2_x.append(vdist2)
        
        velbins=np.linspace(0,5,100)
        velbins2=np.linspace(-2,2,100)
        vav, vdist,vdist2 = ss.getVelDist(data_y_i_s, velbins,velbins2, usetype=[0,1],verbose=False)

        vdist = vdist[1:]
        vdist2 = vdist2[vdist2 != max(vdist2)]
        
        vav_y.append(vav)
        vdist_y.append(vdist)
        vdist2_y.append(vdist2)
        
        tval, msd, d = ss.getMSD(data_x_i_s, usetype=[1],verbose=False)
        
        msd_x.append(msd)
        
        tval, msd, d = ss.getMSD(data_y_i_s, usetype=[1],verbose=False)
        
        msd_y.append(msd)
        
    vav_x = np.array(vav_x)
    vdist_x = np.array(vdist_x)
    vdist2_x = np.array(vdist2_x)

    vav_y = np.array(vav_y)
    vdist_y = np.array(vdist_y)
    vdist2_y = np.array(vdist2_y)

    msd_x = np.array(msd_x)
    msd_y = np.array(msd_y)

    vdist_x_mean = np.mean(vdist_x, axis=0)
    vdist_x_std = np.std(vdist_x, axis=0)

    vdist2_x_mean = np.mean(vdist2_x, axis=0)
    vdist2_x_std = np.std(vdist2_x, axis=0)

    vav_x_mean = np.mean(vav_x, axis=0)
    vav_x_std = np.std(vav_x, axis=0)

    vdist_y_mean = np.mean(vdist_y, axis=0)
    vdist_y_std = np.std(vdist_y, axis=0)

    vdist2_y_mean = np.mean(vdist2_y, axis=0)
    vdist2_y_std = np.std(vdist2_y, axis=0)

    vav_y_mean = np.mean(vav_y, axis=0)
    vav_y_std = np.std(vav_y, axis=0)

    msd_x_mean = np.mean(msd_x, axis=0)
    msd_x_std = np.std(msd_x, axis=0)

    msd_y_mean = np.mean(msd_y, axis=0)
    msd_y_std = np.std(msd_y, axis=0)

    fig=plt.figure()
    db=velbins[1]-velbins[0]
    plt.semilogy(velbins[2:]-db/2,vdist_x_mean,'r.-',lw=2, label='Data')
    plt.fill_between(velbins[2:]-db/2,vdist_x_mean+vdist_x_std,vdist_x_mean-vdist_x_std,color='r',alpha=0.2)
    plt.semilogx(velbins[2:]-db/2,vdist_y_mean,'b.-',lw=2, label='Model')
    plt.fill_between(velbins[2:]-db/2,vdist_y_mean+vdist2_y_std,vdist_y_mean-vdist_y_std,color='b',alpha=0.2)
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity magnitude distribution')
    plt.legend()
    fig.savefig('svmd' + name_complete + '.pdf')

    fig=plt.figure()
    db=velbins2[1]-velbins2[0]
    plt.semilogy(velbins2[2:],vdist2_x_mean,'r.-',lw=2, label='Data')
    plt.fill_between(velbins2[2:],vdist2_x_mean+vdist2_x_std,vdist2_x_mean-vdist2_x_std,color='r',alpha=0.2)
    plt.semilogx(velbins2[2:],vdist2_y_mean,'b.-',lw=2, label='Model')
    plt.fill_between(velbins2[2:],vdist2_y_mean+vdist2_y_std,vdist2_y_mean-vdist2_y_std,color='b',alpha=0.2)
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity component distribution')
    plt.legend()
    fig.savefig('svcd' + name_complete + '.pdf')

    fig = plt.figure()
    xval=np.linspace(0,(Nsnap-1)*dt*output_time,num=Nsnap-1)
    plt.plot(xval,vav_x_mean,'r.-',lw=2, label='Data')
    plt.fill_between(xval,vav_x_mean+vav_x_std,vav_x_mean-vav_x_std,color='r',alpha=0.2)
    plt.plot(xval,vav_y_mean,'b.-',lw=2, label='Model')
    plt.fill_between(xval,vav_y_mean+vav_y_std,vav_y_mean-vav_y_std,color='b',alpha=0.2)
    plt.xlabel('time')
    plt.ylabel('mean velocity')
    plt.legend()
    fig.savefig('mv' + name_complete + '.pdf')

    fig=plt.figure()
    plt.loglog(tval,msd_x_mean,'r.-',lw=2, label='Data')
    plt.fill_between(tval,msd_x_mean+msd_x_std,msd_x_mean-msd_x_std,color='r',alpha=0.2)
    plt.loglog(tval,msd_x_mean[1]/(1.0*tval[1])*tval,'--',lw=2,color="orange")
    plt.loglog(tval,msd_y_mean,'b.-',lw=2, label='Model')
    plt.fill_between(tval,msd_y_mean+msd_y_std,msd_y_mean-msd_y_std,color='b',alpha=0.2)
    plt.loglog(tval,msd_y_mean[1]/(1.0*tval[1])*tval,'--',lw=2,color="cyan")
    plt.xlabel('time (hours)')
    plt.ylabel('MSD')
    plt.title('Mean square displacement')
    plt.legend()
    fig.savefig('msd' + name_complete + '.pdf')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--load_all', type=bool, default=True, help='load directly from a pickle')
    parser.add_argument('--pre_separated', type=bool, default=False, help='if three subfolders already exist for train test and val')
    parser.add_argument('--override', type=bool, default=False, help='make this true to always use the same ones')
    parser.add_argument('--extension', type=str, default="_open_ht_hv", help='extension of the dataset')
    parser.add_argument('--number_of_messages', type=int, default=2, help='number of messages')
    parser.add_argument('--size_of_messages', type=int, default=64, help='size of messages')
    parser.add_argument('--absolute', type=int, default=0, help='absolute or relative')
    parser.add_argument('--epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--distrib', type=str, default="laplace", help='distribution')
    parser.add_argument('--aggr', type=str, default="mean", help='aggregation')

    args = parser.parse_args()

    load_all = args.load_all
    pre_separated = args.pre_separated
    override = args.override
    extension = args.extension
    number_of_messages = args.number_of_messages
    size_of_messages = args.size_of_messages
    absolute = args.absolute
    epochs = args.epochs
    distrib = args.distrib
    aggr = args.aggr
    
    run(load_all, pre_separated, override, extension, number_of_messages, size_of_messages, absolute, epochs, distrib, aggr)