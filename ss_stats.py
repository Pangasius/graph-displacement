import torch
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('/home/nstillman/1_sbi_activematter/cpp_model')

from allium import SimData
import allium.summstats as ss  

import numpy as np
import json 

from cell_utils import make_animation, make_real_animation

def create_stats(data_y, data_x, path_name, name_complete, extension=None) :
    data_y = torch.cat((data_y[:,1:,:,:2], data_y[:,1:,:,:2] - data_y[:,:-1,:,:2]), dim=3)
    data_x = torch.cat((data_x[:,1:,:,:2], data_x[:,1:,:,:2] - data_x[:,:-1,:,:2]), dim=3)

    #here we skip the first of the data because it has 0 speed and it makes the plot crash (1/0)
    data_y = data_y.cpu().numpy() #model
    data_x = data_x.cpu().numpy() #data
    
    #make an animation of the prediction
    make_animation((data_y[0], data_x[0]), path_name + "model" + name_complete + ".gif", True)
    
    if extension is None :
        #make an animation of the prediction using all predicted parameters
        #make_real_animation((data_y[0], data_x[0]), path_name + "model" + name_complete + "_real.gif")
        label = "Real data"
    else :
        label = "Synthetic data"
    
    #we will show the distribution of speeds
    speed_x_axis0 = data_x[:,:,:,2].flatten() 
    speed_x_axis1 = data_x[:,:,:,3].flatten()
    speed_x_total = np.sqrt(speed_x_axis0**2 + speed_x_axis1**2)
    
    speed_y_axis0 = data_y[:,:,:,2].flatten()
    speed_y_axis1 = data_y[:,:,:,3].flatten()
    speed_y_total = np.sqrt(speed_y_axis0**2 + speed_y_axis1**2) 
    
    f, ax = plt.subplots(1, 3, figsize=(10, 3))
    
    ax[0].hist(speed_y_axis0, bins=100, color="red", density=True, alpha=0.5)
    ax[1].hist(speed_y_axis1, bins=100, color="red", density=True, alpha=0.5)
    ax[2].hist(speed_y_total, bins=100, color="red", density=True, alpha=0.5)
    
    ax[0].hist(speed_x_axis0, bins=100, color="blue", density=True, alpha=0.5)
    ax[1].hist(speed_x_axis1, bins=100, color="blue", density=True, alpha=0.5)
    ax[2].hist(speed_x_total, bins=100, color="blue", density=True, alpha=0.5)
    
    ax[0].set_title("Speed distribution in x")
    ax[1].set_title("Speed distribution in y")
    ax[2].set_title("Speed distribution")
    
    if extension is not None :
        ax[0].set_xlim(-3, 3) if extension.__contains__("_hv") else ax[0].set_xlim(-0.5, 0.5)
        ax[1].set_xlim(-3, 3) if extension.__contains__("_hv") else ax[1].set_xlim(-0.5, 0.5)
        ax[2].set_xlim(0, 3) if extension.__contains__("_hv") else ax[2].set_xlim(0, 0.5)
    
    #make a single x label
    f.text(0.5, 0.04, 'Speed Magnitude', ha='center')
    
    #shift the graph up a bit to allow for the x label
    f.subplots_adjust(bottom=0.15)
    
    #make a single y label
    f.text(0.04, 0.5, 'Number of particles, normalized', va='center', rotation='vertical')
    
    #make a global title
    f.suptitle('Speed magnitude distribution of the synthetic data (blue) and the model (red)')
    
    #make a little room for the title
    f.subplots_adjust(top=0.85)
    
    f.savefig(path_name + "speed_distribution" + name_complete + ".png")
    plt.close()

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
        
    duration = data_x.shape[1]
        
    velbins=np.linspace(0,5,duration + 2)
    velbins2=np.linspace(-2,2,duration + 2)

        
    for i in range(data_x.shape[0]):
        data_x_i = data_x[i]
        data_y_i = data_y[i]
        
        data_x_i_s = SyntheticData(loadtimes = [0,duration-1], types = [0,1], debug = False, data = data_x_i, params = parameters, trackAll=True, dt = 1)
        data_y_i_s = SyntheticData(loadtimes = [0,duration-1], types = [0,1], debug = False, data = data_y_i, params = parameters, trackAll=True, dt = 1)
        
        Nsnap  = data_x_i_s.Nsnap
        dt = data_x_i_s.param.dt
        output_time = data_x_i_s.param.output_time

        vav, vdist,vdist2 = ss.getVelDist(data_x_i_s, velbins,velbins2, usetype=[0,1],verbose=False)

        vdist = vdist[1:]
        vdist2 = vdist2[vdist2 != max(vdist2)]
        
        vav_x.append(vav)
        vdist_x.append(vdist)
        vdist2_x.append(vdist2)
        
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
    plt.plot(velbins[2:]-db/2,vdist_x_mean,'b.-',lw=2, label=label)
    plt.fill_between(velbins[2:]-db/2,vdist_x_mean+vdist_x_std,vdist_x_mean-vdist_x_std,color='b',alpha=0.2)
    plt.plot(velbins[2:]-db/2,vdist_y_mean,'r.-',lw=2, label='Model')
    plt.fill_between(velbins[2:]-db/2,vdist_y_mean+vdist2_y_std,vdist_y_mean-vdist_y_std,color='r',alpha=0.2)
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity magnitude distribution')
    plt.legend()
    #also set the axis to center on x
    if extension is not None :
        plt.ylim(0, 3e1)
    fig.savefig(path_name + 'svmd' + name_complete + '.png')
    plt.close()
    
    #save the data of the mean of the distribution
    np.save(path_name + 'svmd' + name_complete + '.npy', np.array([velbins[2:]-db/2,vdist_x_mean,vdist_x_std,vdist_y_mean,vdist_y_std]))

    fig=plt.figure()
    db=velbins2[1]-velbins2[0]
    plt.plot(velbins2[2:],vdist2_x_mean,'b.-',lw=2, label=label)
    plt.fill_between(velbins2[2:],vdist2_x_mean+vdist2_x_std,vdist2_x_mean-vdist2_x_std,color='b',alpha=0.2)
    plt.plot(velbins2[2:],vdist2_y_mean,'r.-',lw=2, label='Model')
    plt.fill_between(velbins2[2:],vdist2_y_mean+vdist2_y_std,vdist2_y_mean-vdist2_y_std,color='r',alpha=0.2)
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity component distribution')
    plt.legend()
    if extension is not None :
        plt.ylim(2e-4,1e1)
    fig.savefig(path_name + 'svcd' + name_complete + '.png')
    plt.close()
    
    #save the data of the mean of the distribution
    np.save(path_name + 'svcd' + name_complete + '.npy', np.array([velbins2[2:],vdist2_x_mean,vdist2_x_std,vdist2_y_mean,vdist2_y_std]))

    fig = plt.figure()
    xval=np.linspace(0,(Nsnap-1)*dt*output_time,num=Nsnap-1)
    plt.plot(xval,vav_x_mean,'b.-',lw=2, label=label)
    plt.fill_between(xval,vav_x_mean+vav_x_std,vav_x_mean-vav_x_std,color='b',alpha=0.2)
    plt.plot(xval,vav_y_mean,'r.-',lw=2, label='Model')
    plt.fill_between(xval,vav_y_mean+vav_y_std,vav_y_mean-vav_y_std,color='r',alpha=0.2)
    plt.xlabel('time')
    plt.ylabel('Mean velocity')
    plt.title('Mean velocity')
    plt.legend()
    if extension is not None :
        plt.ylim(vav_x_mean.min() * 0.85,vav_x_mean.max() * 1.15)
    fig.savefig(path_name + 'mv' + name_complete + '.png')
    plt.close()
    
    #save the data of the mean of the distribution
    np.save(path_name + 'mv' + name_complete + '.npy', np.array([xval,vav_x_mean,vav_x_std,vav_y_mean,vav_y_std]))

    fig=plt.figure()
    plt.loglog(tval,msd_x_mean,'b.-',lw=2, label=label)
    plt.fill_between(tval,msd_x_mean+msd_x_std,msd_x_mean-msd_x_std,color='b',alpha=0.2)
    plt.loglog(tval,msd_x_mean[1]/(1.0*tval[1])*tval,'--',lw=2,color="cyan")
    
    plt.loglog(tval,msd_y_mean,'r.-',lw=2, label='Model')
    plt.fill_between(tval,msd_y_mean+msd_y_std,msd_y_mean-msd_y_std,color='r',alpha=0.2)
    plt.loglog(tval,msd_y_mean[1]/(1.0*tval[1])*tval,'--',lw=2,color="orange")
    plt.xlabel('time (hours)')
    plt.ylabel('MSD')
    plt.title('Mean square displacement')
    plt.legend()
    if extension is not None :
        if extension.__contains__("_lv") :
            plt.ylim(1e-2,msd_x_mean.max() * 1.05)
        else :
            plt.ylim(1,msd_x_mean.max() * 1.05)
    fig.savefig(path_name + 'msd' + name_complete + '.png')
    plt.close()
    print("Done")
    
    #save the data of the mean of the distribution
    np.save(path_name + 'msd' + name_complete + '.npy', np.array([tval,msd_x_mean,msd_x_std,msd_y_mean,msd_y_std]))