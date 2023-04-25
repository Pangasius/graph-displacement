import torch
import andi
from sbi import utils

import pickle
import numpy as np
import matplotlib.pyplot as plt

def eMSD(data,framerate=1,verbose = False):
    Nparticles = np.arange(data.shape[0])
    Nsnap = data.shape[1]
    msds = np.zeros(data.shape[1])
    for n in Nparticles:
        sqdist = data[n].sum(axis=1)**2
        msds[:]+=sqdist
    msd = msds/data.shape[0]
    tval=np.linspace(0,Nsnap*framerate,num=Nsnap)
    return msd,tval

def tMSD(data,framerate=1,verbose = False):
    shifts = np.arange(data.shape[0])
    msd = np.zeros(shifts.size)    
    Nsnap = data.shape[0]
    
    for i, shift in enumerate(shifts):
        
        diffs = data[:-shift if shift else None] - data[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msd[i] = sqdist.mean()
    
    tval=np.linspace(0,Nsnap*framerate,num=Nsnap)
    return msd,tval

def etMSD(data):
    Nparticles = np.arange(data.shape[0])
    Nsnap = data.shape[1]
    msds = np.zeros(data.shape[1])
    
    for n in Nparticles:
        tmsd, tval = tMSD(data[n,:])
        msds += tmsd

    msd = msds/data.shape[0]
    return msd,tval

def simulate(theta, settings=(250,1,2),verbose=False, out='data'):
    T,N,m = settings
    dataset = AD.create_dataset(T = T, N = N, exponents = float(theta), models = models[m], dimension = 2)

    new_d = dataset[:,2:]

    data = np.zeros((N,T,2))
    X = new_d[:,:T]
    Y = new_d[:,T:]
    for n in range(N):
        data[n,:,0] = X[n,:]
        data[n,:,1] = Y[n,:]
        
    tmsd, tval1 = tMSD(data[0])

    emsd, tval2 = eMSD(data)

    etmsd, etval = etMSD(data)

    if verbose:
        plt.figure(dpi=200)
        plt.loglog(tval1,emsd,'--', label='e-msd')
        plt.loglog(tval2,tmsd,'--', label='t-msd')
        plt.loglog(etval,etmsd,'--', label='et-msd')
        plt.xlabel('t')
        plt.ylabel('<r$^2$>')
        plt.legend()
        
    if out == 'alpha': 
        alpha1 = np.polyfit(np.log(tval1[1:]), np.log(tmsd[1:]), 1)[0]
        alpha2 = np.polyfit(np.log(tval2[1:]), np.log(emsd[1:]), 1)[0]
        alpha3 = np.polyfit(np.log(etval[1:]), np.log(etmsd[1:]), 1)[0]
        alpha = [alpha1, alpha2, alpha3]
        return alpha
    
    elif out == 'msd':
        return np.array([emsd, tmsd, etmsd]).flatten()

    elif out == 'data':
        data = data.flatten()
        data = torch.tensor(data).float()
        data = data.reshape(1,-1)
        return data
    
    else:
        alpha1 = np.polyfit(np.log(tval1[1:]), np.log(tmsd[1:]), 1)[0]
        alpha2 = np.polyfit(np.log(tval2[1:]), np.log(emsd[1:]), 1)[0]
        alpha3 = np.polyfit(np.log(etval[1:]), np.log(etmsd[1:]), 1)[0]
        alpha = [alpha1, alpha2, alpha3]

        msd = np.array([emsd, tmsd, etmsd]).flatten()

        data = data.flatten()
        data = torch.tensor(data).float()
        data = data.reshape(1,-1)
        
        return alpha, msd, data

AD = andi.andi_datasets()
# set prior distribution for the parameters 
prior = utils.BoxUniform(low=torch.tensor([0]), 
                             high=torch.tensor([1]))                           

