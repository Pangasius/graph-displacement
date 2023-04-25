import copy 

import numpy as np
import pandas as pd
import glob
import json
import torch


import contextlib
import joblib

# import vtk

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import scipy.interpolate

ss_length = 50

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

try:
    import zuko
    
    def init_prior(bounds ): 
        """
        Returns prior 
        """
        prior_min = bounds[0]
        prior_max = bounds[1]
        return zuko.distributions.BoxUniform(torch.as_tensor(prior_min),torch.as_tensor(prior_max))

except:
    print("Cannot import zuko. Continuing without prior")
        
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallBack(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def read_output(f):
    """
    Reads in previously saved .dat files incase of debugging
    """
    build = True
    for line in open(f, 'r'):
        item = line.rstrip()
        if build:
            col = item.split(',')
            data = pd.DataFrame(columns = col)
            build = False
        else:
            row = pd.Series(item.split(','), index=col)
            data = data.append(row, ignore_index = True)
    return data

def read_params(configfile):
    """
    Reads in parameters from the config files
    """
    params = dict()
    with open(configfile) as jsonFile:
        parameters = json.load(jsonFile)
        for attribute in parameters:
            params[attribute] = parameters[attribute]
    return parameters


# def create_vtk(data,filename, experiment = False):
#     for t in range(data.Nsnap - 1):
#         Nvals = data.rval[t].shape[0] -1
#         vpoints = vtk.vtkPoints()
#         vpoints.SetNumberOfPoints(data.Nvals[t])
#         for i in range(Nvals):
#             if not (data.rval[t][i] == [0., 0.]).all():
#                 vpoints.SetPoint(i, np.append(data.rval[t][i],0))
#         vpoly = vtk.vtkPolyData()
#         vpoly.SetPoints(vpoints)

#         # # Set velocity 
#         vvel = vtk.vtkUnsignedCharArray()
#         vvel.SetNumberOfComponents(3)
#         vvel.SetName("Velocity")
#         vvel.SetNumberOfTuples(data.Nvals[t])
#         for i in range(Nvals):
#             if not (data.rval[t][i] == [0., 0.]).all():
#                 vpoints.SetPoint(i, np.append(data.rval[t][i],0))
#         vpoly.GetPointData().AddArray(vvel)
#         vpoly.GetPointData().SetActiveScalars('Velocity')

#         if not experiment:        
#             # Set R
#             vradius = vtk.vtkUnsignedCharArray()
#             vradius.SetNumberOfComponents(1)
#             vradius.SetName("Radius")
#             vradius.SetNumberOfTuples(data.Nvals[t])
#             for i in range(data.Nvals[t]):
#                 vradius.SetTuple(i, np.array([data.radius[t][i]]))
#             vpoly.GetPointData().AddArray(vradius)
#             vpoly.GetPointData().SetActiveScalars('Radius')
            
#             # Set Z
#             vZ = vtk.vtkUnsignedCharArray()
#             vZ.SetNumberOfComponents(1)
#             vZ.SetName("Z")
#             vZ.SetNumberOfTuples(data.Nvals[t])
#             for i in range(data.Nvals[t]):
#                 vZ.SetTuple(i, np.array([data.Z[t][i]]))
#             vpoly.GetPointData().AddArray(vZ)
#             vpoly.GetPointData().SetActiveScalars('Z')

#             # Set type
#             vtype = vtk.vtkUnsignedCharArray()
#             vtype.SetNumberOfComponents(1)
#             vtype.SetName("Type")
#             vtype.SetNumberOfTuples(data.Nvals[t])
#             for i in range(data.Nvals[t]):
#                 vtype.SetTuple(i, np.array([data.ptype[t][i]]))
#             vpoly.GetPointData().AddArray(vtype)
#             vpoly.GetPointData().SetActiveScalars('Type')

#         writer = vtk.vtkPolyDataWriter()
#         writer.SetInputData(vpoly)
#         writer.SetFileName(f"{filename}_{t}.vtk")
#         writer.Write()

#         print(t,end='\r')
#     return 0 

# def plot_sim(data, filename, frames= 0):
#     """
#     Makes animation of simulation output
#     """

#     def animate(t,drift=False):
#         ax.clear()
#         ax.scatter(data.rval[t,:,0], data.rval[t,:,1]
#                    , s=10, color='#f3f6f4', facecolor = '#444444')
#         ax.quiver(data.rval[t,:,0],data.rval[t,:,1],data.vval[t,:,0],data.vval[t,:,1],color='#fff2cc')
#         ax.set_title(f'time={t}')    
#         ax.set_xlim([-data.param.Lx/2, data.param.Lx/2])
#         ax.set_ylim([-data.param.Ly/2,data.param.Ly/2])
#         # ax.get_yaxis().set_visible(False)
#         # ax.get_xaxis().set_visible(False)

#     t = 0
#     fig, ax = plt.subplots(dpi=200)
#     ax.set_facecolor("#444444")
#     ax.set_xlim([-data.param.Lx/2, data.param.Lx/2])
#     ax.set_ylim([-data.param.Ly/2,data.param.Ly/2])

#     # ax.get_yaxis().set_visible(False)
#     # ax.get_xaxis().set_visible(False)
#     l1 = ax.scatter(data.rval[t,:,0],data.rval[t,:,1], s=10, color='#f3f6f4', facecolor = '#444444')
#     l2 = ax.quiver(data.rval[t,:,0],data.rval[t,:,1],data.vval[t,:,0],data.vval[t,:,1],color='#fff2cc')
#     # ax.set_title(f'time={t}')

#     # Set up formatting for the movie files
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

#     if frames == 0:
#         time = data.rval.shape[0]
#     else:
#         time = frames
#     anim = animation.FuncAnimation(
#         fig, animate, interval=10, frames=time, blit=False,fargs=(False,))
#     anim.save(f'{filename}.mp4', writer=writer) 

#     return 0 


def prep_vav(ss, bounds = [0,1], nstd = 2):
    
    def moving_average(x, w, option='valid'):
        return np.convolve(x, np.ones(w), option) / w

    x = np.linspace(0,1,len(ss['vav']))
    y = ss['vav']

    params = np.polyfit(x, y, 1)
    linpart = x*params[0] + params[1]
    lin_y = y/linpart
    std_vel = lin_y.std()
    vav = lin_y.mean() 
    ind = lin_y > (vav + nstd*std_vel)

    y_interp = scipy.interpolate.interp1d(x[~ind], y[~ind])  
    X = np.linspace(x[~ind][0],x[~ind][-1],ss_length + 4)
    Y = y_interp(X)
    Y = moving_average(Y, 5, 'valid') 
    
    return X[2: ss_length + 2], Y 

def prep_msd(ss, bounds = [-1.75,1.75]):
    x = np.log(ss['tval'])
    Y = np.log(ss['msd'])

    X = np.linspace(bounds[0],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)

def prep_vmag(ss, bounds = [0,4]):
    ind = ss['vdist'] > 0

    dummy = ss['velbins'][1:]
    db = dummy[1]-dummy[0]
    x = dummy[ind] - db/2 
    y = np.log(ss['vdist'][ind])
    
    X = np.linspace(x[0],bounds[1],ss_length)

    y_interp = scipy.interpolate.interp1d(x, y)
    
    return X, y_interp(X)

def prep_vcompx(ss, bounds = [-3.25,3.25]):
    ind = ss['vdistx'] > 0

    dummy = ss['velbin2'][1:]
    db = dummy[1]-dummy[0]
    x = dummy[ind] - db/2 
    y = np.log(ss['vdistx'][ind])
    
    X = np.linspace(bounds[0],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, y)
    
    return X, y_interp(X)

def prep_vcompy(ss, bounds = [-3.25,3.25]):
    ind = ss['vdisty'] > 0

    dummy = ss['velbin2'][1:]
    db = dummy[1]-dummy[0]
    x = dummy[ind] - db/2 
    y = np.log(ss['vdisty'][ind])
    
    X = np.linspace(bounds[0],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, y)
    
    return X, y_interp(X)

def prep_Nvals(ss, bounds = [1,100]):

    Y = np.log(ss['Nvals'])
    x = np.arange(len(Y))

    X = np.linspace(x[0],x[-1],100)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)

def prep_gr(ss, bounds = [3,50]):
    x = ss['rdist']
    Y = ss['gr']

    X = np.linspace(bounds[0],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)

def prep_fourier_vel(ss, bounds = [0,0.4]):
    
    x = ss['qrad1'][1:]
    Y = np.log(np.mean(ss['Sqrad1'],axis=0)[1:])

    X = np.linspace(ss['qrad1'][1],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)


def prep_fourier_spatial(ss, bounds = [-4.8,-0.85]):
    
    x = ss['qrad0'][1:]
    Y = np.mean(ss['valrad0'],axis=0)[1:]

    X = np.linspace(bounds[0],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)

def prep_velautocorr(ss, bounds = [0,4]):
    x = ss['tval2']
    Y = ss['velauto']

    X = np.linspace(x[1],bounds[1],ss_length)
    
    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)


def prep_selfint(ss, bounds = [-1.75, 1.75], log=True):

    x = np.log(ss['tval3'])
    Y = ss['SelfInt2']

    X = np.linspace(bounds[0],bounds[1],ss_length)

    y_interp = scipy.interpolate.interp1d(x, Y)
    
    return X, y_interp(X)

# def prep_selfint(ss, bounds = [-1.8, 1.2], npnts = 92, log=True):

#     x = ss['tval3']
#     Y = ss['SelfInt2']
#     if not log: 
#         X = np.linspace(x[1],bounds[1],ss_length)
#     else:
#         X = np.logspace(bounds[0],bounds[1],ss_length)       
#     y_interp = scipy.interpolate.interp1d(x, Y)
    
#     return X, y_interp(X)


def normbyvel(data,types=[1,2]):
    newdata2 = copy.deepcopy(data)
    vicsek_param = []
    data.vvect = np.zeros((data.Nsnap,2))
    newdata2.vvect = np.zeros((data.Nsnap,2))
    for whichframe in range(data.Nsnap-1):   
        # get only points with velocity data
        isdata = data.gettypes(types, whichframe)         

        vx = data.vval[whichframe,isdata,0]
        vy = data.vval[whichframe,isdata,1]

        vav = np.sqrt(np.sum(vx**2 + vy**2)/len(isdata))
        data.vvect[whichframe,:] = [np.mean(vx), np.mean(vy)]
        newdata2.vvect[whichframe,:] = [np.mean(vx), np.mean(vy)]
        newdata2.vval[whichframe,isdata] /= vav

        vicsek_vect = np.mean(newdata2.vval[whichframe,isdata],axis=0)
        # vicsek_param_t = np.sqrt(vicsek_vect[0]**2 + vicsek_vect[1]**2)
        vicsek_param.append(vicsek_vect)

    return newdata2, np.array(vicsek_param)