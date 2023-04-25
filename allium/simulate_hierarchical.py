import sys
import time
import copy 
import signal

sys.path.append('simulator/') 
import pycapmd as capmd

import psutil
import json
import numpy as np
import pickle 
import lzma
import random
import torch

import allium
from joblib import Parallel, delayed
from tqdm import tqdm

class Sim(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.counter = 1
        try:
            print(f'Parameter file loaded from {self.parameterFile}')
        except:
            self.parameterFile = "include/config/simconfig.json"
        #setting default values for testing simulation runs only 
        if not hasattr(self,'log'):
            self.log = False
        if not hasattr(self,'nfeatures'):
            self.nfeatures = 15
        if not hasattr(self,'framerate'):
            self.framerate = 10/60
        if not hasattr(self,'test'):
            self.test = False
        if not hasattr(self,'keys'):
            print('no parameter map detected, using default')
            self.keys = ['factive', 'pairatt', 'tau', 'N']    
        
        if self.test:
            print(self.keys)
            if not hasattr(self,'test_theta'):
                if len(self.keys) == 3:
                    self.test_theta = [130, 85, 7]
                if len(self.keys) == 6:
                    self.test_theta = [6.0225e+01, 7.4157e+01, 4.0066e+00, 4.4993e-01, 2.3977e-03, 9.9745e+02]
                else:
                    print("ERROR: No default parameters saved for this number of parameters. Set test parameters with test_theta")

    def sample(self):
        """
        Sample simulator from proposed prior 

        """
        print(self.num_simulations)
        x = torch.Size([self.num_simulations])
        if len(self.starttime) > 1:
            print("Caution: Running scratch and confluent simultaneously")

        theta = self.proposal.sample(sample_shape=torch.Size([self.num_simulations]))
        with open(f'{self.folder}/{self.run}_{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}_sampled_theta.p', "wb") as f:
            pickle.dump(theta,f)
        self.thetafilename = self.folder + self.run + '_'
        batches = torch.split(theta, self.batch_size, dim=0)
        
        with allium.utils.tqdm_joblib(tqdm(desc="Running simulations", total=self.num_simulations)) as progress_bar:
            simulation_outputs = Parallel(n_jobs=self.num_workers)(
                    delayed(self.wrapper)(idx, p) for idx, p in enumerate(batches))

        x = torch.cat(simulation_outputs, dim=0)
        print(x.shape)
        return theta, simulation_outputs
        
    def wrapper(self, idx, p):
        """
        Returns summary statistics from active particle model of cells.

        Summarizes the output of the simulator and converts it to `torch.Tensor`.
        """
        xout = torch.Tensor()
        for i, params in enumerate(p):
            if not self.log:
                print(f'\nRunning simulation {self.counter}/{self.num_simulations} w params {params}')
            else:
                print(f'\nRunning simulation {self.counter}/{self.num_simulations} w params {params}', 
                      file=open(f'{self.folder}_log.txt', 'a'))

            def sig_handler(signum, frame):
                print(f'Error: segfault w params {params}')

            signal.signal(signal.SIGSEGV, sig_handler)

            self.thetafilename = self.folder + self.run + '_'
            for (a,b) in zip(self.keys,params):
                self.thetafilename+=f'{a}_{b:.1e}_'
            if self.test:
                theta = self.test_theta
                try:
                    testout = self.test_folder
                except:
                    testout = './test_output'
                file = f'{self.run}'
                for (p,t) in zip(self.params,theta):
                    file+= f'_{p}_{t}'

                file = f'{testout}/{file}.p'
                with open( file, 'rb') as f:
                    obs = pickle.load(f)
            else:    
                obs = self.simulate(params)
                if obs == None:
                    return torch.as_tensor([0]*self.nfeatures)

            save = random.uniform(0,1) < self.save_prob
            file = f'p{psutil.Process().cpu_num():03}_r{idx:03}_sb{i:03}_{self.run}'
            if save and not self.test:
                with lzma.open(f'{self.folder}/{file}.pz','wb') as f:
                    pickle.dump(obs, f)
            #Calculate summary statistics here
            if len(self.ssopts) > 0:
                try:
                    if len(self.starttime) > 1:
                        ssvect =[]
                        ssdata =[]
                        for s,e in zip(self.starttime, self.endtime):
                            tmp_obs = copy.deepcopy(obs)
                            tmp_obs.param.framerate = self.framerate
                            # rescale time based on frame rate
                            vect0, data0 = allium.summstats.calculate_summary_statistics(tmp_obs,useall = self.useall,opts = self.ssopts,log = self.log, starttime=s, endtime=e,usetypes=[1,2],log_output=f'{self.folder}_log.txt')
                            ssvect.append(vect0)
                            ssdata.append(data0)
                            #save with starttime            
                            with lzma.open(f'{self.folder}/{file}_start_{s}_ss.pz','wb') as f:
                                pickle.dump([ssvect, ssdata, obs.param], f)

                        ssvect = torch.as_tensor(np.asarray((ssvect[0],ssvect[1])).reshape(1,len(ssvect[0]),2))
                        xout = torch.cat((xout, ssvect),0)                
                    else:
                        obs.param.framerate = self.framerate
                        ssvect, ssdata = allium.summstats.calculate_summary_statistics(obs,useall = self.useall,opts = self.ssopts,log = self.log, starttime=self.starttime[0], endtime=self.endtime[0],usetypes=[1,2],log_output=f'{self.folder}_log.txt')
                        
                        #save with starttime            
                        with lzma.open(f'{self.folder}/{file}_start_{self.starttime[0]}_ss.pz','wb') as f:
                            pickle.dump([ssvect, ssdata, obs.param], f)
                        ssvect = torch.as_tensor(np.asarray(ssvect).reshape(1,len(ssvect)))    
                        xout = torch.cat((xout, ssvect),0)                
                except Exception as e:
                    bad_output = f'{self.thetafilename}_badss.p'
                    print(f"Error: Exception raised during calculation of summary statistiscs. Output saved to {bad_output}")
                    print(e)
                    with open(bad_output,'wb') as f:
                        pickle.dump(obs, f)
                    pass          
        return torch.as_tensor(xout)

    def simulate(self, theta):
        """
        Main simulation function

        """
        def printOutput(t, tic, p, log=False):
            toc = time.perf_counter()
            message =  f"""-----------------------
                        \n Timestep: {t} \
                        \n # of Cells: {len(p)} \
                        \n Since last log {toc - tic[1]:.4} seconds \
                        \n Total Runtime: {toc - tic[0]:.4} seconds
                        \n -----------------------"""
            if log:
                return 0 
            else:
                print(f'{message}')        
                print("\033[9A")
                return 0
        def setHeterogenous(sim):
            pop = sim.popSize()
            popidx = []
            for i in range(pop): 
                popidx.append(i)
                
            sim.setCellTypes(capmd.VectorInt(popidx),i)

        def getPopulation(sim, neighbours = False):
            pop = sim.popSize()
            popidx = []
            for i in range(pop): 
                popidx.append(i)

            popId = np.array(sim.getPopulationId(capmd.VectorInt(popidx)))
            popPosn = np.array(sim.getPopulationPosition(capmd.VectorInt(popidx)))
            popVel = np.array(sim.getPopulationVelocity(capmd.VectorInt(popidx)))
            popTheta = np.array(sim.getPopulationTheta(capmd.VectorInt(popidx)))
            popRadius = np.array(sim.getPopulationRadius(capmd.VectorInt(popidx)))
            popType = np.array(sim.getPopulationType(capmd.VectorInt(popidx)))
            popZ = np.array(sim.getPopulationZ(capmd.VectorInt(popidx)))
            popArray = np.stack([popId,popPosn[:,0], popPosn[:,1],
                                 popVel[:,0], popVel[:,1],
                                 popTheta, popRadius ,
                                 popType,popZ], axis=1)

            if neighbours:
                print('Error: currently not implemented')
                #get neighbours here
            return popArray

        def updateParams(p, params,keys,tracers=0.1, log=False, dist = np.random.normal, rinit=False):
            setattr(params, 'log', log)
            if rinit:
                setattr(params,'initseed', np.random.randint(1000000))
            if not len(p):
                print("No parameters updated")
            else:
                #check to see if hierarchical
                if len(p) != len(keys):
                    newp = []
                    counter = 0
                    for i,k in enumerate(keys):
                        if self.multilevel[i] == 1:
                            newp.append([p[counter], p[counter+1]])
                            counter +=1
                        else:
                            newp.append(p[counter])
                        counter +=1
                    p = newp   
                for key, value in zip(keys[:len(p)],p):  
#                     value = value.tolist()
                    if not log:
                        print(f'{key} = {value}\n')
                    if (key == 'pairstiff') or (key == 'pairatt'):
                        if type(value) is list:
                              print("Error: this is not implemented. Consider how to combine different interactions")
#                             values = dist(value[0],value[1],(params.N,params.N)).tolist()
                        else:
                            values = [[value,value,value],[value,value,value],[value,value,value]]
                        setattr(params, key, values)
                    elif key == 'phi':
                        setattr(params, 'phi', value)
                    elif key == 'N':
                        setattr(params, 'N', int(value))
                        print(int(value.astype(int)*tracers))
                        setattr(params, 'Ntracer', int(value.astype(int)*tracers))
                    elif (key == 'deathrate') or (key == 'divrate'):
                        if type(value) is list:
                            values = dist(value[0],value[1],params.N).tolist()
                            values[1] = 0  #for tracers
                        else:
                            values = [value,0,value]
                    else: #e.g. factive, tau, alignment
                        if type(value) is list:
                            values = dist(value[0],value[1],params.N).tolist()
                            values = [abs(v) for v in values]
                        else:
                            values = [value,value,value]
                        setattr(params, key, values)
                if not log:
                    print("Parameters updated")
            return params

        def paramsFromFile(paramObj, fileName):
            paramObjCopy = paramObj
            try:
                with open(fileName) as jsonFile:
                    parameters = json.load(jsonFile)
                    for attribute in parameters:
                        setattr(paramObjCopy, attribute, parameters[attribute])
                    return paramObjCopy
            except Exception as e:
                print(e)
                print("Incorrect json format, returning default parameters")
                return paramObj
    
        if not self.log:
            print(f"# of parameters = {len(theta)}")            
        tic = time.perf_counter()
        tic2 = time.perf_counter()
        defaultparams = paramsFromFile(capmd.Parameters(), self.parameterFile)
        params = updateParams(theta, defaultparams, self.keys,log=self.log, rinit=self.rinit)
        sim = capmd.interface(params)
        timesteps = []
        x = []
        Rlength = params.Lx/4
        maxR = [ Rlength/2,  params.Ly]
        minR = [-Rlength/2, -params.Ly]
        popArray = []
#         setHeterogenous(sim)
        for t in range(params.t_final):
            sim.move()
            # Test for output
            if (t % params.output_time == 0): 
                p = getPopulation(sim)   
                printOutput(t, [tic, tic2], p,self.log)
                popArray.append(p)         
                if (params.output_type == 'all'):
                    sim.saveData("text")
                    sim.saveData("vtp")
                else:
                    sim.saveData(params.output_type)            
                tic2 = time.perf_counter()
                timesteps.append(t)

            # Test for scratch
            if (t == params.zaptime):
                p = getPopulation(sim)            
                zapList = []
                for i in range(sim.popSize()):
                    x = sim.getPopulationPosition(capmd.VectorInt([i]))[0]
                    if ((x[0] < maxR[0]) & (x[0] > minR[0])):
                        if ((x[1] < maxR[1]) & (x[1] > minR[1])):
                            idx = sim.getPopulationId(capmd.VectorInt([i]))[0]
                            zapList.append(idx)
                
                sim.killCells(capmd.VectorInt(zapList))
                if self.log:
                    print("\n"*10+"Cell zapping stage completed" + "\n"*2,end="", file=open(f'{self.folder}_log.txt', 'a'))
            # Test for population dynamicss
            if (t % params.popdynfreq == 0): 
                sim.populationDynamics(params.popdynfreq)
            
        d = {}
        for att in dir(params):
            if not(att.startswith('__')):
                d[att] =  getattr(params,att)
                
        d['theta'] = theta
        d['keys'] = self.keys
        self.counter +=1
        return allium.data.SimData(params=d, data=popArray, loadtimes = [0,int(params.t_final/params.output_time)])

