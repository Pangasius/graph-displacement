import os.path as osp
import glob

from torch_geometric.data import Dataset
from torch_geometric.nn import radius_graph, knn_graph
import torch

import numpy as np

import pickle
import lzma

import os

import random

import threading
#find /scratch/users/nstillman/data-cpp/train/ -name "*fast4p.p" -type f  | head | xargs du
"""
27804   ./p000_r004_sb015_2148010_b039_fast4p.p                                                                         
33133   ./p013_r000_sb009_2147623_b071_fast4p.p                                                                         
9628    ./p056_r000_sb040_2147623_b132_fast4p.p                                                                         
23679   ./p029_r003_sb006_2148010_b038_fast4p.p                                                                         
16676   ./p030_r000_sb038_2147623_b119_fast4p.p                                                                         
39793   ./p019_r001_sb013_2147250_b028_fast4p.p                                                                         
21402   ./p023_r000_sb018_2147623_b131_fast4p.p                                                                         
22863   ./p001_r002_sb001_2148010_b026_fast4p.p                                                                         
38547   ./p040_r000_sb000_2147623_b082_fast4p.p                                                                         
23422   ./p058_r000_sb020_2147623_b134_fast4p.p
"""

class CellGraphDataset(Dataset):
    def __init__(self, root, max_size, transform=None, pre_transform=None, pre_filter=None, rdts=False, inmemory=False, bg_load=False, wrap=False, T_limit=0):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        #path to all the files
        self.paths = self.each_path(root, max_size)
        
        #in memory 
        self.memorize = inmemory
        self.memory = {}
        
        self.attributes = ["distance_x", "distance_y", "distance_z",\
                            "degree", "velocity_x", "velocity_y", "velocity_z",\
                            "epsilon", "tau", "v0", \
                            "avg_radius", "radius"]
        
        assert(inmemory | (not bg_load)) # bg load can't be true alone
        
        self.bg_load = bg_load
        self.bg_load_running = False
        self.waiting_for = -1
        self.thread = None
        
        self.T_limit = T_limit
        
    def _download(self):
        pass

    def _process(self):
        pass

    class each_path():
        def __init__(self, root, max_size):
            self.root = root
            self.max_size = max_size
            self._each_path = self.read()
            
        def read(self):
            relative = [s.replace('\\', '/') for s in glob.glob(str(self.root) + '/*fast4p.p*')]
            max_size = min(self.max_size, len(relative))
            relative = relative[:max_size]
            absolute = [osp.abspath(s) for s in relative]
            return absolute

        def fset(self, value):
            self._each_path = value
        
        def fget(self):
            return self._each_path
        
    def process_file(self, path):
        if self.memory.get(path) is None :
            if path.endswith(".p") :
                with open(path, 'rb') as f:
                    x = pickle.load(f)
                    
            elif path.endswith(".pz") :
                with lzma.open(path, 'rb') as f:
                    x = pickle.load(f) # type: ignore    
                
            else :
                raise ValueError("File type not supported for path: " + path)
            
        else :
            return self.memory[path]
            
        # Parameters of interest: 
        #Attraction force: 
        epsilon = x.param.pairatt[0][0]
        # Persistence timescale 
        tau = x.param.tau[0]
        # Active force
        v0 = x.param.factive[0]

        #cutoff distance defines the interaction radius. You can assume below:
        #cutoff = 2*(x.param.cutoffZ + 2*x.param.pairatt[0][0])
        #Get position data
        rval = torch.tensor(x.rval)
        
        #normalize the data
        
        border = [rval[:,:,0].min(), rval[:,:,0].max(), rval[:,:,1].min(), rval[:,:,1].max()]
        
        #so we want to wrap around so we need the max and min be at 0,1
        rval = (rval - rval.min(dim=0)[0].min(dim=0)[0]) / (rval.max(dim=0)[0].max(dim=0)[0] - rval.min(dim=0)[0].min(dim=0)[0])
        
        #Get time and number of cells from shape of position data
        old_T = rval.shape[0]
        if self.T_limit :
            T = min(rval.shape[0], self.T_limit) #limiting for testing
        else :
            T = rval.shape[0]
        N = rval.shape[1]
        
        try :
            assert(T > 1)
            assert(N > 1)
        except :
            raise Exception("Data unfit for use")
            
        rval = rval[:T,:N,:2]
        
        #ideally we would like to only have those connections but radius_graph doesn't work on GPU for some reason
        batch = torch.arange(T).repeat_interleave(N).to(torch.long)
        
        #keep in mind there are distorsions
        rval = self.to_torus(rval[:,:,0], rval[:,:,1])
        
        #we will add to rval dx and dy to get the velocity
        rval = torch.cat((rval, torch.zeros((T, N, 3))), dim=2)
        for t in range(T-1):
            rval[t+1, :, 3:6] = rval[t+1, :, :3] - rval[t, :, :3]
        
        edge_index, edge_attr, batch_edge = self.get_edges(rval, batch, tau, epsilon, v0, x.param.R, x.radius)
        
        if self.memorize :
            self.memory[path] = (rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), batch_edge.to(torch.long), border)
        
        return rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), batch_edge.to(torch.long), border

    #although There is no local isometry (i.e. a map that preserves local distances and angles) between the torus and the unit square, or any flat domain for that matter. This is a consequence of Gauss's Theorema Egregium and the fact that the Gaussian curvature of the torus is not everywhere zero. 
    #we can use the relations 
    #x=scos(2πu/s)/w
    #y=ssin(2πu/s)/w
    #z=sin(2πv)/w
    #where w=s2+1−−−−−√−cos(2πv)
    def to_torus(self,u, v):
        s = torch.tensor(1.0) # is a 1x1x1 rectangle
        w = torch.sqrt(s**2 + 1) - torch.cos(2 * np.pi * v)
        return torch.stack((s * torch.cos(2 * np.pi * u / s) / w, s * torch.sin(2 * np.pi * u / s) / w, torch.sin(2 * np.pi * v) / w), dim=2)

    #can be inverted to 
    # #u=(s/2π)atan(y/x)
    # v=sign(z)/2π * acos((z^2 * sqrt(s^2+1) ± sqrt(1−z^2 * s^2)) / sqrt(z^2+1))
    def from_torus(self,x, y, z):
        s = torch.tensor(1.0) # is a 1x1x1 rectangle
        u = (s / (2 * np.pi)) * torch.atan(y / x)
        v = (torch.sign(z) / (2 * np.pi)) * torch.acos((z**2 * torch.sqrt(s**2 + 1) - torch.sqrt(1 - z**2 * s**2)) / torch.sqrt(z**2 + 1))
        return torch.stack((u, v), dim=2)
    
    def get_edges(self, rval, batch, tau, epsilon, v0, r, radius):
        T = rval.shape[0]
        N = rval.shape[1]
        
        max_degree = 12
        
        #we want to project our (theta, phi) coordinates onto a sphere, so we need to convert them to (x, y, z) coordinates
        #we can do this by using the following equations:
        #x = r*sin(theta) * cos(phi)
        #y = r*sin(theta) * sin(phi)
        #z = r*cos(theta)
        #we can do this in one line of code
        
        rval = rval.reshape(T*N, -1)
        
        edge_index = knn_graph(rval, k = max_degree, batch=batch, loop=False)
        
        edge_attr = (rval[edge_index[0, :], :3] - rval[edge_index[1, :], :3]).reshape(-1,3)
        
        #assumes they are perfectly split already so we need only to check one node for each pair
        batch_edge = edge_index[0, :] // N

        #we will also add the degree of the first node of the pair
        #degree of each node
        deg = torch.zeros((T, N))
        edge_index_local = torch.zeros_like(edge_index)
        for t in range(T):
            edge_index_local[:, batch_edge == t] = edge_index[:, batch_edge == t] - N*t
            deg[t, :] = torch.bincount(edge_index_local[0, batch_edge == t], minlength=N) / max_degree
            
        #degree of the first node of the pair
        deg = deg.reshape(T*N)
        edge_attr = torch.cat((edge_attr.reshape(-1, 3), deg[edge_index[0, :]].reshape(-1, 1)), dim=1).reshape(-1,4)
        
        #also the difference of velocities
        rval = rval.reshape(T,N,-1)
        edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.shape[0], 3))), dim=1)
        for t in range(T):
            edge_attr[batch_edge == t, -3:] = rval[t, edge_index_local[0, batch_edge == t], 3:6] - rval[t, edge_index_local[1, batch_edge == t], 3:6]

        #add tau, epsilon and v0
        edge_attr = torch.cat((edge_attr, torch.ones((edge_attr.shape[0], 1))*tau), dim=1)
        edge_attr = torch.cat((edge_attr, torch.ones((edge_attr.shape[0], 1))*epsilon), dim=1)
        edge_attr = torch.cat((edge_attr, torch.ones((edge_attr.shape[0], 1))*v0), dim=1)
        
        #to that we will add the average radius given by x.R and the particular radius given by x.radius
        edge_attr = torch.cat((edge_attr, torch.ones((edge_attr.shape[0], 1))*r), dim=1)
        radii = (torch.tensor(radius[:T,:N].reshape(T*N)[edge_index[0, :]]) - r)
        edge_attr = torch.cat((edge_attr, radii.reshape(-1, 1)), dim=1)
        
        #also add the polydispersity
        #edge_attr = torch.cat((edge_attr, torch.ones((edge_attr.shape[0], 1))*poly), dim=1)
        assert(edge_attr.shape[1] == 12)
        
        return edge_index, edge_attr, batch_edge
    
    def len(self):
        return len(self.paths.fget())
    
    def load_all(self) :
        for i in reversed(range(self.len())) :
            if self.memory.get(self.paths.fget()[i]) is None :
                self.waiting_for = i
                self.process_file(self.paths.fget()[i])
            else :
                self.waiting_for = -2
                break

    def get(self, idx):
        if self.bg_load and not self.bg_load_running :
            self.bg_load_running = True
            self.thread = threading.Thread(target=self.load_all)
            self.thread.start()
            
        if self.thread != None :
            if self.waiting_for <= idx :
                self.thread.join()
            
        return self.process_file(self.paths.fget()[idx])
        
    def _dump_source(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.paths.fget(), f)
            
    def _overwrite_source(self, path):
        with open(path, 'rb') as f:
            #set the property to the new list of paths
            self.paths.fset(pickle.load(f))
            
    def save_or_load_if_exists(self, path):
        if "sources" not in os.listdir():
            os.mkdir("sources")

        if path not in os.listdir("sources"):
            #first time running, dump the paths to a pickle file
            self._dump_source("sources/" + path)
        else :
            #overwrite the paths to the previous configuration
            self._overwrite_source("sources/" + path)