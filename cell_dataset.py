import os.path as osp
import glob

from torch_geometric.data import Dataset
from torch_geometric.nn import radius_graph, knn_graph
import torch

import numpy as np

import pickle
import lzma

import os

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
        super().__init__(root, transform, pre_transform, pre_filter) # type: ignore 
       
        #path to all the files
        self.paths = self.each_path(root, max_size)
        
        #in memory 
        self.memorize = inmemory
        self.memory = {}
        
        self.attributes = ["distance_x", "distance_y",\
                            "degree",\
                            "avg_radius"]
        
        assert(inmemory | (not bg_load)) # bg load can't be true alone
        
        self.bg_load = bg_load
        self.bg_load_running = False
        self.waiting_for = -1
        self.thread = None
        
        self.T_limit = T_limit
        
        self.max_degree = 5
        
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
        

        rval, edge_index, edge_attr = self.get_edges(rval, self.max_degree, wrap=True, T=T, N=N)
        edge_attr, batch_edge = self.get_edges_attributes(edge_index, edge_attr, self.max_degree, x.param.R, x.radius, T=T, N=N)

        #additional parameters : tau, epsilon, v0, r
        params = torch.tensor([tau, epsilon, v0, x.param.R]).to(torch.float)
        
        if self.memorize :
            self.memory[path] = (rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), batch_edge.to(torch.long), border, params)

        return rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), batch_edge.to(torch.long), border, params
    
    @staticmethod
    def get_edges(rval : torch.Tensor, max_degree : int, wrap : bool, T : int, N : int) :

        if not wrap :
            batch = torch.arange(T).repeat_interleave(N).to(torch.long)
        
            rval = rval.reshape(T*N, -1)
            
            edge_index = knn_graph(rval, k = max_degree, batch=batch, loop=False)
            
            edge_attr = (rval[edge_index[0, :], :2] - rval[edge_index[1, :], :2]).reshape(-1,2)
        else :
            #if the world wraps around itself, we need to copy the rval and translate it by 1 in each direction
            #we will then use the knn_graph to find the nearest neighbors

            rval_expanded = torch.cat((rval, rval + torch.tensor([1, 0]), rval - torch.tensor([1, 0]), rval + torch.tensor([0, 1]), rval - torch.tensor([0, 1]), rval + torch.tensor([1, 1]), rval - torch.tensor([1, 1]), rval + torch.tensor([1, -1]), rval - torch.tensor([1, -1])), dim=1)
        
            rval_expanded = rval_expanded.reshape(T*N*9, -1)
            
            batch = torch.arange(T).repeat_interleave(N * 9).to(torch.long)
        
            edge_index = knn_graph(rval_expanded, k = max_degree, loop=False, batch=batch, num_workers=2)
        
            edges_final = torch.zeros((2, 0), dtype=torch.long)
            for j in range(T) :
                edge_mask = ((edge_index[0, :] < (j+1)*9*N) | (edge_index[1, :] < (j+1)*9*N)) & ((edge_index[0, :] >= j*9*N) | (edge_index[1, :] >= j*9*N))
                edge_index_ = (edge_index[:, edge_mask] % (N)) + j*N
                edge_index_ = torch.unique(edge_index_, dim=1)

                edges_final = torch.cat((edges_final, edge_index_), dim=1)
            edge_index = edges_final
            
            rval = rval.reshape(T*N, -1)
    
            edge_attr = (rval[edge_index[0, :], :2] - rval[edge_index[1, :], :2])

            #we modify the edge_attr to account for the wrapping by taking the minimum distance between the two points
            #we can do this by taking the absolute value of the edge_attr and then subtracting 1 if the edge_attr is greater than 0.5
            #we can do this in one line of code

            edge_attr = edge_attr - (edge_attr > 0.5).float() + (edge_attr < -0.5).float()
        
        #we will add to rval the difference of positions from a time step to the next
        rval = rval.reshape(T, N, -1)
        rval = torch.cat((rval, torch.zeros_like(rval)), dim=2)
        rval[1:,:,2:] = rval[1:,:,:2] - rval[:-1,:,:2]
        
        if wrap :
            #substract 1 to the speeds if they are greater than 0.5
            rval[:,:,2:] = rval[:,:,2:] - (rval[:,:,2:] > 0.5).float() + (rval[:,:,2:] < -0.5).float()
        
        return rval, edge_index, edge_attr
    
    @staticmethod
    def get_edges_attributes(edge_index : torch.Tensor, edge_attr : torch.Tensor, max_degree : int, r : int | float, radius : torch.Tensor, T : int, N : int):
        
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
        edge_attr = torch.cat((edge_attr.reshape(-1, 2), deg[edge_index[0, :]].reshape(-1, 1)), dim=1).reshape(-1,3)

        #radius of each cell
        radii = (torch.tensor(radius[:T,:N].reshape(T*N)[edge_index[0, :]]) / r)
        edge_attr = torch.cat((edge_attr, radii.reshape(-1, 1)), dim=1)
        
        return edge_attr, batch_edge
    
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