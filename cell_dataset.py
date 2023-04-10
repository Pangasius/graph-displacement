import os.path as osp
import glob

from torch_geometric.data import Dataset
from torch_geometric.nn import radius_graph, radius_graph, knn_graph
import torch

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
    def __init__(self, root : str | list[str], max_size : int , transform=None, pre_transform=None, pre_filter=None, inmemory=False, bg_load=False, wrap=False, T_limit=0):
        super().__init__(root, transform, pre_transform, pre_filter) # type: ignore 
       
        #path to all the files
        self.paths = self.each_path(root, max_size)

        #in memory 
        self.memorize = inmemory
        self.memory = {}
        
        self.attributes = ["distance_x", "distance_y"]
        
        assert(inmemory | (not bg_load)) # bg load can't be true alone
        
        self.bg_load = bg_load
        self.bg_load_running = False
        self.waiting_for = -1
        self.thread = None
        
        self.T_limit = T_limit
        
        self.max_degree = 10
        
        self.wrap = wrap
        
    def _download(self):
        pass

    def _process(self):
        pass

    class each_path():
        def __init__(self, root : str | list[str], max_size : int):
            self.root = root
            self.max_size = max_size
            
            if type(root) is str:
                self._each_path = self.read()
            else :
                self._each_path = root
            
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
        
    def process_file(self, path : str):
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
        cutoff = 2*(x.param.cutoffZ + 2*x.param.pairatt[0][0])
        #Get position data
        rval = torch.tensor(x.rval)
        
        #normalize the data
        
        border = torch.tensor([rval[:,:,0].min(), rval[:,:,0].max(), rval[:,:,1].min(), rval[:,:,1].max()])
        
        mean = torch.tensor([rval[:,:,0].mean(), rval[:,:,1].mean()])
        std = torch.tensor([rval[:,:,0].std(), rval[:,:,1].std()])
        
        if self.wrap :
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
        
        rval, edge_index, edge_attr = self.get_edges(rval, self.max_degree, wrap=self.wrap, T=T, N=N)

        #additional parameters : tau, epsilon, v0, r, dt, framerate
        params = torch.tensor([tau, epsilon, v0, x.param.R, x.param.dt, x.param.output_time])
        
        params = torch.cat((params, mean, std, torch.tensor([cutoff])), dim=0).to(torch.float)
        
        if self.memorize :
            self.memory[path] = (rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), border, params)

        return rval.to(torch.float), edge_index.to(torch.long), edge_attr.to(torch.float), border, params
    
    @staticmethod
    def get_edges(rval : torch.Tensor, max_degree : int, wrap : bool, T : int, N : int) :

        if not wrap :
            batch = torch.arange(T).repeat_interleave(N).to(torch.long)
        
            rval = rval.reshape(T*N, -1)
            
            #edge_index = radius_graph(rval, r = cutoff, batch=batch, loop=False, max_num_neighbors=max_degree)
            edge_index = knn_graph(rval, k = max_degree, batch=batch, loop=True, flow="source_to_target")
            
            edge_attr = (rval[edge_index[0, :], :2] - rval[edge_index[1, :], :2]).reshape(-1,2)
        else :
            #if the world wraps around itself, we need to copy the rval and translate it by 1 in each direction
            #we will then use the radius_graph to find the nearest neighbors

            rval_expanded = torch.cat((rval, rval + torch.tensor([1, 0]), rval - torch.tensor([1, 0]), rval + torch.tensor([0, 1]), rval - torch.tensor([0, 1]), rval + torch.tensor([1, 1]), rval - torch.tensor([1, 1]), rval + torch.tensor([1, -1]), rval - torch.tensor([1, -1])), dim=1)
        
            rval_expanded = rval_expanded.reshape(T*N*9, -1)
            
            batch = torch.arange(T).repeat_interleave(N * 9).to(torch.long)
        
            #edge_index = radius_graph(rval_expanded, r = cutoff, batch=batch, loop=False, max_num_neighbors=max_degree, num_workers=2)
            edge_index = knn_graph(rval_expanded, k = max_degree, batch=batch, loop=True, flow="source_to_target", num_workers=2)
        
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
            
        #add the degree of each node using the edge_index
        rval = torch.cat((rval, torch.zeros((T, N, 1), dtype=torch.float)), dim=2)
        rval[:,:,4] = torch.bincount(edge_index[0, :], minlength=T*N).reshape(T, N)
        
        
        return rval, edge_index, edge_attr
    
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
                self.thread=None
            
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
            
    def to(self, device):
        for i in range(self.len()) :
            rval, edge_index, edge_attr, border, params, cutoff = self.memory[self.paths.fget()[i]]
            border = torch.tensor(border)
            self.memory[self.paths.fget()[i]] = (rval.to(device), edge_index.to(device), edge_attr.to(device), border.to(device), params.to(device), cutoff)

    @staticmethod
    def merge_datasets(datasets):
        paths = []
        memory = {}
        for dataset in datasets :
            paths += dataset.paths.fget()
            memory = {**memory, **dataset.memory}
            
        dataset = datasets[0]
            
        dataset.paths.fset(paths)
        dataset.memory = memory
        
        return dataset
            

def load(load_all = True, suffix = "", pre_separated = False, override = False) -> tuple[CellGraphDataset, CellGraphDataset, CellGraphDataset]:
    """_summary_

    Args:
        load_all (bool, optional): load directly from a pickle.
        pre_separated (bool, optional): if three subfolders already exist for train test and val.
        override (bool, optional): make this true to always use the same ones.
    """
    
    data_train, data_test, data_val = None, None, None

    if load_all : 
        if os.path.exists("data/training" + suffix + ".pkl") :
            with open("data/training" + suffix + ".pkl", "rb") as f:
                data_train = pickle.load(f)
        else :
            print("Training data not found")
            
        if os.path.exists("data/testing" + suffix + ".pkl") :
            with open("data/testing" + suffix + ".pkl", "rb") as f:
                data_test = pickle.load(f)
        else :
            print("Test data not found")
            
        if os.path.exists("data/validation" + suffix + ".pkl") :
            with open("data/validation" + suffix + ".pkl", "rb") as f:
                data_val = pickle.load(f)
        else :
            print("Validation data not found")
    else :
        if pre_separated :
            path = "/scratch/users/nstillman/data-cpp/" 
            
            data_train = CellGraphDataset(root=path + 'train', max_size=1000, inmemory=True, bg_load=True, wrap=True, T_limit=16)
            print("Training data length : ", data_train.len())

            data_test = CellGraphDataset(root=path + 'test', max_size=50, inmemory=True, bg_load=True, wrap=True, T_limit=16)
            print("Test data length : ", data_test.len())
            
            data_val = CellGraphDataset(root=path + 'valid', max_size=50, inmemory=True, bg_load=True, wrap=True, T_limit=8)
            print("Validation data length : ", data_val.len())
        else :
            path = "/scratch/users/nstillman/open/high_tau_high_v0/"
            
            data_train, data_test, data_val =  extract_train_test_val(path, max_size=1000, inmemory=True, bg_load=True, wrap=False, T_limit=0)

        if override :
            data_train.save_or_load_if_exists("train_paths.pkl")
            data_test.save_or_load_if_exists("test_paths.pkl")
            data_val.save_or_load_if_exists("val_paths.pkl")
        else :
            torch.autograd.set_detect_anomaly(True) #type: ignore
            
    return data_train, data_test, data_val #type: ignore #up to the user to make sure they are ok

def single_overfit_dataset(data_train : CellGraphDataset, data_test : CellGraphDataset) :
    """
    creates a single data point data set in data_train, and copies it to data_test
    """
    data_train.paths.fset([data_train.paths.fget()[0]])
    data_test.paths.fset([data_train.paths.fget()[0]])
    data = data_train.memory[data_train.paths.fget()[0]]
    data_train.memory = {data_train.paths.fget()[0] : data}
    data_test.memory = {data_test.paths.fget()[0] : data}
    
    return data_train, data_test

def extract_train_test_val(root, max_size, inmemory=False, bg_load=False, wrap=False, T_limit=0):
    """
    root: root directory
    train_size: number of training examples
    test_size: number of test examples
    val_size: number of validation examples
    max_size: max number of examples to use
    inmemory: if true, load all data into memory
    bg_load: if true, load data in the background
    wrap: if true, wrap the data
    T_limit: if > 0, limit the number of time steps to T_limit
    """
    #get all the files
    relative = [s.replace('\\', '/') for s in glob.glob(str(root) + '/*interaction.p*')]
    max_size = min(max_size, len(relative))
    relative = relative[:max_size]
    absolute = [osp.abspath(s) for s in relative]
    absolute.sort(key=hash)
    
    #split into train, test, val
    train = absolute[:int(0.8*len(absolute))]
    test = absolute[int(0.8*len(absolute)):int(0.9*len(absolute))]
    val = absolute[int(0.9*len(absolute)):]
    
    #create the datasets
    train_dataset = CellGraphDataset(train, len(train), inmemory=inmemory, bg_load=bg_load, wrap=wrap, T_limit=T_limit)
    test_dataset = CellGraphDataset(test, len(test), inmemory=inmemory, bg_load=bg_load, wrap=wrap, T_limit=T_limit)
    val_dataset = CellGraphDataset(val, len(val), inmemory=inmemory, bg_load=bg_load, wrap=wrap, T_limit=T_limit)
    
    return train_dataset, test_dataset, val_dataset
