import pickle

import sys
from cell_dataset import loadDataset

sys.path.append('/home/nstillman/1_sbi_activematter/cpp_model')
try :
    import allium
except :
    print("Could not import allium")
    
load_all =  False #load directly from a pickle
pre_separated = False #if three subfolders already exist for train test and val

override = False #make this true to always use the same ones

for extension in ["_open_ht_hv", "_open_lt_hv", "_open_lt_lv", "_open_ht_lv"] :
    data_train, data_test, data_val = loadDataset(load_all, extension, pre_separated, override)

    data_train.thread = None
    for i in range(data_train.len()) :
        data_point = data_train.get(i, -1)
        
    with open("data/training" + extension + ".pkl", 'wb') as f:
        pickle.dump(data_train, f)
        
    data_test.thread = None
    for i in range(data_test.len()) :
        data_point = data_test.get(i, -1)
        
    with open("data/testing" + extension + ".pkl", 'wb') as f:
        pickle.dump(data_test, f)
        
    data_val.thread = None
    for i in range(data_val.len()) :
        data_point = data_val.get(i, -1)
        
    with open("data/validation" + extension + ".pkl", 'wb') as f:
        pickle.dump(data_val, f)
    print("Saved datasets")