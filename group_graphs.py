import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt

epoch = 51

extensions = ["_open_ht_hv", "_open_lt_hv", "_open_lt_lv", "_open_ht_lv"]
number_of_messages = [1,2,3,4]
size_of_messages = [32, 64, 128, 256, 512, 1024]
distribs = ["laplace", "normal"]
out_channels = [4,8]
horizons = [1,2,3,4,5]

base_extension = ["_open_lt_hv"]
base_number_of_message = [4]
base_size_of_message = [128]
base_distrib = ["laplace"]
base_out = [8]
base_horizon = [5]

base_entry = {"extension": base_extension,
                            "number_of_messages": base_number_of_message,
                            "size_of_messages": base_size_of_message,
                            "distribution": base_distrib,
                            "out_channels": base_out,
                            "horizon": base_horizon}

def fetch_data(entry, prefix, suffix) :
    name_complete = entry["extension"][0] + "_" + str(entry["number_of_messages"][0]) + "_" + str(entry["size_of_messages"][0]) + "_" + entry["distribution"][0] + "_" + str(entry["out_channels"][0]) + "_h" + str(entry["horizon"][0])
    
    path_name = "models/new_model/out_" + str(entry["out_channels"][0]) + "_eps_-4/" + entry["distribution"][0] + "/" + entry["extension"][0][6:] + "/h" + str(entry["horizon"][0]) + "/"
    
    if suffix == ".npy" :
        data = np.load(path_name + prefix + name_complete + suffix)
    elif suffix == ".pkl" :
        with open(path_name + prefix + name_complete + suffix, "rb") as f:
            data = pkl.load(f)
            data = [data, pkl.load(f)]
    else :
        print("suffix not recognized")
        return None
        
    return data

def compile_msd(part_of_interest, values) :
    tval = values[list(values.keys())[0]][0]
    msd_y_mean = values[list(values.keys())[0]][1]
    
    plt.loglog(tval,msd_y_mean,'b.-',lw=2, label='Synthetic data')
    plt.loglog(tval,msd_y_mean[1]/(1.0*tval[1])*tval,'--',lw=2,color="cyan")
    
    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        msd_x_mean = values[list(values.keys())[i]][3]
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        plt.loglog(tval,msd_x_mean,'.-',lw=2, label=label)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Mean square displacement')
    plt.title('Mean square displacement over time')
    plt.legend()

    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_msd.png")
    
    plt.close()
    
def compile_mv(part_of_interest, values) :
    xval = values[list(values.keys())[0]][0]
    vav_y_mean = values[list(values.keys())[0]][1]
    vav_y_std = values[list(values.keys())[0]][2]
    
    plt.plot(xval,vav_y_mean,'b.-',lw=2, label='Synthetic data')
    plt.fill_between(xval, vav_y_mean-vav_y_std, vav_y_mean+vav_y_std, alpha=0.2, color='b')
    
    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        vav_x_mean = values[list(values.keys())[i]][3]
        vav_x_std = values[list(values.keys())[i]][4]
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        plt.plot(xval,vav_x_mean,'.-',lw=2, label=label)
        plt.fill_between(xval, vav_x_mean-vav_x_std, vav_x_mean+vav_x_std, alpha=0.2)
        
    plt.xlabel('Time (hours)')
    plt.ylabel('Mean velocity')
    plt.title('Mean velocity over time')
    plt.legend()
    
    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_mv.png")
    
    plt.close()
    
def compile_svcd(part_of_interest, values) :
    velbins2 = values[list(values.keys())[0]][0]
    vdist2_y_mean = values[list(values.keys())[0]][1]
    vdist2_y_std = values[list(values.keys())[0]][2]
    
    plt.semilogx(velbins2,vdist2_y_mean,'b.-',lw=2, label='Synthetic data')
    plt.fill_between(velbins2,vdist2_y_mean+vdist2_y_std,vdist2_y_mean-vdist2_y_std,color='b',alpha=0.2)
    
    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        vdist2_x_mean = values[list(values.keys())[i]][3]
        vdist2_x_std = values[list(values.keys())[i]][4]
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        plt.semilogx(velbins2,vdist2_x_mean,'.-',lw=2, label=label)
        plt.fill_between(velbins2,vdist2_x_mean+vdist2_x_std,vdist2_x_mean-vdist2_x_std,alpha=0.2)
        
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity component distribution')
    plt.legend()
    
    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_svcd.png")
    
    plt.close()
    
def compile_svmd(part_of_interest, values) :
    velbins = values[list(values.keys())[0]][0]
    vdists_y_mean = values[list(values.keys())[0]][1]
    vdists_y_std = values[list(values.keys())[0]][2]
    
    plt.semilogx(velbins,vdists_y_mean,'b.-',lw=2, label='Synthetic data')
    plt.fill_between(velbins,vdists_y_mean+vdists_y_std,vdists_y_mean-vdists_y_std,color='b',alpha=0.2)
    
    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        vdists_x_mean = values[list(values.keys())[i]][3]
        vdists_x_std = values[list(values.keys())[i]][4]
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        plt.semilogx(velbins,vdists_x_mean,'.-',lw=2, label=label)
        plt.fill_between(velbins,vdists_x_mean+vdists_x_std,vdists_x_mean-vdists_x_std,alpha=0.2)
        
    plt.xlabel('v/<v>')
    plt.ylabel('P(v/<v>)')
    plt.title('Scaled velocity magnitude distribution')
    plt.legend()
    
    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_svmd.png")
    
    plt.close()
    
def compile_losses(part_of_interest, values) :
    f = plt.figure()

    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        plt.loglog(values[list(values.keys())[i]][0]["loss"], label=label, color=color)
        
        #plt.loglog(values[list(values.keys())[i]][0]["loss_mean"], linestyle=':', color=color)
        
        #plt.loglog(np.exp(values[list(values.keys())[i]][0]["loss_log"]), linestyle='-.', color=color)
    
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.title('Testing Loss over testing samples')
    plt.legend()
    
    #move the graph slightly to the right for the ylabel to be in the figure
    plt.subplots_adjust(left=0.15)
    
    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_testing_losses.png")
    
    plt.close(f)
    
    f = plt.figure()
    
    for i in range(len(part_of_interest[list(part_of_interest.keys())[0]])) :
        label = list(part_of_interest.keys())[0] + " = " + str(part_of_interest[list(part_of_interest.keys())[0]][i])
        plt.loglog(values[list(values.keys())[i]][1]["loss"], label=label)
    
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.title('Training Loss over training samples')
    plt.legend()
    
    plt.savefig("models/summary/" + list(part_of_interest.keys())[0] + "_training_losses.png")
    
    plt.close(f)

def compile_values(part_of_interest, values, to_analyse) :
    if to_analyse == "msd" :
        compile_msd(part_of_interest, values)
    elif to_analyse == "mv" :
        compile_mv(part_of_interest, values)
    elif to_analyse == "svcd" :
        compile_svcd(part_of_interest, values)
    elif to_analyse == "svmd" :
        compile_svmd(part_of_interest, values)
    elif to_analyse == "losses" :
        compile_losses(part_of_interest, values)
    else :
        print("to_analyse not recognized")
        return None

def group(entry) :
    part_of_interest = {}
    values = {}
    
    to_analyse = ["msd", "mv", "svcd", "svmd", "losses"]
    to_analyse_suffix = [".npy"] * 4 + [".pkl"]
    
    for a in range(len(to_analyse)):
        
        #iterate over the entry which has more than one value
        for i in range(len(entry)):
            if len(entry[list(entry.keys())[i]]) == 1:
                continue
            
            part_of_interest.update({list(entry.keys())[i]: list(entry.values())[i]})
            
            for j in range(len(entry[list(entry.keys())[i]])):
                #create a new entry with the same values as the base entry
                new_entry = base_entry.copy()
                #replace the value of the entry with the value of the current iteration
                new_entry.update({list(entry.keys())[i]: [list(entry.values())[i][j]]})
                #launch the job with the new entry
                data = fetch_data(new_entry, to_analyse[a], to_analyse_suffix[a])
                
                values.update({list(entry.values())[i][j]: data})
                
        compile_values(part_of_interest, values, to_analyse[a])
    
num_messages_entry = base_entry.copy()
num_messages_entry.update({"number_of_messages": number_of_messages})
group(num_messages_entry)

size_messages_entry = base_entry.copy()
size_messages_entry.update({"size_of_messages": size_of_messages})
group(size_messages_entry)

distrib_entry = base_entry.copy()
distrib_entry.update({"distribution": distribs})
group(distrib_entry)

out_entry = base_entry.copy()
out_entry.update({"out_channels": out_channels})
group(out_entry)

horizon_entry = base_entry.copy()
horizon_entry.update({"horizon": horizons})
group(horizon_entry)