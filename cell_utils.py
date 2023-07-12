import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
import numpy as np

class GraphingLoss():
    def __init__(self):
        pass

    def plot_losses(self, title : str, data : dict[str, list[float]], length : int, extension = "") :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if "loss_mean" in data :
            l_means = np.log(data["loss_mean"]).reshape(-1, length)
            l_means_mean = np.mean(l_means, axis=1)
            l_means_std = np.std(l_means, axis=1)
            
            #plot the means and the std as shaded areas
            ax.plot(l_means_mean, label="log(loss mean)", color="red")
            ax.fill_between(np.arange(len(l_means_mean)), l_means_mean - l_means_std, l_means_mean + l_means_std, alpha=0.2, color="red")
        
        if "loss_log" in data :
            l_logs = np.array(data["loss_log"]).reshape(-1, length)
            l_logs_mean = np.mean(l_logs, axis=1)
            l_logs_std = np.std(l_logs, axis=1)
            
            ax.plot(l_logs_mean, label="loss log", color="green")
            ax.fill_between(np.arange(len(l_logs_mean)), l_logs_mean - l_logs_std, l_logs_mean + l_logs_std, alpha=0.2, color="green")
        
    
        l = np.array(data["loss"]).reshape(-1, length)
        l_mean = np.mean(l, axis=1)
        l_std = np.std(l, axis=1)
        
        ax.plot(l_mean, label="loss", color="blue")
        ax.fill_between(np.arange(len(l_mean)), l_mean - l_std, l_mean + l_std, alpha=0.2, color="blue")
        
        ax.legend(loc="upper left") 
        ax.set_title(title)
        fig.canvas.draw()

        plt.savefig("models/Losses" + extension + title.replace(" ", "") + ".pdf", format="pdf")
        plt.close()
        
    def plot_params(self, params_out : list[dict[str, torch.Tensor]], params_true : list[dict[str, torch.Tensor]], epoch : int = 0, extension = "") :
        #unwind the parameters into a numpy array
        #first we extract the keys which will be the title of the graph
        keys = list(params_out[0])
        
        #all lengths should be the same
        length_inside_keys = params_out[0].get(keys[0]).size()[0] #type: ignore

        #then we can add the values to the array
        values_out = np.zeros((len(params_out), len(keys), length_inside_keys))
        for i, param in enumerate(params_out) :
            for j, key in enumerate(keys) :
                values_out[i, j] = param[key]
                
        values_true = np.zeros((len(params_true), len(keys), length_inside_keys))      
        for i, param in enumerate(params_true) :
            for j, key in enumerate(keys) :
                values_true[i, j] = param[key]
                
        #we will compute the mean and std of the value along the first dimension
        values_out_mean = values_out.mean(axis=0)
        values_out_std = values_out.std(axis=0)
        
        values_true_mean = values_true.mean(axis=0)
        values_true_std = values_true.std(axis=0)
                

        fig, axs = plt.subplots(len(keys) // 2, 2, layout='constrained', figsize=(10, 4))
        
        for i, key in enumerate(keys) :
            #we will plot the mean and std as a shaded area
            axs.flat[i].plot(values_true_mean[i], label="True " + key)
            axs.flat[i].fill_between(np.arange(length_inside_keys), values_true_mean[i] - values_true_std[i], values_true_mean[i] + values_true_std[i], alpha=0.5)
            
            axs.flat[i].plot(values_out_mean[i], label="Predicted " + key)
            axs.flat[i].fill_between(np.arange(length_inside_keys), values_out_mean[i] - values_out_std[i], values_out_mean[i] + values_out_std[i], alpha=0.5)
                
            axs.flat[i].legend(loc="upper left", fontsize="xx-small")
            
        fig.canvas.draw()
        
        fig.suptitle('Parameters evolution at epoch {:d}'.format(epoch), fontsize='xx-large')
        
        plt.savefig("models/Params{:s}_{:d}.pdf".format(extension, epoch), format="pdf")
        plt.close()


#need ffmpeg to run the following

from matplotlib.animation import FuncAnimation
from IPython import display
import pickle

def make_animation(saved_result, animation_name, show_speed = True) :
    
    if isinstance(saved_result, str) :
        with open(saved_result, "rb") as f:
            out, x = pickle.load(f)
    else :
        out, x = saved_result

    figure, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    ax_true = ax[0]
    ax_pred = ax[1]

    left = np.nanmin(x[:,:,0])
    right = np.nanmax(x[:,:,0])
    down = np.nanmin(x[:,:,1])
    up = np.nanmax(x[:,:,1])

    def AnimationFunction(i):
        ax_true.clear()
        ax_pred.clear()
        
        pos_out = out[i, :, :2]
        pos_x = x[i, :, :2]

        #now plot the graph as bubbles to show the difference between the two
        ax_true.scatter(pos_x[:, 0], pos_x[:, 1], s=100, c='b', alpha=0.5, label="Synthetic data position")
        ax_pred.scatter(pos_out[:, 0], pos_out[:, 1], s=100, c='r', alpha=0.5, label="Model position")

        if show_speed :
            speed_out = out[i, :, 2:]
            speed_x = x[i, :, 2:]

            #show an arrow for the speed
            ax_true.quiver(pos_x[:, 0], pos_x[:, 1], speed_x[:, 0], speed_x[:, 1], color='b', alpha=0.5, label="Synthetic data speed")
            ax_pred.quiver(pos_out[:, 0], pos_out[:, 1], speed_out[:, 0], speed_out[:, 1], color='r', alpha=0.5, label="Model speed")
        
        for ax in [ax_true, ax_pred] :
            ax.set_xlim(left, right)
            ax.set_ylim(down, up)
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax.legend(loc="upper left")
            

        #make a global title
        figure.suptitle("Cell movement (position and speed) at time " + str(i) + " (out of " + str(x.shape[0]) + ")")
        
        #save incrementally for the report
        if i % 25 == 0 and i > 0 :
            plt.savefig(animation_name.replace(".gif", "") + "_" + str(i) + ".png", format="png")

    anim_created = FuncAnimation(figure, AnimationFunction, frames=x.shape[0], interval=70)

    #we can show the animation with the following
    #video = anim_created.to_html5_video()
    #html = display.HTML(video)
    #display.display(html)

    #we can recover the animation with the following
    anim_created.save(animation_name, writer="ffmpeg")
    
    # good practice to close the plt object.
    plt.close()
    
def make_real_animation(saved_result, animation_name) :
    
    if isinstance(saved_result, str) :
        with open(saved_result, "rb") as f:
            out, x = pickle.load(f)
    else :
        out, x = saved_result

    figure, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    ax_true = ax[0]
    ax_pred = ax[1]

    left = 300
    right = 600
    down = 100
    up = 350

    def AnimationFunction(i):
        ax_true.clear()
        ax_pred.clear()
        
        pos_out = out[i, :, :2]
        pos_x = x[i, :, :2]

        #we have the following x,y, speed_x, speed_y,ori,major,minor,area
        #but we will ignore the speed
        
        orientation_out = out[i, :, 4] * 360 / np.pi - 90
        orientation_x = x[i, :, 4] * 360 / np.pi - 90
        
        major_out = out[i, :, 5]
        major_x = x[i, :, 5]
        
        minor_out = out[i, :, 6]
        minor_x = x[i, :, 6]
        
        #scatter ovals for the true and predicted
        for j in range(pos_x.shape[0]) :
            #if value is nan, we don't plot it
            if np.isnan(pos_x[j, 0]) :
                continue
            ax_true.add_artist(Ellipse(pos_x[j, :2], major_x[j], minor_x[j], orientation_x[j], color='b', alpha=0.5))
            ax_pred.add_artist(Ellipse(pos_out[j, :2], major_out[j], minor_out[j], orientation_out[j], color='r', alpha=0.5))
            
        
        for ax in [ax_true, ax_pred] :
            ax.set_xlim(left, right)
            ax.set_ylim(down, up)
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
        figure.suptitle("Cell movement (position and orientation) at time " + str(i) + " (out of " + str(x.shape[0]) + ")")
            
    anim_created = FuncAnimation(figure, AnimationFunction, frames=x.shape[0], interval=70)

    #we can show the animation with the following
    #video = anim_created.to_html5_video()
    #html = display.HTML(video)
    #display.display(html)

    #we can recover the animation with the following
    anim_created.save(animation_name, writer="ffmpeg")
    
    # good practice to close the plt object.
    plt.close()
    