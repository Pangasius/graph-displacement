import threading

import matplotlib.pyplot as plt

import torch

class GraphingLoss():
    def __init__(self, losses):
        self.losses = losses
        self.stop = False
        self.timer = 0
        self.last_len = 0
        self.fig = None
        self.ax = None

    def plot_and_reschedule(self):
        if not self.stop:
            if (self.last_len != len(self.losses)) :
                if self.fig is None :
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111)
                
                self.ax.clear() # type: ignore
                self.ax.plot(self.losses[::2], label="Recursive Loss") # type: ignore
                self.ax.plot(self.losses[1::2], label="Static loss") # type: ignore
                self.ax.legend(["True position", "Predicted position"], loc="upper left")
                self.fig.show()
                self.fig.canvas.draw()
                self.last_len = len(self.losses)
                
                plt.savefig("Losses.pdf", format="pdf")

            threading.Timer(self.timer, self.plot_and_reschedule).start()
            
    def gstop(self):
        self.stop = True
        
    def gstart(self, timer=20):
        self.timer = timer
        if (not self.timer or self.timer != int(self.timer)):
            raise ValueError("timer must be a positive integer")
        
        threading.Timer(self.timer, self.plot_and_reschedule).start()

#need ffmpeg to run the following

from matplotlib.animation import FuncAnimation
from IPython import display
import pickle

def make_animation(saved_result, animation_name) :
    with open(saved_result, "rb") as f:
        out, x = pickle.load(f)

    figure = plt.figure()
    ax = figure.add_subplot(111)

    left = x[:,:,0].min()
    right = x[:,:,0].max()
    down = x[:,:,1].min()
    up = x[:,:,1].max()

    def AnimationFunction(i):
        ax.clear()
        
        pos_out = out[i, :, :2]
        pos_x = x[i, :, :2]

        speed_out = out[i, :, 2:]
        speed_x = x[i, :, 2:]

        #now plot the graph as bubbles to show the difference between the two
        ax.scatter(pos_x[:, 0], pos_x[:, 1], s=100, c='b', alpha=0.5, label="True position")
        ax.scatter(pos_out[:, 0], pos_out[:, 1], s=100, c='r', alpha=0.5, label="Predicted position")

        #show an arrow for the speed
        ax.quiver(pos_x[:, 0], pos_x[:, 1], speed_x[:, 0], speed_x[:, 1], color='b', alpha=0.5, label="True speed")
        ax.quiver(pos_out[:, 0], pos_out[:, 1], speed_out[:, 0], speed_out[:, 1], color='r', alpha=0.5, label="Predicted speed")
        
        ax.set_xlim(left, right)
        ax.set_ylim(down, up)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.set_title("Cell movement (position and speed) at time " + str(i) + " (out of " + str(x.shape[0]) + ")")

        ax.legend(["True position", "Predicted position"], loc="upper left")

    anim_created = FuncAnimation(figure, AnimationFunction, frames=x.shape[0], interval=100)

    #we can show the animation with the following
    #video = anim_created.to_html5_video()
    #html = display.HTML(video)
    #display.display(html)

    #we can recover the animation with the following
    anim_created.save(animation_name, writer="ffmpeg")
    
    # good practice to close the plt object.
    plt.close()