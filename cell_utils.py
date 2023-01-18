import threading

import matplotlib.pyplot as plt

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
                self.ax.plot(losses) # type: ignore
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
