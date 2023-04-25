import numpy as np
class Parameters(object):
    def __init__(self, p):
        for key, values in p.items():
            setattr(self, key, values)

# global param #required for pickling
class param:
    def __init__(self, framerate, Lx, Ly, R ):
        self.Lx = Lx
        self.Ly = Ly
        self.R = R
        self.framerate = framerate #given in hours
        self.dt = 0.001
        self.output_time = int(self.framerate/self.dt)

class ExperimentData:
    def gettypes(self, readtypes, frames):
        return np.isin(self.ptype[frames],readtypes)
        
    def truncateto(self,start, endtime):
        if not self.truncated:
            self.Nsnap = endtime - start
            self.flag =  self.flag[start:endtime]
            self.rval = self.rval[start:endtime]
            self.vval = self.vval[start:endtime]
            # self.radius = self.radius[start:endtime]
            self.ptype = self.ptype[start:endtime]
            self.truncated = True
            self.Nvals = self.Nvals[start:endtime]
        else:
            print("Already truncated. Skipping this step")

    # Subtracting avg posn and vel and each timepoint
    def takeDrift(self):
        if not self.drift_removed:
            rdrift = self.rval[1:self.Nsnap,:,:].mean(axis=0) - self.rval[:self.Nsnap-1,:,:].mean(axis=0)
            vdrift = self.vval.mean(axis=1)

            for t in range(self.Nsnap):
                ind = self.vval[t,:,0] != 0
                self.rval[t,ind,:] -= rdrift[t,:]
                self.vval[t,ind,:] -= vdrift[t,:]

            self.drift_removed = True
        else:
            print("Drift already removed. Skipping this step")

    def __init__(self,data,properties, framerate=0.166, Lx = 800, Ly =800, R = 10, umpp=0.8):

        self.Nvariable = True
        self.truncated = False
        self.drift_removed = False
        self.param = param(framerate, Lx, Ly, R )
        self.umpp = umpp #microns per pixel
        self.sigma = R

        self.Nvals = np.array([data[data[:,1] == n].shape[0] for n in range(properties['t'][-1])])
        self.Nvals = self.Nvals[self.Nvals != 0][1:]
        self.flags = np.unique(data[:,0]) 
        # count number of occurances of each flag
        counts = np.bincount(data[:,0].astype(int))
        # remove those flags that never occur (such as 0)
        counts = counts[counts !=0]
        # remove those flags which are only there for 5 frames
        self.flags = self.flags[counts>5]
        self.flag_lifespan = counts[counts>5]
        self.all_flags = len(self.flags)
        print(properties['t'][-1],properties['t'][0])
        self.Nsnap = properties['t'][-1] - properties['t'][0] -1

        rvalues = []
        timevalues = []
        velvalues = []

        #generate flag specific rvalues,timevalues and vel values
        for f in self.flags:
            # get all rvalues related to a flag and turn into a numpy array
            rval = data[:,2:][data[:,0] == f]*umpp 
            rval[:,0] -= Lx/2
            rval[:,1] -= Ly/2
            # get all time values related to a flag
            time = data[:,1][data[:,0] == f] 
            
            # append everything but the first which is removed as there is no veloc data there
            rvalues.append(rval[1:])
            # note, we don't take any of the 0 time values that end up padding based on bool check above
            timevalues.append(time[time !=0][1:])
            velvalues.append(np.diff(rval, axis=0)/framerate)

        # ^data now ordered by flag number
        # length of all flags
        r_new = []
        flag_new = []
        v_new = []

        #Reorder to become time -> This is horrific... 
        for t in range(properties['t'][0]+1,properties['t'][-1]):
            print(t,end='\r')
            flagtmp = []
            rtmp =[]
            vtmp = []
            #loop through flags
            for i, flag in enumerate(self.flags):
                # find flag that has this timevalue
                if bool(sum(timevalues[i]==t)):
                    #append to tmp lists
                    flagtmp.append(flag)
                    rtmp.append(rvalues[i][timevalues[i]==t])            
                    vtmp.append(velvalues[i][timevalues[i]==t])            
            flag_new.append(np.asarray(flagtmp))
            r_new.append(np.concatenate(rtmp,axis=0))
            v_new.append(np.concatenate(vtmp,axis=0))

        # number of flags per timestep
        self.flags_per_timestep = np.asarray([len(flag) for flag in flag_new])
        self.maxN = self.flags_per_timestep.max()

        self.rval = np.zeros((self.Nsnap, self.flags_per_timestep.max(),2))
        self.vval = np.zeros((self.Nsnap, self.flags_per_timestep.max(),2))
        self.flag = np.zeros((self.Nsnap, self.flags_per_timestep.max()))
        #Turn back into arrayss
        for t in range(self.Nsnap):
            self.rval[t,:len(r_new[t]),:] = r_new[t]
            self.vval[t,:len(v_new[t]),:] = v_new[t]
            self.flag[t,:len(flag_new[t])] =  flag_new[t]
        # get all those cells which are in all frames (tracers) 
        self.tracers = self.flags[self.flag_lifespan == (self.Nsnap+2)]
        self.ptype = np.isin(self.flag,self.tracers)*1

class SimData:
    def checkTypes(readtypes,data):
        #check which particles to load 
        if len(readtypes) > 0:
            usetypes = np.isin(data[:,-1],readtypes)
        else:
            usetypes = [True]*len(data)
        return usetypes

    # Data object for summary statistics
    def __init__(self,**kwargs):
        # check for debugging
        try:
            self.debug = kwargs['debug']
            if self.debug:
                print('kwargs: ', kwargs)
        except:
            self.debug = False
        # check for specific loadtimes
        try:    
            self.start = kwargs["loadtimes"][0]
            self.end = kwargs["loadtimes"][1]
            self.multiopt = True
        except:
            self.multiopt = False
        # check for specific types
        try:
            self.readtypes = kwargs["readtypes"]
        except:
            self.readtypes = []
        # load parameters
        try:    
            self.param = Parameters(kwargs['params'])
        except:
            print('Error! Parameters must be a dictionary')
            return 1
        # load multiple simulation snapshots
        if self.multiopt:
            self.Nsnap = self.end - self.start
            #get maximum number of particles
            self.N = sum(SimData.checkTypes(self.readtypes, kwargs['data'][0]))
            self.Nvals = []
            self.Nvariable =  False
            for t in range(self.start,self.end):
                self.Nvals.append(sum(SimData.checkTypes(self.readtypes, kwargs['data'][t])))
                if self.Nvals[t] > self.N:
                    self.N = self.Nvals[t] 
                    self.Nvariable = True

            self.flag=np.zeros((self.Nsnap,self.N))
            self.Z=np.zeros((self.Nsnap,self.N))
            self.rval=np.zeros((self.Nsnap,self.N,2))
            self.vval=np.zeros((self.Nsnap,self.N,2))
            self.theta =np.zeros((self.Nsnap,self.N))
            self.nval=np.zeros((self.Nsnap,self.N,2))
            self.radius=np.zeros((self.Nsnap,self.N))
            self.ptype=np.zeros((self.Nsnap,self.N))
            self.sigma = 0.

            for t in range(self.start,self.end):
                # only get particles we're interestsed in
                usetypes = SimData.checkTypes(self.readtypes, kwargs['data'][t])
                
                idx = range(sum(usetypes))
                #check whether data is old or new style
                if kwargs['data'][t].shape[1] > 4:
                    #new output includes v,theta,radius
                    self.flag[t,idx] =  kwargs['data'][t][usetypes,0]
                    self.rval[t,idx,:] = kwargs['data'][t][usetypes,1:3]
                    self.vval[t,idx,:] = kwargs['data'][t][usetypes,3:5]
                    self.theta[t,idx] = kwargs['data'][t][usetypes,5]
                    self.nval[t,idx,:] = np.array([np.cos(kwargs['data'][t][usetypes,5]), np.sin(kwargs['data'][t][usetypes,5])]).T
                    self.radius[t,idx] = kwargs['data'][t][usetypes,6]
                    self.ptype[t,idx] = kwargs['data'][t][usetypes,7]
                    self.Z[t,idx] = kwargs['data'][t][usetypes,8]
                    sigma = np.mean(kwargs['data'][t][usetypes,6])
                    if sigma>self.sigma:
                        self.sigma = sigma
                else:
                    #old output only contains flag, r and type
                    self.flag[t,idx] =  kwargs['data'][t][usetypes,0]
                    self.rval[t,idx,:] = kwargs['data'][t][usetypes,1:3]
                    self.ptype[t,idx] = kwargs['data'][t][usetypes, 3]

        # or a single snapshot
        else:
            # only get particles we're interestsed in
            usetypes = SimData.checkTypes(self.readtypes, kwargs['data'])
            self.Ntrack = sum(usetypes)
            #check whether data is old or new style
            if kwargs['data'].shape[1] > 4:
                #new output includes v,theta,radius
                self.flag =  kwargs['data'][usetypes,0]
                self.rval = kwargs['data'][usetypes,1:3]
                self.vval = kwargs['data'][usetypes,3:5]
                self.theta = kwargs['data'][usetypes,5]
                self.nval = np.array([np.cos(self.theta), np.sin(self.theta)]).T
                self.radius = kwargs['data'][usetypes,6]
                self.ptype = kwargs['data'][usetypes,7]
                self.Z = kwargs['data'][usetypes,8]
            else:
                #old output only contains flag, r and type
                self.flag =  kwargs['data'][usetypes,0]
                self.rval = kwargs['data'][usetypes,1:3]
                self.ptype = kwargs['data'][usetypes, 3]
        
                # For defect tracking
                self.vnorm = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2+self.vval[:,2]**2)
                self.vhat = self.vval / np.outer(vnorm,np.ones((3,)))
                
                self.N = len(radius)
                self.sigma = np.mean(radius)
                print("New sigma is " + str(self.sigma))

    def gettypes(self, readtypes, frames):
        return np.isin(self.ptype[frames],readtypes)
        
    def truncateto(self,start, endtime):
        self.Nsnap = endtime - start
        self.flag =  self.flag[start:endtime]
        self.rval = self.rval[start:endtime]
        self.vval = self.vval[start:endtime]
        self.theta = self.theta[start:endtime]
        self.nval = self.nval[start:endtime]
        self.radius = self.radius[start:endtime]
        self.ptype = self.ptype[start:endtime]
        self.Nvals = self.Nvals[start:endtime]
        
    def spatialcut(self,minL=-400, maxL=400, dim=0):
        for t in range(self.Nsnap):
            cut_indices = (self.rval[t][:,dim] < minL) | (self.rval[t][:,dim] > maxL)
            self.flag[t][cut_indices] = [0]
            self.vval[t][cut_indices,:] = [0,0]
            self.theta[t][cut_indices] = [0]
            self.nval[t][cut_indices,:] = [0,0]
            self.radius[t][cut_indices] = [0]
            self.ptype[t][cut_indices] = [0]
            self.rval[t][cut_indices,:] = [0,0]
            self.Z[t][cut_indices,:] = [0,0]