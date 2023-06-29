import numpy as np
from scipy import optimize
from scipy import stats
import scipy.interpolate
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

def calculate_summary_statistics(d, opts = ['A','B','C','D','E','F','G','H'],useall=True, log=False,starttime=60,endtime=320,takeDrift=False, plot = False, usetypes = [0, 1,2],log_output="log.txt"):
    """
    Calculates summary statistics.

    """
    # 0 is new cells, 1 is tracer, 2 is original (check this)    

    if hasattr(d.param, 'framerate') is False:
#         print('No framerate set, setting it to 1')
        d.param.framerate = 1

    # remove any data post zap
    d.truncateto(starttime, endtime)
    if takeDrift:
        print('removing drift')
        d.takeDrift()

    ssdata = {}
    ssvect = []
    all_ssvect = []
    ssdata['params'] = d.param
    if d.Ntracers == 0:
        #if there are no tracers, everything is a tracer
        d.Ntracers = len(d.pytype[-1])

    if 'A' in opts:
        # # # # # A - Velocity distributions and mean velocity
        velbins=np.linspace(0,5,100)
        velbins2=np.linspace(-2,2,100)
        vav, vdist,vdist2 = getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)
        
        vdist = vdist[1:]
        vdist2 = vdist2[vdist2 != max(vdist2)]

        ssdata['vav'] = vav
        ssdata['vdist'] = vdist
        ssdata['vdist2'] = vdist2
        if log: print('Finished calculating A. vel. dist & mean vel', file=open(log_output, 'a'))
        ssvect.append(vav.mean()) 
        all_ssvect.append(vav)
        ssvect.append(stats.kurtosis(vdist,fisher=False))
        ssvect.append(vdist.mean())
        ssvect.append(vdist.var())
        all_ssvect.append(vdist)
        ssvect.append(stats.kurtosis(vdist2,fisher=False))
        all_ssvect.append(vdist2)
        ssvect.append(vdist2.mean())
        ssvect.append(vdist2.var())
    if 'B' in opts:
        # # B - Autocorrelation Velocity Function
        tval2, velauto, v2av = getVelAuto(d, usetype=[1],verbose=plot)
        ssdata['tval2'] = tval2
        ssdata['velauto'] = velauto
        ssdata['v2av'] = v2av
        if log: print('Finished calculating B. autocorr vel fcn', file=open(log_output, 'a'))
        ssvect.append(tval2[velauto < 5e-1][0])
        all_ssvect.append(velauto)
    if 'C' in opts:
        # C - Mean square displacement
        tval, msd, d = getMSD(d, usetype=[1],verbose=plot)
        ssdata['tval'] = tval
        ssdata['msd'] = msd
        if log: print('Finished calculating C. MSD', file=open(log_output, 'a'))
        ssvect.append(np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0])
        ssvect.append(np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[1])
        ssvect.append(ssdata['msd'][-1])
        all_ssvect.append(msd)
        # v0, tau = optimize.curve_fit(lambda t, v0, tau:  2*v0*v0*tau*(t - tau*(1-np.exp(-t/tau))),xdata = tval[1:], ydata = msd[1:])[0]
        # ssvect.append(v0)
        # ssvect.append(tau)
    if 'D' in opts:     
        # # D - Self Intermediate Scattering Function 
        qval = 2*np.pi/d.sigma*np.array([1,0])
        tval3, SelfInt2, SelfInt = SelfIntermediate(d, qval,usetype=[1],verbose=plot)
        ssdata['tval3'] = tval3
        ssdata['SelfInt2'] = SelfInt2
        ssdata['SelfInt'] = SelfInt
        step = 10
        qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
        all_ssvect.append(SelfInt2)
        if log: print('Finished calculating D. self-intermediate scattering fcn', file=open(log_output, 'a'))
        if np.nansum(SelfInt2 < 0.5) > 0:
            ssvect.append(tval3[SelfInt2 < 0.5][0])
        else:
            ssvect.append(tval3[-1])
            
    if 'E' in opts:
        # # E - real space velocity correlation function ('swirlyness')
        step = 10
        velcorrReal = np.zeros((150,))
        dx =  d.sigma*0.9
        xmax = d.param.Ly
        ssdata['dx'] = dx
        ssdata['xmax'] = xmax
        count = 0
        for u in range(0,endtime - starttime,step):
            # # # E - Real space velocity correlation function
            spacebins,velcorr = getVelcorrSingle(d, dx,xmax,whichframe=u,usetype=usetypes,verbose=False)
            velcorrReal[:len(spacebins)] += velcorr  
            count+=1

        velcorrReal = velcorrReal[:len(spacebins)]
        velcorrReal/=count
        ssdata['velcorrReal'] = velcorrReal
        ssdata['spacebins'] = spacebins
        all_ssvect.append(velcorrReal)
        x = spacebins[(50<spacebins) & (spacebins < 300)]
        y = velcorrReal[(50<spacebins) & (spacebins< 300)]
        
        if log: print('Finished calculating E. vel. corr. fcn', file=open(log_output, 'a'))
        if np.nansum(y>0) > 0:
            ssvect.append(np.polyfit(np.log(x[y>0]), np.log(y[y>0]), 1)[0])
        else:
            ssvect.append(0)

    if 'F' in opts:
        # # F - Radial distribution function, g(r)
        rdist, gr = calcgr(d, verbose=plot)
        if log: print('Finished calculating F. g(r)', file=open(log_output, 'a'))
        ssdata['rdist'] = rdist
        ssdata['gr'] = gr
        # ssvect.append(rdist[np.where(gr == max(gr))][0])
        # all_ssvect.append(gr)

    if 'G' in opts:
        # Fourier Transformed (Spatial)

        # upper limit of particle size in wavelength
        qmax = np.pi/d.param.R #= 2*pi/particle size where particle size = 2*R 
        L = d.param.Lx
        # Note to self: only low q values will be interesting in any case. 
        # The stepping is in multiples of the inverse box size. Assuming a square box.
        dq=2*np.pi/L
        nq=int(qmax/dq)

        print(f"Stepping Fourier transform with step {dq:.2f}, resulting in {nq} steps (qmax = {qmax}).")
        qrad,valrad = FourierTrans(d, qmax,whichframe = 0, usetype=[0,1],verbose=plot)
        ssdata['qrad0'] = qrad
        ssdata['valrad0'] = [valrad]
        for u in range(1,endtime - starttime):
            qrad,valrad = FourierTrans(d, qmax,whichframe = u, usetype=[0,1],verbose=plot)
            ssdata['valrad0'].append(valrad)

        if log: print('Finished calculating G. Fourier transform (spatial)', file=open(log_output, 'a'))
    if 'H' in opts:
        # Fourier Transformed (Velocity)

        # upper limit of particle size in wavelength
        qmax = np.pi/d.param.R #= 2*pi/particle size where particle size = 2*R 
        L = d.param.Lx
        # Note to self: only low q values will be interesting in any case. 
        # The stepping is in multiples of the inverse box size. Assuming a square box.
        dq=2*np.pi/L
        nq=int(qmax/dq)
            
        print(f"Stepping Fourier transform with step {dq:.2f}, resulting in {nq} steps (qmax = {qmax}).")
        qrad,valrad,Sqrad = FourierTransVel(d, qmax,whichframe = 0, usetype=[0,1],verbose=plot)
        ssdata['qrad1'] = qrad
        ssdata['valrad1'] = [valrad]
        ssdata['Sqrad1'] = [Sqrad]
        for u in range(1,endtime - starttime):
            qrad,valrad,Sqrad = FourierTransVel(d, qmax,whichframe = u, usetype=[0,1],verbose=plot)
            ssdata['valrad1'].append(valrad)
            ssdata['Sqrad1'].append(Sqrad)

        if log: print('Finished calculating H. Fourier transform (velocity)', file=open(log_output, 'a'))


    if 'I' in opts:
        # # H - Change in density
        if log: print('Finished calculating I. change in phi', file=open(log_output, 'a'))
        ssvect.append(deltaphi(d))
        if log: print('Finished calculating I. mean_vect_vel', file=open(log_output, 'a'))
        ssvect.append(mean_vect_vel(d))
        if log: print('Finished calculating I. avg. horiz. disp. (from midway point)', file=open(log_output, 'a'))
        ssvect.append(deltax(d))
    if 'J' in opts:
        # # J - Division fit

        tvals, Nvals = fit_div_rate(d, verbose=plot)
        ssdata['tval3'] = tvals
        ssdata['Nvals'] = Nvals
        
        # fit exponential
        model = np.polyfit(tvals, np.log(Nvals), 1)
        predict = np.poly1d(model)
        # r2 = r2_score(Nvals, predict(tvals))

        if log: print('Finished calculating G. Division details', file=open(log_output, 'a'))
        # append model fits, error, N0
        ssvect.append(model[0])
        ssvect.append(model[1])
        # ssvect.append(r2)
        # ssvect.append(Nvals[0])
        all_ssvect.append(np.log(Nvals))
    if log: print('Finished calculating summary statistics', file=open(log_output, 'a'))
    if useall:                
        return combine_sumstats(ssdata), ssdata
    else:
        return ssvect, ssdata

def combine(ssdata, ss_opt = ['vav', 'vdist', 'vdist2', 'msd','Nvals'], plot = False, rescale_Y = True, ss_length = 100):
    
    def rescale(d):
        #simple rescaling function (to btwn [0,1]) for saving all summvects
        return (d - np.nanmin(d))/(np.nanmax(d) - np.nanmin(d))

    def prep_vav(ss, bounds = [0,1], nstd = 2):
        
        def moving_average(x, w, option='valid'):
            return np.convolve(x, np.ones(w), option) / w

        x = np.linspace(0,1,len(ss['vav']))
        y = ss['vav']

        params = np.polyfit(x, y, 1)
        linpart = x*params[0] + params[1]
        lin_y = y/linpart
        std_vel = lin_y.std()
        vav = lin_y.mean() 
        ind = lin_y > (vav + nstd*std_vel)

        y_interp = scipy.interpolate.interp1d(x[~ind], y[~ind])  
        X = np.linspace(x[~ind][0],x[~ind][-1],ss_length + 4)
        Y = y_interp(X)
        Y = moving_average(Y, 5, 'valid') 
        
        return X[2: ss_length + 2], Y 

    def prep_MSD(ss, bounds = [-1,2.5]):
        x = np.log(ss['tval'])
        Y = np.log(ss['msd'])

        X = np.linspace(bounds[0],bounds[1],ss_length)
        
        y_interp = scipy.interpolate.interp1d(x, Y)
        
        return X, y_interp(X)

    def prep_vmag(ss, bounds = [0,5]):
        velbins = np.linspace(0,10,100)
        db = velbins[1]-velbins[0]
        x = velbins[2:]-db/2
        y = np.log(ss['vdist'])
        
        X = np.linspace(x[0],bounds[1],ss_length)
        
        y_interp = scipy.interpolate.interp1d(x, y)
        
        return X, y_interp(X)

    def prep_vcomp(ss, bounds = [-5,5]):
        velbins = np.linspace(-10,10,100)
        db = velbins[1]-velbins[0]
        x = velbins[2:]-db/2

        y = np.log(ss['vdist2'])
        
        X = np.linspace(bounds[0],bounds[1],ss_length)
        
        y_interp = scipy.interpolate.interp1d(x, y)
        
        return X, y_interp(X)

    def prep_Nvals(ss, bounds = [1,100]):

        Y = np.log(ss['Nvals'])
        x = np.arange(len(Y))

        X = np.linspace(x[0],x[-1],100)
        
        y_interp = scipy.interpolate.interp1d(x, Y)
        
        return X, y_interp(X)

    all_Y = np.array([])

    for opt in ss_opt: 
        if opt == 'vav':
            X, Y = prep_vav(ssdata)
        elif opt == 'msd':
            X, Y = prep_MSD(ssdata)
        elif opt == 'vdist':
            X, Y = prep_vmag(ssdata)
        elif opt == 'vdist2':
            X, Y = prep_vcomp(ssdata)
        elif opt == 'Nvals':
            X, Y = prep_Nvals(ssdata)
        else:
            print("This option is not yet ready for training")
        if plot:
            plt.plot(X,Y)
            plt.show()
        if rescale_Y:
            all_Y = np.append(all_Y,rescale(Y))
        else:
            all_Y = np.append(all_Y,Y)

    return all_Y

def fit_div_rate(data,verbose=True):
    # get time, N values
    tval = np.linspace(0,data.Nsnap*data.param.framerate,num=data.Nsnap)
    Nvals = data.Nvals
    if verbose:
        plt.figure()
        plt.plot(tval, np.poly1d(np.polyfit(tval, np.log(Nvals), 1))(tval), 'r',lw=2)
        plt.plot(tval,np.log(Nvals),'r.-',lw=2)
        plt.show()
    return tval, Nvals

def deltax(data,usetype=[1]):
    tracers_start = data.gettypes(usetype,0)
    tracers_end = data.gettypes(usetype,len(data.rval)-1)
    return np.nanmean(data.rval[-1][tracers_end,0] - data.rval[0][tracers_start,0])
    
def deltaphi(data):
    return (data.Nvals[-1] - data.Nvals[0])*(np.pi*data.param.R*data.param.R)/(data.param.Lx*data.param.Ly)        

def mean_vect_vel(data):
    return data.vval.mean(axis=1)

def ApplyPeriodic2d(data,dr):
    dr[:,0]-=data.param.Lx*np.round(dr[:,0]/data.param.Lx)
    dr[:,1]-=data.param.Ly*np.round(dr[:,1]/data.param.Ly)
    return dr

# relative velocity distribution (and average velocity)
# component wise as well, assumes x and y directions only
# use all:
def getVelDist(data,bins,bins2,usetype=[1,2],verbose=True):
    vav=np.zeros((data.Nsnap-1,))
    vdist=np.zeros((len(bins)-1,))
    vdistx=np.zeros((len(bins2)-1,))
    vdisty=np.zeros((len(bins2)-1,))
    for u in range(data.Nsnap-1):
        
        # The particles we want to average over
        tracers = data.gettypes(usetype,u)
        # get all the magnitudes, and all the components
        vmagnitude=np.sqrt(data.vval[u,tracers,0]**2+data.vval[u,tracers,1]**2)
        vx = data.vval[u,tracers,0]
        vy = data.vval[u,tracers,1]
        # mean velocity magnitude
        vav[u]=np.nanmean(vmagnitude)
        # normalised magnitude histogram
        vdist0,dummy=np.histogram(vmagnitude/vav[u],bins,density=True)
        vdist+=vdist0

        # normalised component histogram (x)
        vdistx0,dummy=np.histogram(vx/vav[u],bins2,density=True)
        vdistx+=vdistx0

        # normalised component histogram (y)
        vdisty0,dummy=np.histogram(vy/vav[u],bins2,density=True)
        vdisty+=vdisty0

    vdist/=data.Nsnap-1
    vdistx/=data.Nsnap-1
    vdisty/=data.Nsnap-1
    if verbose:
        fig=plt.figure()
        db=bins[1]-bins[0]
        plt.semilogy(bins[1:]-db/2,vdist,'r.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(v/<v>)')
        plt.title('Scaled velocity magnitude distribution')
        plt.show()

        fig=plt.figure()
        db=bins2[1]-bins2[0]
        plt.semilogy(bins2[1:]-db/2,vdistx,'r.-',lw=2)
        plt.semilogy(bins2[1:]-db/2,vdisty,'k.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(v/<v>)')
        plt.title('Scaled velocity component (x & y) distribution')
        plt.show()

        xval=np.linspace(0,(data.Nsnap-1)*data.param.dt*data.param.output_time,num=data.Nsnap-1)
        plt.figure()
        plt.plot(xval,vav,'r.-',lw=2)
        plt.xlabel('time')
        plt.ylabel('mean velocity')
        plt.show()

    return vav, vdist, np.sqrt(vdistx**2 + vdisty**2)


def getMSD(data, usetype=[1],verbose=True, periodic=True):
    msd=np.empty((data.Nsnap-1,))
    
    for t in range(data.Nsnap-1): 
        smax=data.Nsnap-t

        # get tracer idx for each timestep
        isdata = data.gettypes(usetype,t)
        
        # get rval for up to smax
        rt = data.rval[:smax,isdata]
        # g et rval for u to end
        rtplus = data.rval[t:,isdata]

        dr  = rt - rtplus
        if periodic:             
            for n in range(smax):
                dr[n] = ApplyPeriodic2d(data, dr[n])

        msd[t]=np.nansum(np.nansum(np.nansum(dr**2,axis=2),axis=1),axis=0)/(data.Ntracers*smax)

    data.hasMSD = True
    data.msd = msd

    tval=np.linspace(0,(data.Nsnap - 1)*data.param.dt*data.param.output_time,num=(data.Nsnap-1))
    if verbose:
        fig=plt.figure()
        plt.loglog(tval,msd,'r.-',lw=2)
        plt.loglog(tval,msd[1]/(1.0*tval[1])*tval,'--',lw=2,color="k")
        plt.xlabel('time (hours)')
        plt.ylabel('MSD')
        plt.title('Mean square displacement')
        plt.show()

    return tval, msd, data

# Velocity autocorrelation function
# do use tracers
def getVelAuto(data,usetype=[1],verbose=True):

    velauto=np.empty((data.Nsnap-1,))    
    #this by defn has to have fixed type for all time 
    # i.e. tracer particles which are present throughout and which don't change label / flag
    isdata = data.gettypes(usetype,0)

    #print(data.vval[:,isdata,])
    if verbose:
        plt.figure()
        plt.pcolor(data.vval[:,isdata,0])
        plt.figure()
        plt.pcolor(data.vval[:,isdata,1])

    for t in range(data.Nsnap-1):
        tmax=(data.Nsnap-1)-t
        velauto[t]=np.nansum(np.nansum((data.vval[t:,isdata,0]*data.vval[:tmax,isdata,0]+\
                                  data.vval[t:,isdata,1]*data.vval[:tmax,isdata,1]),axis=1),axis=0)/(data.Ntracers*tmax)
    #normalising back to velauto = 1 due to different averaging (types 1 & 2 rather than 1) - out by approx 4%                        
    xval=np.linspace(0,round((data.Nsnap-1)*data.param.dt*data.param.output_time),num=(data.Nsnap-1))
    velauto /= velauto[0]
    if verbose:
        fig=plt.figure()
        plt.plot(xval,velauto,'r.-',lw=2)
        plt.xlabel('time')
        plt.ylabel('correlation')
        plt.title('Normalised Velocity autocorrelation function')
        plt.show()

    return xval, velauto

def calcgr(data, usetype=[0,1,2], step=10, verbose = True, periodic=True, limit = 8,resolution=2):
    
    def ApplyPeriodic2d(L,dr):
        outside_range = np.round(dr/L)#.astype(int)
        shift = L*outside_range.astype(float)
        dr -= shift
        return dr

    def calc_dist(p,xy):
        L = [data.param.Lx,data.param.Ly]
        squared_periodic_dr = ApplyPeriodic2d(L,p-xy)**2
        dr_sum = squared_periodic_dr.sum(axis=1)     
        dr = np.sqrt(dr_sum)
        return dr

    def find_near(dr,resolution):
        return (dr/resolution).astype(int)

    max_distance = min(data.param.Lx,data.param.Ly)/limit
    Nrings = int(max_distance/resolution)
    rdist = np.linspace(0,max_distance,Nrings)
    print(Nrings)

    bins = np.zeros(Nrings)
    area = np.zeros(Nrings)
    gr_time_avg= np.zeros(Nrings)
    # density = 2*np.pi*8/(data.param.Lx*data.param.Ly)
    for j in range(Nrings):
        r1 = j * resolution
        area[j] = 2*np.pi*(r1)*resolution

    # loop through the frames and calculate g(r) 
    for t in range(0,data.Nsnap-1,step):
        print(f"frame = {t}", end="\r")
        isdata = isdata = data.gettypes(usetype,t)
        xy =  data.rval[t,isdata]

        bins = np.zeros(Nrings)
        bins_new = np.zeros(Nrings)
        for p in xy:
            dr = calc_dist(p,xy)
            p_all_bins = find_near(dr,resolution)

            for b in range(Nrings):
                bins[b] += (p_all_bins==b+1).sum()
           
        rho = data.Nvals[t]/(data.param.Lx*data.param.Ly)
        # bins normalised by number of particles in system
        renormed_bins = bins/data.Nvals[t]
        # area of annulus times density of particles in the system (N/A) according to ideal gas assumption
        ideal_gas_density = (area)*rho

        gr = (renormed_bins)/(ideal_gas_density)
        gr_time_avg += gr

    #Step here is normalising to reach 1 at long distances
    gr_time_avg /= np.floor((data.Nsnap-1)/step)

    rdist = rdist[~np.isnan(gr_time_avg)]
    gr_final = gr_time_avg[~np.isnan(gr_time_avg)]
    if verbose:
        fig=plt.figure()
        plt.plot(rdist, gr_final,'r.-',lw=2)
        # plt.xlim([0,max_distance])
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.title('Radial density distribution (2d)')
        plt.show()
    return rdist, gr_final


# Definition of the self-intermediate scattering function (Flenner + Szamel)
# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
def SelfIntermediate(data,qval,usetype=[1],verbose=True, periodic=True):
    # This is single particle, single q, shifted time step. Equivalent to the MSD, really
    SelfInt=np.empty((data.Nsnap-1,),dtype=complex)
            
    for t in range(data.Nsnap-1):

        smax=data.Nsnap-t

        # get tracer idx for each timestep
        isdata = data.gettypes(usetype,t)

        rt = data.rval[t:,isdata]
        rtplus = data.rval[:smax,isdata]
        dr = rt - rtplus
        for n in range(smax):
            dr[n] = ApplyPeriodic2d(data, dr[n])

        if periodic:
            SelfInt[t]=np.nansum(np.nansum(np.exp(1.0j*qval[0]*dr[:,:,0]+ \
                                            1.0j*qval[1]*dr[:,:,1] \
                                        ),axis=1),axis=0)/(data.Ntracers*smax)         
        else:   
            SelfInt[t]=np.nansum(np.nansum(np.exp(1.0j*qval[0]*(rt[:,:,0]-rtplus[:,:,0]) + \
                                            1.0j*qval[1]*(rt[:,:,1]-rtplus[:,:,1])\
                                        ),axis=1),axis=0)/(data.Ntracers*smax)                    

        
    # Looking at the absolute value of it here
    SelfInt2=(np.real(SelfInt)**2 + np.imag(SelfInt)**2)**0.5
    
    tval=np.linspace(0,round(data.Nsnap*data.param.dt*data.param.output_time),num=data.Nsnap-1)
    if verbose:
        qnorm=np.sqrt(qval[0]**2+qval[1]**2)
        fig=plt.figure()
        plt.semilogy(tval,SelfInt2,'.-r',lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate, k = ' + str(qnorm))
        plt.ylim([0,1])
        plt.show()
 
    return tval, SelfInt2, SelfInt

####################### Fourier space and real space equal time correlation functions ##################################

# Generate 2d points for radially averaged Fourier transform computations
def makeQrad(dq,qmax,nq):
    nq2=int(2**0.5*nq)
    qmax2=2**0.5*qmax
    qx=np.linspace(0,qmax,nq)
    qy=np.linspace(0,qmax,nq)
    qrad=np.linspace(0,qmax2,nq2)
    # do this silly counting once and for all
    binval=np.empty((nq,nq))
    for kx in range(nq):
        for ky in range(nq):
            qval=np.sqrt(qx[kx]**2+qy[ky]**2)
            binval[kx,ky]=round(qval/dq)
    ptsx=[]
    ptsy=[]
    # do the indexing arrays
    for l in range(nq2):
        pts0x=[]
        pts0y=[]
        for kx in range(nq):
            hmm=np.nonzero(binval[kx,:]==l)[0]
            for v in range(len(hmm)):
                pts0y.append(hmm[v])
                pts0x.append(kx)
        ptsx.append(pts0x)
        ptsy.append(pts0y)
    return qx, qy, qrad, ptsx, ptsy

# Static structure factor
# Which is implicitly in 2D!!
# FourierTrans(g(R)) = S(q)
def FourierTrans(data,qmax=0.3,whichframe=1,usetype=[0,1,2],verbose=True, debug=False):
    
    L = data.param.Lx
    # Note to self: only low q values will be interesting in any case. 
    # The stepping is in multiples of the inverse box size. Assuming a square box.
    dq=2*np.pi/L

    nq=int(qmax/dq)
    if debug:
        print("Fourier transforming positions")
        print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
    qx, qy, qrad, ptsx, ptsy=makeQrad(dq,qmax,nq)
    #print " After Qrad" 
    fourierval=np.zeros((nq,nq),dtype=complex)
    
    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = len(useparts)
    for kx in range(nq):
        for ky in range(nq):
            # And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
            fourierval[kx,ky]=np.nansum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1])))/N
    plotval=N*(np.real(fourierval)**2+np.imag(fourierval)**2)
    
    # Produce a radial averaging to see if anything interesting happens
    nq2=int(2**0.5*nq)
    valrad=np.zeros((nq2,))
    for l in range(nq2):
        valrad[l]=np.nanmean(plotval[ptsx[l],ptsy[l]])#, axis=0)
    
    if verbose:
        plt.figure()
        plt.pcolor(qx,qy,plotval, vmin=0, vmax=3,shading='auto' )
        plt.colorbar()
        plt.title('Static structure factor (2d)')
        
        plt.figure()
        plt.plot(qrad,valrad,'.-r',lw=2)
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('Static structure factor (radial)')
        
    return qrad,valrad
  
#use all
def FourierTransVel(data,qmax=0.3,whichframe=1,usetype=[0,1],verbose=True, L = 800):
    L = data.param.Lx
    # Note to self: only low q values will be interesting in any case. 
    # The stepping is in multiples of the inverse box size. Assuming a square box.
    dq=2*np.pi/L

    nq=int(qmax/dq)
    qx, qy, qrad, ptsx, ptsy=makeQrad(dq,qmax,nq)
    fourierval=np.zeros((nq,nq,2),dtype=complex)

    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = len(useparts)#data.Nvals[whichframe]

    for kx in range(nq):
        for ky in range(nq):
            fourierval[kx,ky,0]=np.nansum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1]))*data.vval[whichframe,useparts,0])/N
            fourierval[kx,ky,1]=np.nansum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1]))*data.vval[whichframe,useparts,1])/N 
    
    # Sq = \vec{v_q}.\vec{v_-q}, assuming real and symmetric
    # = \vec{v_q}.\vec{v_q*} = v
    Sq = N*(np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2+np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2)

    # Produce a radial averaging to see if anything interesting happens
    nq2=int(2**0.5*nq)
    Sqrad=np.zeros((nq2,))
    for l in range(nq2):
        Sqrad[l]=np.nanmean(Sq[ptsx[l],ptsy[l]])
    
    plotval_x=np.sqrt(np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2)
    plotval_y=np.sqrt(np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2)
    
    # Produce a radial averaging to see if anything interesting happens
    valrad=np.zeros((nq2,2))
    for l in range(nq2):
        valrad[l,0]=np.nanmean(plotval_x[ptsx[l],ptsy[l]])
        valrad[l,1]=np.nanmean(plotval_y[ptsx[l],ptsy[l]])

    if verbose:
        plt.figure()
        plt.plot(qrad,Sqrad,'.-r',lw=2)
        plt.xlabel('q')
        plt.ylabel('correlation')
        plt.title('Fourier space velocity correlation')
    return qrad,valrad,Sqrad

# Real space velocity correlation function
# Note that this can work in higher dimensions. Uses geodesic distance, i.e. on the sphere if necessary

def getVelcorrSingle(data,dx,xmax,whichframe=1,usetype='all',verbose=True):
    # start with the isotropic one - since there is no obvious polar region
    # and n is not the relevant variable, and v varies too much
    # print("Velocity correlation function for frame " + str(whichframe))
    npts=int(round(xmax/dx))
    bins=np.linspace(0,xmax,npts)
    velcorr=np.zeros((npts,))
    velcount=np.zeros((npts,))
    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = sum(useparts)
    velav=np.nanmean(data.vval[whichframe,useparts,:],axis=0)

    for k in range(N):
        
        vdot=np.dot(data.vval[whichframe,useparts,:],data.vval[whichframe,useparts,:][k])
        
        ##Discretise spatially wrt particle distance 
        #ApplyPeriodicBC and take norm
        dr = ApplyPeriodic2d(data, data.rval[whichframe,useparts,:]- data.rval[whichframe,useparts,:][k])
        dr = np.linalg.norm(dr,axis=1)
        #calculate number of bins
        drbin=(np.round(dr/dx)).astype(int)
        #binning velocity correlations based on interparticle distance
        for l in range(npts):
            pts=np.nonzero(drbin==l)[0]
            velcorr[l]+=vdot[pts].sum()
            velcount[l]+=len(pts)
    
    isdata=[index for index, value in enumerate(velcount) if value>0]
    #connected correlation fcn
    velcorr[isdata]=velcorr[isdata]/velcount[isdata] #- np.dot(velav,velav)
    if verbose:
        fig=plt.figure()
        isdata=[index for index, value in enumerate(velcount) if value>0]
        plt.loglog(bins[isdata],velcorr[isdata],'.-r',lw=2)
        #plt.show()
        plt.xlabel("r-r'")
        plt.ylabel('Correlation')
        plt.title('Spatial velocity correlation')
    return bins,velcorr


