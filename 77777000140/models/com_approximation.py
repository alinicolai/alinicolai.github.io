

import numpy as np
from scipy.signal import convolve2d



def compute_com_from_cop_LPF_simplified(cop, frequency, pendulum=None, **kwargs):
    
    """ LPF simplified method """
    
    if pendulum is not None:
        C = (2*np.pi)**2 / pendulum
    else:
        C= [4.2,4.2]    #corresp to pendulum coeff (2pi)^2 / 4.2 = 9.4

    freqs = np.fft.rfftfreq(n=len(cop),d=1/frequency)

    com = []
    ndim = len(cop.shape)

    if ndim == 1 :
        
        cop = cop.reshape((-1,1))

    for axis in range(cop.shape[1]):
        fourier = np.fft.rfft(cop[:,axis]) 
        
        phi_coef = 1/(1+C[axis]*(freqs**2))
        
        fourier = fourier * phi_coef

        x = np.fft.irfft(fourier)[:len(cop)]
        com.append(x) 
    
    if ndim == 1 :
        return np.array(com[0])
    else: 
        com = np.array(com).T
        

    return com


#        
def compute_com_from_cop_Boxcar(cop, frequency, time_window=0.6):  

    
    if len(cop.shape)==1:    
        cop = cop.reshape(-1,1)

    size_filter=int(frequency*(2*time_window))
       
    coefs = np.array([1 for k in range(size_filter)]) 

    if len(cop.shape)>1:
        coefs = np.expand_dims(coefs,-1)

    coefs= coefs / np.sum(coefs)
   
    hpt = convolve2d(cop,coefs,mode="same")
    

    hpt[0:1]=cop[0:1]
    for t in range(1,int(time_window*frequency)):
          hpt[t] = np.mean( cop[max(0,t-int(time_window*frequency)):min(len(cop),int(t+time_window*frequency))], axis=0,keepdims=True)
    
    for t in range(len(hpt) - int(time_window*frequency), len(hpt)):
          hpt[t] = np.mean( cop[max(0,t-int(time_window*frequency)):min(len(cop),int(t+time_window*frequency))], axis=0,keepdims=True)
   
    if hpt.shape[1]==1:
        hpt = hpt[:,0]

    return hpt





class InterpForces():

    def __init__(self):
        
        self.forces = None
        self.roots = None
        self.speeds = None
        self.dt = None

    def fit(self,forces,cop,mass,dt=0.01):
        
        self.forces = forces
        self.cop = cop
        self.mass = mass
        self.dt = dt



        roots_0, interpolated_roots_0 = self._find_roots(axis=0)
        roots_1, interpolated_roots_1 = self._find_roots(axis=1)

        self.roots = [roots_0,roots_1]
        self.interpolated_roots = [interpolated_roots_0,interpolated_roots_1]
        self.speeds  = [self._find_root_speed(cop=cop,roots=self.roots,axis=0),self._find_root_speed(cop=cop,roots=self.roots,axis=1)]


    def _find_roots(self,axis=0):
        
         force1d = self.forces[:,axis]
         dif = force1d[1:] * force1d[:-1]
         is_root = dif<0
         

         where = np.arange(len(is_root))
         where = where[is_root]

         interpolated_roots = []

         for u, w in enumerate(where) :

             before = force1d[w]
             after = force1d[w+1]
             
             interpolated_root = w - (before)/(after - before)
             
             interpolated_roots.append(interpolated_root)

         return where, interpolated_roots

    def _integrate_forces(self,time_start,time_stop,axis=0):

        first_int = np.cumsum(self.forces[time_start:time_stop,axis] * (self.dt/self.mass))

        second_int = np.sum( first_int * self.dt)

        return second_int


    def _find_root_speed(self,cop,roots,axis=0):
        roots1d = roots[axis]
        cop1d = cop[:,axis]
        spd1d  = []

        for i in range(len(roots1d)-1):
            r1=roots1d[i]
            r2= roots1d[i+1]

            v = ( cop1d[r2] - cop1d[r1]  - self._integrate_forces(time_start=r1,time_stop=r2,axis=axis) )/ (self.dt*(r2-r1 ))

            spd1d.append(v)

        return spd1d



    def predict(self,time):

        v = np.array([self._predict_per_axis(time,0),self._predict_per_axis(time,1)])
        return v


    def _predict_per_axis(self,time,axis):
        roots1d = self.roots[axis][:-1]
        spd1d = self.speeds[axis]

        if time <= roots1d[0]:
            return self.cop[time,axis]

        argroot = np.argmax(roots1d[roots1d<time])
        root = roots1d[argroot]
        init_spd = spd1d[argroot]


        pred = self.cop[root,axis] + init_spd*(time-root)*self.dt + self._integrate_forces(time_start=root, time_stop=time, axis=axis)
        return pred




    


def compute_com_force_integration(cop, forces,infos, frequency=100, **kwargs):

        m = infos["patient_weight"] #kg

        predictor = InterpForces()
        predictor.fit(forces,cop,m,dt = 1/frequency)

        com = np.array([predictor.predict(i) for i in range(len(cop))])

        return com
