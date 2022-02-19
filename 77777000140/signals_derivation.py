
import numpy as np
from scipy.signal import savgol_filter


def compute_deriv(signal, deriv, dt, method="empirical"):
 
    if method=="empirical" or method=="instantaneous":
        
        spd = (signal[1:] - signal[:-1])/dt
        
        if deriv==1:
            return spd
        
        elif deriv==2:

            return (spd[1:] - spd[:-1])/dt
        
    elif method=="instantaneous":
        
        spd = np.array([[(signal[1,0]-signal[0,0])/dt,(signal[1,1]-signal[0,1])/dt]])
        spd = np.concatenate([spd, (signal[2:]-signal[:-2])/(2*dt)], axis=0)
        
        if deriv==1:
            return spd
        
        elif deriv==2:
            acc = np.array([[(spd[1,0]-spd[0,0])/dt,(spd[1,1]-spd[0,1])/dt]])
            acc = np.concatenate([acc, (spd[2:]-spd[:-2])/(2*dt)], axis=0)
            return acc
            
    
    elif method=="savgol":
        
        if deriv==1:
    
            savgol_spd = np.array([savgol_filter(signal[:,i], window_length=5, polyorder=3, deriv=1, delta=dt) for i in range(2)]).T
            signal_spd = savgol_spd[:-1]
        
#            signal_spd = (savgol_spd[1:] + savgol_spd[:-1])/2
            
            return signal_spd
            
        elif deriv==2:
            
#            savgol_acc = np.array([savgol_filter(signal[:,i], window_length=11, polyorder=5, deriv=2, delta=dt) for i in range(2)]).T
            savgol_acc = np.array([savgol_filter(signal[:,i], window_length=5, polyorder=3, deriv=2, delta=dt) for i in range(2)]).T
            signal_acc = savgol_acc[:-2]


#            signal_acc = savgol_acc[1:-1]
##        
#            signal_acc = signal_acc[:-1]
            return signal_acc

