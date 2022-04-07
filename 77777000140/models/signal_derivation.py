





def compute_discrete_difference(signal, order, dt):
 
    spd = (signal[1:] - signal[:-1])/dt
    
    if order==1:
        return spd
    
    elif order==2:

        return (spd[1:] - spd[:-1])/dt
    