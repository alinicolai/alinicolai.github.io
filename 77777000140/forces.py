
import numpy as np
from scipy.signal import convolve2d



def get_forces_dict_cop(cop, cop_spd, frequency, com=None, com_spd=None, com_acc=None, cop_acc=None, pendulum=None, coef_eq=None, coef_spd_eq=None):
    

    forces_dict = {"damping" : -cop_spd, 
                  "pull" : -cop,
                  }
    
#
    if com is not None:
#
#        hpt = com.copy()
#        
#        time_window = 2
#        
#        window = frequency*time_window
#        
#        for t in range(1,len(hpt)):
#            
#            hpt[:t] = np.mean(com[max(0,t-window):t])
            
#        pendulum = 8
#        estimated_com = com.copy()
#        for i in range(2):
#            for t in range(2,len(com)):
#                estimated_com[t,i] = com[t-1,i] + (com[t-1,i]-com[t-2,i]) + pendulum*(com[t-2,i]-cop[t-2,i])*0.05**2
#    
            
            
        
        forces_dict.update({
                            "push":com,
                            "com_spd" : com_spd,
                            "local_recall" : (com - cop),
                            "push1":com,
                            "local_recall1":(com-cop),
                            "push2":com,
                            "local_recall2":(com-cop),
                            "intercept":np.ones((len(cop),2)),
                            "local_damping":com_spd-cop_spd,
                            "global_recall":com,
                            "global_damping":com_spd,
                            
                            "cop_acc1" : cop_acc,
                            "cop_acc2" : cop_acc   ,
                            "z_int":np.cumsum(com-cop,axis=0),
                            "cop_spd1":cop_spd,
                            "com_spd1":com_spd,
                            "cop_spd2":cop_spd,
                            "cop_spd":cop_spd,
                            "com_int":np.cumsum(com,axis=0),
                            
                            "push3":com,
                            "push4":com,
                            "push5":com,
                            "push6":com,
                            "push7":com,
                            "pull1":cop,
                            "z":com-cop,
#                            "hpt":hpt,
                            "z1":com-cop,
                            "z2":com-cop,
                            "z3":com-cop,
                            'com_spd2':com_spd,
                            "com_acc1":com_acc,
                            "com_acc2":com_acc,
                            
                            "local_recall_deducted":(com-cop),
                            "global_recall_deducted":com,
                            
                            "double_damping":com_spd-cop_spd
                            

                            
                            })


    if pendulum is not None:
        forces_dict.update({"xi": pendulum*(com - cop )+ (com_acc - pendulum*(com-cop))})


    if coef_eq is not None:
        coef_eq_slope = [coef_eq[i] for i in range(2)]
        forces_dict.update({"eq":-(cop-coef_eq_slope*com)})        
#    if coef_eq is not None:
#        
#        coef_eq_slope = [coef_eq[i][1] for i in range(2)]
#        intercept_eq = [coef_eq[i][0] for i in range(2)]
#
#        forces_dict.update({"eq":- (cop - coef_eq_slope*com - intercept_eq),
#
#                        })
#        
#    if coef_spd_eq is not None:
#        
#        coef_spd_eq_slope = [coef_spd_eq[i][1] for i in range(2)]
#        intercept_spd_eq = [coef_spd_eq[i][0] for i in range(2)]
#        forces_dict.update({
##                            'deriv_eq': -(cop_spd - coef_spd_eq_slope*com_spd - intercept_spd_eq),
#                            
#                            'deriv_eq':-(cop_spd - coef_eq_slope*com_spd - intercept_eq),
##                            'xi': - pendulum*(cop - coef_eq*com) + (com_acc - pendulum*(cop-coef_eq*com)) 
#
#                        })        
     
    if com_acc is not None:

        forces_dict.update({
                            "com_acc" : com_acc,

                            })            

    return forces_dict





def get_forces_dict_com(cop, com, cop_spd, cop_acc, com_spd, com_acc, coef_eq=None):

    forces_dict = {
#                "pendulum" : com-cop  ,
                   "com_spd1":com_spd,
                   "cop_spd1":cop_spd,
                   "pendulum":com-cop,
                   "com_spd2":com_spd,
                   "cop_spd2":cop_spd,
                   "com1":com,
                   "com2":com,
                   "com3":com,
                   "push":com,
                   "push1":com,
                   "com_spd":com_spd,
                   "pull":cop,
                   "z":com-cop,
                   "cop1":cop,
                   "com_int" : np.cumsum(com, axis=0)
                   


                  }
    

#    if coef_eq is not None:
#        
#        coef_eq_slope = [coef_eq[i][1] for i in range(2)]
#        intercept_eq = [coef_eq[i][0] for i in range(2)]
#
#        forces_dict.update({"eq":- (cop - coef_eq_slope*com - intercept_eq)})
#                            

    return forces_dict
 
    


def eq_point(signal,window=3,freq=25,gamma=1.01):


    size_filter=int(freq*window)-1

    coefs = np.array([ gamma**(k+1) for k in range(size_filter)]) 
    if len(signal.shape)>1:
        coefs = coefs.reshape(-1,1)
        
    print(coefs)
        
    coefs = coefs / np.sum(coefs)#gamma*((1-gamma**size_filter)/(1-gamma))
    

    sig_eq_mvt = convolve2d(signal,coefs,mode="full")
    
    sig_eq_mvt=np.concatenate([signal[0:1], sig_eq_mvt[:-size_filter+1]])
    sig_eq = np.cumsum(sig_eq_mvt, axis=0)
#
#
#
#    print("OK")
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots(2)
#    
#    for i in range(2):
#        ax[i].plot(signal[:,i])
#        ax[i].plot(sig_eq[:,i]) 
#        
#        
#        
    return sig_eq