#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import argparse
import matplotlib.pyplot as plt
from swarii import SWARII
#from stato import Stabilogram
import pandas
from scipy.signal import butter, filtfilt
from models.com_approximation import compute_com_from_cop_LPF_simplified
from models.model_cop_com import ModelCoupledCoPCoM
from models.model_cop import ModelCoP
import os
from statsmodels.tsa.stattools import adfuller


frequency=25

### BLOBS

# Young healthy : SM 764

# Parkinson premier et 3421

# Proprioceptif 696



import matplotlib.font_manager
font = {'family' : 'serif'} 
matplotlib.rc('font', **font)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model")
args = parser.parse_args()




# def filter_(signal, frequency, lower_bound=0, upper_bound=10, order = 4) -> None:
#     """
#     Filter the stabilogram using a Butterworth filter. Default parameters are the one used in the paper. 
#     """

#     dt = 1 / frequency
#     nyq = 0.5 * frequency
#     low = lower_bound / nyq
#     high = upper_bound / nyq

#     if low == 0 :
#         b, a = butter(order, high, btype='lowpass')
#     elif high == np.inf :
#         b, a = butter(order, low, btype='highpass')
#     else :
#         b, a = butter(order, (low,high), btype='bandpass')

#     y = filtfilt(b, a, signal,axis=0)    
    
#     return y


def resample(signal, time, target_frequency):

    """
    Resample the stabilogram using SWARII, using the parameters recommended in the paper
    """

    signal = np.concatenate([time, signal], axis=1)
    signal = SWARII.resample(data = signal, desired_frequency=target_frequency)

    return signal



model_name = args.model

#for file in ["parameters_table.csv", "parameters_table.txt"]:
#    if os.path.exists(file):
#        os.remove(file)


with open("input_0.txt") as file:
    lines = file.readlines()


time = np.array([lines[i].split(" ")[0] for i in range(1,len(lines))]).astype(float).reshape(-1,1)

cop_ml = np.array([lines[i].split(" ")[1] for i in range(1,len(lines))]).astype(float).reshape(-1,1)
    
cop_ap = np.array([lines[i].split(" ")[2] for i in range(1,len(lines))]).astype(float).reshape(-1,1)

cop = np.concatenate([cop_ml, cop_ap], axis=1)
# original_frequency = int(1/(time[1] - time[0]))
 





#stato = Stabilogram()
#print(original_frequency)

#stato.from_array(array=cop, 
##                 time=time, 
#                 filter_=True, 
#                 original_frequency=original_frequency,
#                 center=True, resample=True, resample_frequency=frequency, filter_upper_bound=10)
##stato.from_array(cop, original_frequency=db.frequency, \
# #                   resample_frequency=target_frequency, resample=True, filter_=True, center=True)



# cop = filter_(cop, frequency=original_frequency, lower_bound=0, upper_bound=10, order= 4)
   
cop = resample(cop, time, target_frequency=frequency)



#cop = stato.get_signal(name="ML_AND_AP")

cop = cop - np.mean(cop, axis=0)




if model_name=="total_recall":
    
    estimated_com = compute_com_from_cop_LPF_simplified(cop, frequency=frequency)

    estimated_com = estimated_com[frequency:-frequency]
    cop = cop[frequency:-frequency]

    model_instance = ModelCoupledCoPCoM(list_forces_cop = ["local_recall", "local_damping", "global_recall", "global_damping"], list_forces_com = ["pendulum"])                        
    model_instance.fit(cop=cop, com=estimated_com, frequency=frequency)
    generated_cop, generated_com = model_instance.generate(true_cop=cop, true_com=estimated_com, frequency=frequency)

elif model_name=="global_recall":
    model_instance = ModelCoP(list_forces_cop = ["pull", "damping"])                        
    model_instance.fit(cop=cop, frequency=frequency)

    generated_cop = model_instance.generate(true_cop=cop, frequency=frequency)

params_model = []

for axis, name_axis in enumerate(["ML","AP"]):
    
    params = model_instance.fit_cop_results[axis]["coefs"]
    
    for key in params:
        params_model.append([key + ", "+ name_axis, params[key]])
            
#
params_pandas = pandas.DataFrame(params_model, columns=["Parameter name", "Value"])



params_pandas.to_csv("parameters_table.csv", index=False)


    
with open("parameters_table.txt", "wt") as f:
    
    for i in range(len(params_model)):
        
        name = params_model[i][0]
        value = params_model[i][1]
        
        f.write("<b>" + name.replace("_", " ") + "</b>" + ": " + "%.2f"%value + "\n\n")



fig, ax = plt.subplots(2,2, figsize=(13,5), sharex=True, sharey=True)

    
time = np.arange(len(cop))/frequency

generated_time = np.arange(len(generated_cop))/frequency

for i, axis in enumerate(["ML","AP"]):

    ax[0,i].plot(time, cop[:,i], color="darkblue")

    ax[0,i].set_title("Centered preprocessed CoP trajectory ("+axis+" axis)", fontsize=10)
    ax[0,i].plot([time[0], time[-1]], [0,0], linestyle="--", color="grey")
    
    ax[1,i].set_title("Model simulated CoP trajectory ("+axis+" axis)", fontsize=10)

    ax[1,i].plot([time[0], time[-1]], [0,0], linestyle="--", color="grey")

    ax[0,i].plot([time[3], time[3]], [ax[1,i].get_ylim()[0],ax[1,i].get_ylim()[1]], linestyle="--", color="grey")
    ax[1,i].plot([time[3], time[3]], [ax[1,i].get_ylim()[0],ax[1,i].get_ylim()[1]], linestyle="--", color="grey")

    ax[0,i].set_xlabel("Time (s)", fontsize=10)
    ax[1,i].set_xlabel("Time (s)", fontsize=10)
    ax[0,i].set_ylabel("Position (cm)", fontsize=10)
    ax[1,i].set_ylabel("Position (cm)", fontsize=10)

    if model_name=="total_recall":

        ax[1,i].plot(generated_time, generated_cop[:,i])

    elif model_name=="global_recall":

        ax[1,i].plot(generated_time, generated_cop[:,i])

fig.subplots_adjust(hspace=0.4)  
fig.savefig("plot_model.pdf")
plt.close(fig)