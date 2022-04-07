
#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import argparse
import matplotlib.pyplot as plt

from stato import Stabilogram
import pandas

from com_approximation import compute_com_from_cop_LPF_simplified
from model_cop_com import ModelCoupledCoPCoM
from model_cop import ModelCoP
import shutil


import matplotlib.font_manager
font = {'family' : 'serif'} 
matplotlib.rc('font', **font)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model")
args = parser.parse_args()



model_name = args.model


with open("input_0.txt") as file:
    lines = file.readlines()


#### Test if it is demo file

name_demo_file = None
    
if "Demo file " in lines[0] and lines[0].split()[2] in list_demo_files:
    
    start = 2
    name_demo_file = lines[0].split()[2]
    
else:
    start = 1

if name_demo_file is None:

    time = np.array([lines[i].split(" ")[0] for i in range(start,len(lines))]).astype(float).reshape(-1,1)
    
    cop_ml = np.array([lines[i].split(" ")[1] for i in range(start,len(lines))]).astype(float).reshape(-1,1)
        
    cop_ap = np.array([lines[i].split(" ")[2] for i in range(start,len(lines))]).astype(float).reshape(-1,1)
    
    cop = np.concatenate([cop_ml, cop_ap], axis=1)
      
    stato = Stabilogram()
    
    stato.from_array(array=cop, time=time, filter_=True, resample=20, filter_upper_bound=10)
    
    
    frequency = 20
    
    cop = stato.get_signal(name="ML_AND_AP")
    
    
    
    if model_name=="total_recall":
        
        estimated_com = compute_com_from_cop_LPF_simplified(cop, frequency=frequency)
    
        estimated_com = estimated_com[frequency:-frequency]
        cop = cop[frequency:-frequency]
    
        model_instance = ModelCoupledCoPCoM(list_forces_cop = ["local_recall", "local_damping", "global_recall", "global_damping"], list_forces_com = ["pendulum"])                        
        model_instance.fit(cop=cop, com=estimated_com, frequency=20)
        generated_cop, generated_com = model_instance.generate(true_cop=cop, true_com=estimated_com, frequency=frequency, length_factor=1)
    
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
    
    
    
    fig, ax = plt.subplots(2,2, figsize=(15,10))
    
    generated_time = np.arange(len(generated_cop))/frequency
    
    for i, axis in enumerate(["ML","AP"]):
    
        ax[0,i].plot(np.arange(len(cop))/frequency, cop[:,i])
        ax[0,i].set_title("Measured CoP trajectory ("+axis+" axis)", fontsize=12)
        ax[1,i].set_title("Generated CoP trajectory from the model ("+axis+" axis)", fontsize=12)
    
    
    if model_name=="total_recall":
    
        for i, axis in enumerate(["ML","AP"]):
            ax[1,i].plot(generated_time, generated_cop[:,i])
            ax[1,i].plot(generated_time, generated_com[:,i])
        
    elif model_name=="global_recall":
    
        ax[1,i].plot(generated_time, generated_cop[:,i])
            
    fig.subplots_adjust(hspace=0.4)  
    fig.savefig("plot_model.png")
    plt.close(fig)
    
    
else:
    
    for file in ["parameters_table.csv", "parameters_table.txt", "]
    
    parameters_file = name_demo_file+"parameters_table.txt"
    