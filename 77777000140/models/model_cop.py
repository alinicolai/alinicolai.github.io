


import numpy as np

from models.forces import get_forces_dict_cop
from models.OLS_fit import fit_OLS
from models.signal_derivation import compute_discrete_difference




class ModelCoP():

    def __init__(self, list_forces_cop, fitted_cop_variable="cop_acc", forces_delay_cop={}):

        self.list_forces_cop = list_forces_cop

        self.fitted_cop_variable = fitted_cop_variable

        self.forces_delay_cop = {f:forces_delay_cop[f] if f in forces_delay_cop else 0 for f in self.list_forces_cop}

    def fit(self, cop, frequency):

        cop = cop - np.mean(cop, axis=0)

        ## Prepare signals to fit ##
        
        dt = 1/frequency

        cop_spd = compute_discrete_difference(cop, order=1, dt=dt)
        cop_acc = compute_discrete_difference(cop, order=2, dt=dt)
    
        n_min =  len(cop_acc)

        cop = cop[:n_min]
        cop_spd = cop_spd[:n_min]
        cop_acc = cop_acc[:n_min]

        assert len(cop)==len(cop_spd)==len(cop_acc)

        
        ## Define variable to fit ##
        
        if self.fitted_cop_variable == "cop":
            Y = cop
        elif self.fitted_cop_variable == "cop_acc":
            Y = cop_acc

        ## Compute forces ##
        
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=None, com_spd=None, com_acc=None, frequency=frequency)
        
        ## Fit
        
        self.fit_cop_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, forces_delay=self.forces_delay_cop, list_forces=self.list_forces_cop)

        return self.fit_cop_results
    


 

    def generate(self, true_cop, frequency, length_factor=1):
        
        
    
        start = 2
        dt = 1./(frequency)

        cop = np.zeros((len(true_cop)*length_factor,2))
        cop_spd = np.zeros((len(true_cop)*length_factor,2))
        
        cop[:start] = true_cop[:start]
        cop_spd[:start] = compute_discrete_difference(true_cop, order=1, dt=dt)[:start]

        cop_acc = np.zeros((len(true_cop)*length_factor,2))
        cop_acc[:start-1] = compute_discrete_difference(true_cop, order=2, dt=dt)[:start-1]      



        
        generative_results = {
                            "forces_output_cop":{},
                            }

        max_delay_cop = max(self.forces_delay_cop.values())

        if start < max_delay_cop:
            start = max_delay_cop        

        ## Fit COP 
 
        for t in range(start,len(cop)):
            
            
            

            forces_dict_cop = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=None, com_spd=None, com_acc=None, frequency=frequency)

            estimated_torque = np.array([0,0])

            for f in self.list_forces_cop:
                
                coefs_force = np.array([self.fit_cop_results[axis]["coefs"][f] for axis in range(2)])

                if f in self.forces_delay_cop and self.forces_delay_cop[f]>0:
                    predicted_force = forces_dict_cop[f][t-self.forces_delay_cop[f]-1] * coefs_force
                else:
                    predicted_force = forces_dict_cop[f][t-1] * coefs_force

                estimated_torque = estimated_torque + predicted_force

                generative_results["forces_output_cop"][f] = predicted_force

            # sigma * dBt/dt ~ N(0,(sigma²/dt))
            noise = np.random.randn(2) * np.array([self.fit_cop_results[axis]["coefs"]["sigma"] for axis in range(2)]) * (1/np.sqrt(dt))
            generative_results["forces_output_cop"]["perturbation"] = noise
            
            estimated_torque = estimated_torque + noise
              
            if self.fitted_cop_variable == "cop_acc":

                cop_acc[t-1] = estimated_torque
                dspd_cop = (estimated_torque)*dt
                dpos_cop = cop_spd[t-1]*dt
                cop_spd[t] = cop_spd[t-1] + dspd_cop         
                cop[t]=cop[t-1]+dpos_cop
                                
            elif self.fitted_cop_variable == "cop":
                
                cop[t] = estimated_torque


        return cop


