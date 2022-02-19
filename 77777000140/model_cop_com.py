


import numpy as np
from forces import get_forces_dict_cop, get_forces_dict_com

from signals_derivation import compute_deriv
from OLS_fit import fit_OLS
from scipy.signal import savgol_filter


np.random.seed(49)


class ModelCoupledCoPCoM():


    def __init__(self, list_forces_cop=[], list_forces_com=[], fitted_com_variable = "com_acc", fitted_cop_variable="cop_acc", forces_delay_cop={}, forces_delay_com={}):

        self.list_forces_cop = list_forces_cop
        
        self.list_forces_com = list_forces_com

        self.fitted_cop_variable = fitted_cop_variable

        self.forces_delay_cop = {f:forces_delay_cop[f] if f in forces_delay_cop else 0 for f in self.list_forces_cop}
        
        self.forces_delay_com = {f:forces_delay_com[f] if f in forces_delay_com else 0 for f in self.list_forces_com}

        self.fitted_com_variable = fitted_com_variable
        
        self.coef_eq=None
        self.coef_spd_eq = None
        self.pendulum=None
        
    def get_ready_data(self, cop, com, frequency, com_acc=None, center=False):
        

        cop = cop - np.mean(cop, axis=0)
        com = com - np.mean(com, axis=0)

        dt = 1/frequency

        cop_spd = compute_deriv(cop, deriv=1, dt=dt)
        cop_acc = compute_deriv(cop, deriv=2, dt=dt)

        if com_acc is None:
            com_acc = compute_deriv(com, deriv=2, dt=dt)

        n_min = min(len(cop_acc), len(com_acc))

        cop_spd = cop_spd[:n_min]
        cop_acc = cop_acc[:n_min]
        cop = cop[:n_min]        

        com_spd = compute_deriv(com, deriv=1, dt=dt)
        com = com[:n_min]
        com_spd = com_spd[:n_min]
        com_acc = com_acc[:n_min] 

        return cop, com, cop_spd, cop_acc, com_spd, com_acc               
    
    def fit_pendulum(self, cop, com, frequency, com_acc=None, center=False):
        
        dt = 1/frequency

        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)

        forces_dict = get_forces_dict_com(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc)
        results = fit_OLS(forces_dict=forces_dict, Y=com_acc, dt=dt, forces_delay={}, list_forces=["pendulum"])
        self.pendulum = np.array([results[i]["coefs"]["pendulum"] for i in range(2)])
        self.sigma_pendulum = np.array([results[i]["coefs"]["sigma"] for i in range(2)])


    def fit_coef_eq(self, cop, com, frequency, com_acc=None, center=False):    
        
        dt = 1/frequency
        
        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)
            
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=None, cop_acc=None, frequency=frequency)
        results = fit_OLS(forces_dict=forces_dict, Y=cop, dt=dt, forces_delay={}, list_forces=["push"])
        self.coef_eq = np.array([results[i]["coefs"]["push"] for i in range(len(results))])


    def fit_coef_int_eq(self, cop, com, frequency, com_acc=None, center=False):   
        
        dt = 1/frequency

        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)
            
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency)
        results = fit_OLS(forces_dict=forces_dict, Y=np.cumsum(cop,axis=0), dt=dt, forces_delay={}, list_forces=["com_int", "intercept"])
        self.coef_int_eq = np.array([(results[i]["coefs"]["intercept"], results[i]["coefs"]["com_int"]) for i in range(len(results))])
            

    def fit(self, cop, com, frequency, com_acc=None, center=False):
        
        self.fit_pendulum(cop=cop, com=com, com_acc=com_acc, frequency=frequency)
        self.fit_coef_eq(cop=cop, com=com, frequency=frequency)
        self.fit_coef_int_eq(cop=cop, com=com, com_acc=com_acc, frequency=frequency)

        if len(self.list_forces_com)>0:
            self.fit_com(com=com, com_acc=com_acc, cop=cop, frequency=frequency)

        if len(self.list_forces_cop)>0:
            self.fit_cop(cop=cop, com=com, com_acc=com_acc, frequency=frequency)
 

    def fit_cop_1D(self, cop, com, frequency, com_acc=None, center=False):
        
        dt = 1/frequency
        
        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)

        ## Define variable to fit ##
        
        if self.fitted_cop_variable == "cop":
            Y = cop
        elif self.fitted_cop_variable == "cop_acc":
            Y = cop_acc
        elif self.fitted_cop_variable == "cop_spd":
            Y = cop_spd

        # Load forces
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency, pendulum=self.pendulum, coef_eq=self.coef_eq, coef_spd_eq=self.coef_spd_eq)

        # Fit
        self.fit_cop_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, forces_delay=self.forces_delay_cop, list_forces=self.list_forces_cop, dim=1)


        return self.fit_cop_results
           
        

    def fit_cop(self, cop, frequency, com=None, com_acc=None, center=False):

        dt = 1/frequency
        
        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)

        ## Define variable to fit ##
        
        if self.fitted_cop_variable == "cop":
            Y = cop
        elif self.fitted_cop_variable == "cop_acc":
            Y = cop_acc
        elif self.fitted_cop_variable == "cop_spd":
            Y = cop_spd

        # Load forces
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency, pendulum=self.pendulum, coef_eq=self.coef_eq, coef_spd_eq=self.coef_spd_eq)


        # Fit
        self.fit_cop_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, forces_delay=self.forces_delay_cop, list_forces=self.list_forces_cop)


        return self.fit_cop_results
    


    def fit_com(self, com, com_acc, cop, frequency, center=False):
        
        dt = 1/frequency

        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc, center=center)

        ## Define variable to fit ##

        if self.fitted_com_variable == "com_acc":
            Y = com_acc
        elif self.fitted_com_variable == "com_spd":
            Y = com_spd
        elif self.fitted_com_variable =="com":
            Y = com
            
        ## Compute forces ##
        forces_dict = get_forces_dict_com(cop=cop, cop_spd=cop_spd, cop_acc=cop_acc, com=com, com_spd=com_spd, com_acc=com_acc, coef_eq=self.coef_eq)
        
        ## Fit
        self.fit_com_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, forces_delay=self.forces_delay_com, list_forces=self.list_forces_com)

        return self.fit_com_results
    


 

    def generate(self, true_cop, true_com, frequency, length_factor=1):
 
        
#        print([self.fit_com_results[i]["coefs"] for i in range(2)])
        
        start = 10
        dt = 1./(frequency)

        max_delay_com = max(self.forces_delay_com.values())

        if len(self.list_forces_cop) > 0:
            max_delay_cop = max(self.forces_delay_cop.values())

            max_delay = max(max_delay_cop, max_delay_com) 
            
        else:
            max_delay = max_delay_com
                
        if start < max_delay:
            start = max_delay
            
        self.generate_start = start


        cop = np.zeros((int(len(true_cop)*length_factor),2))
        cop_spd = np.zeros((int(len(true_cop)*length_factor),2))
        
        cop[:start] = true_cop[:start]
        cop_spd[:start] = compute_deriv(true_cop, 1, dt, method="instantaneous")[:start]
        
        com = np.zeros((int(len(true_cop)*length_factor),2))
        com[:start] = true_com[:start]
        
        com_spd = np.zeros((int(len(true_cop)*length_factor),2))
        com_spd[:start] = compute_deriv(true_com, 1, dt, method="instantaneous")[:start]

        com_acc = np.zeros((int(len(true_cop)*length_factor),2))
        com_acc[:start-1] = compute_deriv(true_com, 2, dt, method="instantaneous")[:start-1]
        
        cop_acc = np.zeros((int(len(true_cop)*length_factor),2))
        cop_acc[:start-1] = compute_deriv(true_com, 2, dt, method="instantaneous")[:start-1]      
        
        self.generative_results = {
                            "forces_output_cop":{},
                            "forces_output_com":{}
                            }
        

        if len(self.list_forces_cop)>0:
            
            exogeneous_cop = False
            
        else:
            exogeneous_cop = True

        for t in range(start,len(cop)+1): 
            
            if exogeneous_cop==False:
    
                ##################
                ## Generate COP ##
                ##################
            
    
                forces_dict_cop = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency, coef_eq=self.coef_eq, pendulum=self.pendulum, coef_spd_eq=self.coef_spd_eq)
    
                estimated_forces_cop = np.array([0,0])
    
                for f in self.list_forces_cop:
                    
                    coefs_force = np.array([self.fit_cop_results[axis]["coefs"][f] for axis in range(2)])
    
                    if f in self.forces_delay_cop and self.forces_delay_cop[f]>0:
                        predicted_force = forces_dict_cop[f][t-self.forces_delay_cop[f]-1] * coefs_force
                    else:
                        predicted_force = forces_dict_cop[f][t-1] * coefs_force
                        
                    estimated_forces_cop = estimated_forces_cop + predicted_force
                    
                    self.generative_results["forces_output_cop"][f] = predicted_force
    
                # sigma * dBt/dt ~ N(0,(sigma²/dt))
                noise = np.random.randn(2) * np.array([self.fit_cop_results[axis]["coefs"]["sigma"] for axis in range(2)]) * (1/np.sqrt(dt))
                
    
                
                
                self.generative_results["forces_output_cop"]["perturbation"] = noise
    
                estimated_forces_cop = estimated_forces_cop + noise
    
    
                if self.fitted_cop_variable == "cop_acc":
        
                    cop_acc[t-1] = estimated_forces_cop
                    dspd_cop = estimated_forces_cop * dt
                    dpos_cop = cop_spd[t-1]*dt    
                    
                    if t<len(cop):
                        cop_spd[t] = cop_spd[t-1] + dspd_cop         
    
    
    
                        ###MODIF
                        
    #                    cop[t] = cop[t-2] + 2*dt * cop_spd[t-1] 
                        cop[t]=cop[t-1]+dpos_cop
    
        
                elif self.fitted_cop_variable =="cop_spd":
                    
                    cop_spd[t-1] = estimated_forces_cop
                    cop_acc[t-2] = (cop_spd[t-1] - cop_spd[t-2])/dt
                    dpos_cop = estimated_forces_cop * dt
        
                    if t<len(cop):
                        cop[t] = cop[t-1] + dpos_cop                               
        
                elif self.fitted_cop_variable == "cop":
        
                    cop[t-1] = estimated_forces_cop
                    cop_spd[t-2] = (cop[t-1] - cop[t-2])/dt         
                    cop_acc[t-3] = (cop_spd[t-2] - cop_spd[t-3])/dt    
                    
                
            ##################
            ## Generate COM ##
            ##################
        
            estimated_forces_com = np.array([0,0])

            for f in self.list_forces_com:

                forces_dict_com = get_forces_dict_com(cop=cop, cop_spd=cop_spd, cop_acc=cop_acc, com=com, com_spd=com_spd, com_acc=com_acc, coef_eq=self.coef_eq)
                
                coefs_force = np.array([self.fit_com_results[axis]["coefs"][f] for axis in range(2)])

                if f in self.forces_delay_com and self.forces_delay_com[f]>0:
                    predicted_force = forces_dict_com[f][t-self.forces_delay_com[f]-1] * coefs_force
                else:
                    predicted_force = forces_dict_com[f][t-1] * coefs_force
     
                estimated_forces_com = estimated_forces_com + predicted_force
                
            

                self.generative_results["forces_output_com"][f] = predicted_force
    
            # sigma * dBt/dt ~ N(0,(sigma²/dt))
            noise = np.random.randn(2) * np.array([self.fit_com_results[axis]["coefs"]["sigma"] for axis in range(2)]) * (1/np.sqrt(dt))
            self.generative_results["forces_output_com"]["perturbation"] = noise


            estimated_forces_com_without_noise = estimated_forces_com
            estimated_forces_com = estimated_forces_com + noise
            
            
            
            if self.fitted_com_variable == "com_acc":
    
                com_acc[t-1] = estimated_forces_com
                dspd_com = (estimated_forces_com)*dt

                dpos_com = com_spd[t-1]*dt 
                
                if t<len(cop):
                    com_spd[t] = com_spd[t-1] + dspd_com 
                    
                    
                    
                    
                    
#                    com[t]=com[t-1]+dpos_com
                    ###MODIF
#                    com[t] = com[t-2] + 2*dt * com_spd[t-1] 
                    com[t]=com[t-1]+dpos_com
                    
                    
  

            elif self.fitted_com_variable=="com_spd":

                com_spd[t-1] = estimated_forces_com
                com_acc[t-2] = (com_spd[t-1] - com_spd[t-2])/dt

                dpos_com = estimated_forces_com * dt

                if t<len(cop):
                    com[t]=com[t-1]+dpos_com    
                    
                    
                    
            if exogeneous_cop:

#                estimated_forces_cop = (forces_dict_com["push"][t-1]*self.pendulum  - com_acc[t-1])/self.pendulum
                estimated_forces_cop = (forces_dict_com["push"][t-1]*self.pendulum - estimated_forces_com_without_noise)/self.pendulum

#                estimated_forces_cop = (forces_dict_com["push"][t-1]*self.pendulum - noise + com_acc[t-1])/self.pendulum
                cop[t-1] = estimated_forces_cop
                cop_spd[t-2] = (cop[t-1] - cop[t-2])/dt         
                cop_acc[t-3] = (cop_spd[t-2] - cop_spd[t-3])/dt    

                
#
#
#
#
#        if exogeneous_cop:
# 
#            for t in range(start,len(cop)+1): 
#    
#                ##################
#                ## Generate COP ##
#                ##################
#            
#    
##                forces_dict_com = get_forces_dict_com(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc)
##                
##
##                forces_dict_cop = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency, coef_eq=self.coef_eq, coef_spd_eq=self.coef_spd_eq, pendulum=self.pendulum)
##
##                estimated_forces_com = np.array([0,0])
##    
##                for f in self.list_forces_com:
##
##                    coefs_force = np.array([self.fit_com_results[axis]["coefs"][f] for axis in range(2)])
##                    
##                    if f in self.forces_delay_com and self.forces_delay_com[f]>0:
##                        predicted_force = forces_dict_com[f][t-self.forces_delay_com[f]-1] * coefs_force
##                    else:
##                        predicted_force = forces_dict_com[f][t-1] * coefs_force
##                        
##                    estimated_forces_com = estimated_forces_com + predicted_force
#
##                estimated_forces_cop = (forces_dict_com["push"][t-1]*self.pendulum - estimated_forces_com) / self.pendulum
#
#                print(com_acc[t-1], estimated_forces_com)



        return cop, com








