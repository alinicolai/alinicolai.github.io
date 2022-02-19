

import numpy as np
from scipy import stats


from regression_diagnostics import ADF_stationarity_test
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import eval_measures

import scipy.linalg as la
import numpy as np
import scipy.optimize as opt


from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def fit_OLS(forces_dict, Y, dt, list_forces, forces_delay={}, dim=2):

    fit_results = [] 

    if len(forces_delay)==0:
        max_delay = 0
    else:
        max_delay = max(forces_delay.values())
        

    if dim==2:
        for axis in range(2):
    
            forces_axis = np.array([forces_dict[f][max_delay-forces_delay[f]:-forces_delay[f],axis] if f in forces_delay and forces_delay[f]>0 else forces_dict[f][max_delay:,axis] for f in list_forces]).T
    
            OLS_results =  OLS_1D(list_forces=list_forces, forces=forces_axis, Y=Y[max_delay:,axis], dt=dt)
    
            fit_results.append(OLS_results)
            
    elif dim==1:
        
        forces = np.array([forces_dict[f][max_delay-forces_delay[f]:-forces_delay[f]] if f in forces_delay and forces_delay[f]>0 else forces_dict[f][max_delay:] for f in list_forces]).T

        fit_results =  OLS_1D(list_forces=list_forces, forces=forces, Y=Y[max_delay:], dt=dt)


    return fit_results



def OLS_1D(list_forces, forces, Y, dt, forces_to_ignore=[]):



    zero_forces = []

    coefs = {}
    confidence_interval = {}

#    constraint_inf = np.ones(len(list_forces))*(-np.inf)
#   
#    if "push" in list_forces:
#        index_push = list_forces.index("push")
##        constraint_inf[index_push] = 0
#    constraint_sup = np.ones(len(list_forces))*(np.inf)
#   
#    result = opt.lsq_linear(forces, Y.copy(), bounds = (0.1, constraint_sup))
#    
#    
#    fitted_params = result.x
    

    Y = Y.reshape(-1,1)
    

    solution = np.linalg.pinv(forces) @ Y
    
    

        
  
#    solution = fitted_params.reshape(-1,1)
 

    if list_forces == ["constant_com_acc"]:
        estimated_Y = forces
    else:
        estimated_Y = forces @ solution
    
    error = np.var(Y - estimated_Y)

    noise_param = np.sqrt(error*dt)
    

    alpha = 0.05
    quantile_student = stats.t.ppf((1 - alpha/2),df=len(Y)-len(list_forces))
    
    sigma_hat = np.sum((Y - estimated_Y)**2)/(len(Y) - len(list_forces) ) #sigma_hat = SCR/(n-p)
    
    
    covar = sigma_hat * ( np.linalg.inv(forces.T @ forces))
    std = np.sqrt(np.diag(covar))
    

    scr = np.sum((Y-estimated_Y)**2)
    sct = np.sum((Y-np.mean(Y))**2)
    R2 = 1-(scr/sct) 
    

    rmse = np.sqrt( np.mean((Y-estimated_Y)**2) )

    n = len(Y)
    k = forces.shape[1]
    adj_r2 = 1 - (1-R2) * ((n-1)/(n-k-1))

    for i, f in enumerate(list_forces):
        coefs[f]=float(solution[i])
        confidence_parameter=std[i]
        stat_test = coefs[f] / confidence_parameter
        ecart_quantile = np.abs(stat_test) - quantile_student
        conf_bounds = quantile_student * std[i]
        confidence_interval[f] = conf_bounds
        
        if coefs[f]<0 or np.abs(coefs[f])-np.abs(confidence_interval[f]) < 0:
            zero_forces.append(f)
        


        
    coefs["sigma"] = noise_param
            
#    coefs["sigma"] = np.std(Y-estimated_Y)
    
    
#    standardized_forces = forces/np.std(forces,axis=0)
    covar = (forces.T @ forces)/len(forces)
    

    weighted_forces = {f : forces[:,u] * coefs[f] for u,f in enumerate(list_forces) }
    
    
    
#    stat_adf, pval_adf, usedlag, nobs, critical_values, icbest = adfuller(Y-estimated_Y)
    
    stat_durbinwatson = durbin_watson(Y-estimated_Y)

    log_likelihood = gaussian_log_likelihood_from_vector(Y-estimated_Y, mean=0, sigma=None)    
    AIC = eval_measures.aic(llf=log_likelihood, nobs=len(Y), df_modelwc=forces.shape[1])


    ols = OLS(endog=Y, exog=forces)
    results = ols.fit()
    
    loglikelihood=results.llf
    
        
    JB, JBpv, skew, kurtosis = jarque_bera(Y - estimated_Y)
    

    fit_infos = {
            
                "coefs":coefs,
                "fitted":Y[:,0],
                "prediction":estimated_Y,
                "confidence_intervals":confidence_interval,
                "r2":R2,
                "adj_r2":adj_r2,
                "rmse":rmse,
                "residuals":Y-estimated_Y,
                "forces":forces,
                "forces_dict":{list_forces[i]:forces[:,i] for i in range(len(list_forces))},
                "weighted_forces":weighted_forces,
                "durbin_watson":stat_durbinwatson[0],
                "zero_forces":zero_forces,
                "AIC":results.aic,
                "loglikelihood":loglikelihood,
                "BIC":results.bic,
#                "ADF":pval_adf,
                "JB":JBpv[0]
                
            
                }

    return fit_infos



def gaussian_log_likelihood_from_vector(r, mean=0, sigma=None):

    if sigma is None:
        sigma = np.std(r)

    L = -0.5*len(r)*np.log(2*np.pi*(sigma**2)) - np.sum((((r-mean)/sigma)**2)/2)

    return L


