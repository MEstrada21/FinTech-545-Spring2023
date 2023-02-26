import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from scipy.stats import describe
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import t
from scipy.integrate import quad
from scipy.stats.mstats import gmean
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg
import pprint as pprint
import time as time
from sklearn.decomposition import PCA















def fit_normal(x):
    x = x.to_numpy()
    # Mean and Std values
    m = np.mean(x)
    s = np.std(x)
    
    # Create the error model
    error_model = norm(m, s)
    
    # Calculate the errors and U
    errors = x - m
    u = error_model.cdf(x)
    
    # Define the evaluation function
    def eval_func(u):
        return error_model.ppf(u)
    
    # Return the fitted model
    return {"errors": errors, "u": u, "eval": eval_func, "model": error_model}



def fit_normal(x):
    x = x.to_numpy()
    # Mean and Std values
    m = np.mean(x)
    s = np.std(x)
    
    # Create the error model
    error_model = norm(m, s)
    
    # Calculate the errors and U
    errors = x - m
    u = error_model.cdf(x)
    
    # Define the evaluation function
    def eval_func(u):
        return error_model.ppf(u)
    
    # Return the fitted model
    return {"errors": errors, "u": u, "eval": eval_func, "model": error_model}


# NormDistFitted = fit_normal(data)

# num_simulations = 1000
# simulated_value = NormDistFitted['model'].rvs(num_simulations)


def VaRFittedNormal(data, alpha = .05):
    NormDistFitted = fit_normal(data)
    var = -NormDistFitted['model'].ppf(alpha)
    pdf = NormDistFitted['model'].pdf
    def ES_func_norm(x):
        return x * pdf(x)
    ES, _ = quad(ES_func_norm, -np.inf, var)
    ES = -ES / alpha
    return {"VaR": var, "ES": ES}

# # Calculate VaR
# alpha = 0.05 
# var = -NormDistFitted['model'].ppf(alpha)
# pdf = NormDistFitted['model'].pdf

def general_t_ll(mu, s, nu, x):
    td = t(nu)
    log_pdf = td.logpdf((x - mu) / s) - np.log(s)
    return -np.sum(log_pdf)

def fit_general_t(x):
    x = x.to_numpy()
    # Approximate values based on moments
    start_m = np.mean(x)
    start_nu = 6.0 / stats.kurtosis(x) + 4
    start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)

    def _gtl(theta):
        return general_t_ll(*theta, x)

    bounds = ((None, None), (1e-6, None), (2.0001, None))
    result = minimize(_gtl, (start_m, start_s, start_nu), bounds=bounds)

    m, s, nu = result.x
    t_dist_fitted = t(nu, loc=m, scale=s)
    return {'m': m, 's': s, 'nu': nu, 'model': t_dist_fitted}