
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



def calculateReturns(dailyReturns, methodOfCalculation="Arithmetic"):
    if methodOfCalculation == "Arithmetic":
        dailyReturns = dailyReturns.pct_change()
        dailyReturns = dailyReturns.dropna()
        return dailyReturns
    elif methodOfCalculation == "Geometric":
        dailyGeoReturns = np.log(dailyReturns / dailyReturns.shift(1))
        geometricReturns = np.exp(dailyGeoReturns.cumsum()) - 1
        geometricReturns = geometricReturns.dropna()
        return geometricReturns
    else:
        print("Invalid method of calculation. Please choose either 'Arithmetic' or 'Geometric'.")


def simulate_brownian_motion(P0, sigma, n=1000000):
    rdist = norm(0, sigma)
    simR = rdist.rvs(n)
    P1 = P0 + simR
    mean_P1 = np.mean(P1)
    std_P1 = np.std(P1)
    skewness_P1 = skew(P1)
    kurtosis_P1 = kurtosis(P1)
    print(f"expect (u, sigma, skew, kurt) = ({P0}, {sigma}, 0, 0)")
    print(f"({mean_P1}, {std_P1}, {skewness_P1}, {kurtosis_P1})")
    return P1

def simulate_arithmetic_motion(P0, sigma, n=1000000):
    rdist = norm(0, sigma)
    simR = rdist.rvs(n)
    P1 = P0 * (1+ simR)
    mean_P1 = np.mean(P1)
    std_P1 = np.std(P1)
    skewness_P1 = skew(P1)
    kurtosis_P1 = kurtosis(P1)
    print(f"expect (u, sigma, skew, kurt) = ({P0}, {sigma}, 0, 0)")
    print(f"({mean_P1}, {std_P1}, {skewness_P1}, {kurtosis_P1})")
    return P1

def simulate_geometric_motion(P0, sigma, n=1000000):
    rdist = norm(0, sigma)
    simR = rdist.rvs(n)
    P1 = P0 * np.exp(simR)
    mean_P1 = np.mean(P1)
    std_P1 = np.std(P1)
    skewness_P1 = skew(P1)
    kurtosis_P1 = kurtosis(P1)
    print(f"expect (u, sigma, skew, kurt) = ({P0}, {sigma}, 0, 0)")
    print(f"({mean_P1}, {std_P1}, {skewness_P1}, {kurtosis_P1})")
    return P1
    
