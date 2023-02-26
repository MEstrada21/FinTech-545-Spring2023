
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



def calculate_Normal_VaR(DailyReturnsMeansAdj, alpha):
    # Calculate the mean and standard deviation of the returns
    mu = np.mean(DailyReturnsMeansAdj)
    sigma = np.std(DailyReturnsMeansAdj)
    
    # Calculate the alpha-percentile VaR using the normal distribution
    VaR = -(norm.ppf(alpha) * sigma - mu)*100

    def plot_normal_distribution_with_var(DailyReturnsMeansAdj, alpha):
        """
        Plots the normal distribution of the given data along with the Value at Risk (VaR) calculated
        using the specified alpha level.

        Args:
        DailyReturnsMeansAdj (pandas.DataFrame): A pandas DataFrame containing the data.
        alpha (float): The significance level for which to calculate VaR.

        Returns:
        None
        """
        # Calculate mean and standard deviation of daily returns
        mu = DailyReturnsMeansAdj.mean() * 100
        sigma = DailyReturnsMeansAdj.std() * 100

        # Generate x values for plotting the normal distribution
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

        # Calculate VaR using the inverse cumulative distribution function (ppf) of the normal distribution
        VaR = -norm.ppf(alpha, loc=mu, scale=sigma)

        # Plot the normal distribution
        plt.plot(x, norm.pdf(x, mu, sigma), 'k-', label='Normal Distribution')

        # Plot the VaR on the normal distribution plot
        plt.axvline(x=-VaR, color='red', linestyle='--', label=f'VaR ({100*(1-alpha)}%)')

        # Add a legend and axis labels
        plt.legend()
        plt.xlabel('Returns')
        plt.ylabel('Probability Density')

        # Show the plot
        plt.show()
    

    return VaR, plot_normal_distribution_with_var(DailyReturnsMeansAdj, alpha)



def plot_normal_distribution_with_var(data, alpha):
    """
    Plots the normal distribution of the given data along with the Value at Risk (VaR) calculated
    using the specified alpha level.

    Args:
    data (pandas.DataFrame): A pandas DataFrame containing the data.
    alpha (float): The significance level for which to calculate VaR.

    Returns:
    None
    """
    # Calculate mean and standard deviation of daily returns
    mu = data.mean() * 100
    sigma = data.std() * 100

    # Generate x values for plotting the normal distribution
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

    # Calculate VaR using the inverse cumulative distribution function (ppf) of the normal distribution
    VaR = norm.ppf(alpha, loc=mu, scale=sigma)

    # Plot the normal distribution
    plt.plot(x, norm.pdf(x, mu, sigma), 'k-', label='Normal Distribution')

    # Plot the VaR on the normal distribution plot
    plt.axvline(x=-VaR, color='red', linestyle='--', label=f'VaR ({100*(1-alpha)}%)')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')

    # Show the plot
    plt.show()


def calcExpWeightCovarVaROneStock(StockReturn, alpha =.05, lam = .94):
    StockReturnReshaped = np.reshape(StockReturn, (len(StockReturn), 1))
    def ewCovar(StockReturnReshaped, lam):
        m, n = StockReturnReshaped.shape
        w = np.zeros(m)

        # Remove the mean from the series
        StockReturnReshapedmean = np.mean(StockReturnReshaped, axis=0)
        StockReturnReshaped = StockReturnReshaped - StockReturnReshapedmean

        # Calculate weight. Realize we are going from oldest to newest
        for i in range(m):
            w[i] = (1 - lam) * lam**(m-i-1)

        # Normalize weights to 1
        w = w / np.sum(w)

        #covariance[i,j] = (w # x)' * x  where # is elementwise multiplication.
        return np.dot(StockReturnReshaped.T, w[:, None] * StockReturnReshaped)

    # Calculate the exponentially weighted covariance matrix
    ewCovarMatrix = ewCovar(StockReturnReshaped, lam)
    normal = norm(0,np.sqrt(ewCovarMatrix))
    VaR = -normal.ppf(alpha)*100


    return VaR