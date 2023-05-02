
import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from scipy.stats import describe
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import t
from scipy.stats import multivariate_normal
from scipy.stats import spearmanr
from scipy.stats import multivariate_t
from scipy.integrate import quad
from scipy.stats.mstats import gmean
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg
import pprint as pprint
import time as time
from sklearn.decomposition import PCA
from numpy.linalg import eig



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
    



def ExponentiallyWeightedCovarMatrix(stock_returns_matrix_port_A,lam = .94):
    weight = np.zeros(len(stock_returns_matrix_port_A))
    for i in range(len(stock_returns_matrix_port_A)):
        weight[len(stock_returns_matrix_port_A)-1-i]  = (1-lam)*lam**i
    weight = weight/sum(weight)
    ret_means = stock_returns_matrix_port_A - stock_returns_matrix_port_A.mean()
    #print(ret_means.T.values.shape)
    #print(np.diag(weight).shape)
    #print(ret_means.values.shape)
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov
    

def GetCorrespondingStockPrices(port_df, daily_price_df):
    stocks = port_df['Stock'].unique()
    stock_dfs = {}
    for stock in stocks:
        stock_df = daily_price_df[['Date', stock]].copy()
        stock_df = stock_df.rename(columns={stock: 'Price'})
        stock_df = stock_df.set_index('Date')
        stock_dfs[stock] = stock_df
    return pd.concat(stock_dfs, axis=1)




def extract_portfolio(file_path, portfolio_name):
    # Load the portfolio file into a DataFrame
    portfolio = pd.read_csv(file_path)
    
    # Extract the A, B, and C portfolios into separate DataFrames
    portfolio_A = pd.DataFrame(portfolio[portfolio['Portfolio'] == 'A'])
    portfolio_B = pd.DataFrame(portfolio[portfolio['Portfolio'] == 'B'])
    portfolio_C = pd.DataFrame(portfolio[portfolio['Portfolio'] == 'C'])
    
    # Create a new DataFrame for the Total portfolio
    portfolio_Total = pd.concat([portfolio_A, portfolio_B, portfolio_C], ignore_index=True)
    portfolio_Total['Portfolio'] = 'Total'
    
    # Append the Total portfolio to the original DataFrame
    portfolio = pd.concat([portfolio, portfolio_Total], ignore_index=True)
    
    # Set the index to the Portfolio column
    portfolio = portfolio.set_index("Portfolio")
    
    # Return the desired portfolio
    return portfolio.loc[portfolio_name]

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


def GetCorrespondingStockPrices(port_df, daily_price_df):
    stocks = port_df['Stock'].unique()
    stock_dfs = {}
    for stock in stocks:
        stock_df = daily_price_df[['Date', stock]].copy()
        stock_df = stock_df.rename(columns={stock: 'Price'})
        stock_df = stock_df.set_index('Date')
        stock_dfs[stock] = stock_df
    return pd.concat(stock_dfs, axis=1)


def CalcReturnsAndAdjustForMean(portfolio,PriceData):
    stockPrices = pd.DataFrame(GetCorrespondingStockPrices(portfolio, PriceData))
    stock_Ret_Matrix= calculateReturns(stockPrices, methodOfCalculation= "Arithmetic")
    mean_returns_port = stock_Ret_Matrix.mean()
    stock_Ret_Matrix_Mean_Adj = stock_Ret_Matrix -mean_returns_port
    return stock_Ret_Matrix_Mean_Adj


def simulate_pca(a, nsim, pctExp=1, mean=[], seed=1234):
    n = a.shape[0]

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = np.zeros(n)
    if len(mean) != 0:
        _mean = mean.copy()

    #Eigenvalue decomposition
    vals, vecs = eig(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    #julia returns values lowest to highest, flip them and the vectors
    flip = np.arange(vals.size - 1, -1, -1)
    vals = vals[flip]
    vecs = vecs[:,flip]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        #figure out how many factors we need for the requested percent explained
        for i in range(posv.size):
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < posv.size:
            posv = posv[:nval]
    vals = vals[posv]

    vecs = vecs[:,posv]

    # print(f"Simulating with {posv.size} PC Factors: {np.sum(vals)/tv*100}% total variance explained")
    B = vecs*np.diag(np.sqrt(vals))

    np.random.seed(seed)
    m = vals.size
    r = np.random.randn(m,nsim)

    out = np.dot(B, r).T
    #Loop over itereations and add the mean
    for i in range(n):
        out[:,i] = out[:,i] + _mean[i]
    return out

def calculate_var(portfolio_df, prices_df, return_matrix, lam=0.94, alpha=0.05):
    holdings = portfolio_df['Holding'].values

    # get list of stocks and holdings for each portfolio
    portfolios = portfolio_df['Portfolio'].unique()
    portfolio_holdings = {}
    portfolio_returns = {}

    for portfolio_name in portfolios:
        stocks = list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])
        portfolio_holdings[portfolio_name] = holdings[portfolio_df['Portfolio'] == portfolio_name]

        portfolio_prices = prices_df[stocks].values
        portfolio_returns[portfolio_name] = np.diff(np.log(portfolio_prices), axis=0)

    cov_matrices = {}

    for portfolio_name, returns in portfolio_returns.items():
        cov_matrix = ExponentiallyWeightedCovarMatrix(return_matrix, lam)
        cov_matrices[portfolio_name] = cov_matrix

    for portfolio_name, returns in portfolio_returns.items():
        portfolio_values = prices_df[list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])].values[-1, :] * portfolio_holdings[portfolio_name]

        # Calculate portfolio expected return
        portfolio_expected_return = np.sum(portfolio_returns[portfolio_name] * portfolio_holdings[portfolio_name])

        # Calculate the standard deviation of the portfolio returns (sigma)
        sigma = np.sqrt(np.diag(cov_matrix))

        # Calculate VaR$
        portfolio_var = -norm.ppf(alpha) * np.sqrt(np.dot(portfolio_values, np.dot(cov_matrices[portfolio_name], portfolio_values)))
        print(f"Portfolio {portfolio_name} VaR$: ${portfolio_var:.2f}")

        # Calculate VaRret
        var_ret = portfolio_expected_return * norm.ppf(alpha)
        print(f"Portfolio {portfolio_name} VaRret: {var_ret:.2f}")


def calculate_var_monte_carlo(portfolio_df, prices_df, return_matrix, lam=0.94, alpha=0.05, num_simulations=100000):
    holdings = portfolio_df['Holding'].values

    # get list of stocks and holdings for each portfolio
    portfolios = portfolio_df['Portfolio'].unique()
    portfolio_holdings = {}
    portfolio_returns = {}

    for portfolio_name in portfolios:
        stocks = list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])
        portfolio_holdings[portfolio_name] = holdings[portfolio_df['Portfolio'] == portfolio_name]

        portfolio_prices = prices_df[stocks].values
        portfolio_returns[portfolio_name] = np.diff(np.log(portfolio_prices), axis=0)

    cov_matrices = {}

    for portfolio_name, returns in portfolio_returns.items():
        cov_matrix = ExponentiallyWeightedCovarMatrix(return_matrix, lam)
        cov_matrices[portfolio_name] = cov_matrix

    for portfolio_name, returns in portfolio_returns.items():
        portfolio_values = prices_df[list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])].values[-1, :] * portfolio_holdings[portfolio_name]

        # Calculate portfolio expected return
        portfolio_expected_return = np.sum(portfolio_returns[portfolio_name] * portfolio_holdings[portfolio_name])

        # Calculate the standard deviation of the portfolio returns (sigma)
        sigma = np.sqrt(np.diag(cov_matrix))

        # Calculate VaR$ using Monte Carlo simulation
        portfolio_returns_mc = np.random.multivariate_normal(portfolio_returns[portfolio_name].mean(axis=0), cov_matrices[portfolio_name], size=num_simulations)
        portfolio_values_mc = np.dot(portfolio_returns_mc, portfolio_values)
        portfolio_var_mc = -np.percentile(portfolio_values_mc, 100 * alpha)
        print(f"Portfolio {portfolio_name} VaR$: ${portfolio_var_mc:.2f}")

        # Calculate VaRret
        var_ret = portfolio_var_mc/ sum(portfolio_values)
        print(f"Portfolio {portfolio_name} VaRret: {var_ret:.2f}")





def calculate_var_monte_carlo_t(portfolio_df, prices_df, return_matrix, lam=0.94, alpha=0.05, num_simulations=100000):
    holdings = portfolio_df['Holding'].values

    # get list of stocks and holdings for each portfolio
    portfolios = portfolio_df['Portfolio'].unique()
    portfolio_holdings = {}
    portfolio_returns = {}

    for portfolio_name in portfolios:
        stocks = list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])
        portfolio_holdings[portfolio_name] = holdings[portfolio_df['Portfolio'] == portfolio_name]

        portfolio_prices = prices_df[stocks].values
        portfolio_returns[portfolio_name] = np.diff(np.log(portfolio_prices), axis=0)

    cov_matrices = {}

    for portfolio_name, returns in portfolio_returns.items():
        cov_matrix = ExponentiallyWeightedCovarMatrix(return_matrix, lam)
        cov_matrices[portfolio_name] = cov_matrix

    for portfolio_name, returns in portfolio_returns.items():
        portfolio_values = prices_df[list(portfolio_df.loc[portfolio_df['Portfolio'] == portfolio_name, 'Stock'])].values[-1, :] * portfolio_holdings[portfolio_name]

        # Calculate portfolio expected return
        portfolio_expected_return = np.sum(portfolio_returns[portfolio_name] * portfolio_holdings[portfolio_name])

        # Calculate the standard deviation of the portfolio returns (sigma)
        sigma = np.sqrt(np.diag(cov_matrices[portfolio_name]))

        # Calculate VaR$ using Monte Carlo simulation
        # portfolio_returns_mc = multivariate_t.rvs(df=cov_matrix.shape[0], loc=portfolio_returns[portfolio_name].mean(axis=0), cov_matrices[portfolio_name], size=num_simulations)
        portfolio_returns_mc = multivariate_t.rvs(df=cov_matrix.shape[0], loc=portfolio_returns[portfolio_name].mean(axis=0), size=num_simulations)

        portfolio_values_mc = np.dot(portfolio_returns_mc, portfolio_values)
        portfolio_var_mc = -np.percentile(portfolio_values_mc, 100 * alpha)
        print(f"Portfolio {portfolio_name} VaR$: ${portfolio_var_mc/10:.2f}")

        # Calculate VaRret
        var_ret = portfolio_var_mc/ sum(portfolio_values)
        print(f"Portfolio {portfolio_name} VaRret: {var_ret/10:.2f}")
        portfolio_es_mc = -np.mean(portfolio_values_mc[portfolio_values_mc <= -portfolio_var_mc])
        print(f"Portfolio {portfolio_name} ES$: ${portfolio_es_mc/10:.2f}")

        # Calculate ESret
        es_ret = portfolio_es_mc/sum(portfolio_values)
        print(f"Portfolio {portfolio_name} ESret: {es_ret/10:.2f}")

def returns_fit(data):
    out = {"example": ["df", "loc", "scale", "dist"]}
    for i in data.columns:
        temp_results = t.fit(data.loc[:,i])
        temp_dist = t(temp_results[0], temp_results[1], temp_results[2])
        temp = [temp_results[0], temp_results[1], temp_results[2], temp_dist]
        out[i] = temp
    return out    


def UMatrix(portfolioReturns,fittedModel):
    u_returns = portfolioReturns.copy()
    for stock in portfolioReturns.columns:
        df, loc, scale, fitted_dist = fittedModel[stock]
        u_returns[stock] = fitted_dist.cdf(portfolioReturns[stock])
    return u_returns

def calculate_spearman_correlation(returnsDataFrame):
    

    spearman_corr, _ = spearmanr(returnsDataFrame)
    spearman_corr_df = pd.DataFrame(spearman_corr, columns=returnsDataFrame.columns, index=returnsDataFrame.columns)
    
    return spearman_corr_df

def simulate_multivariate_normal(portfolio_data, n_samples=1000):
    

    # Calculate the Spearman correlation matrix
    spearman_corr, _ = spearmanr(portfolio_data)
    
    # Create a mean vector (assuming zero mean for simplicity)
    mean_vector = np.zeros(len(portfolio_data.columns))
    
    # Generate samples using the multivariate normal distribution
    simulated_samples = np.random.multivariate_normal(mean_vector, spearman_corr, n_samples)
    
    # Convert the simulated samples to a DataFrame and set the column names
    simulated_samples_df = pd.DataFrame(simulated_samples, columns=portfolio_data.columns)
    
    return simulated_samples_df

# # Example usage:
# # Assuming u_returns is a DataFrame containing the transformed returns in a U distribution
# PortBSimulatedU = simulate_multivariate_normal(PortBMatrixUTest, n_samples=1000)
# # print("Simulated samples:\n", PortBSimulatedU)

def copula_to_u(copula_samples, fitted_models):
    u_samples = copula_samples.copy()
    for column in copula_samples.columns:
        df, loc, scale, _ = fitted_models[column]
        u_samples[column] = t.cdf(copula_samples[column], df, loc, scale)
    return u_samples

def copula_to_original(u_returns, fitted_models):
    original_samples = pd.DataFrame(index=u_returns.index, columns=u_returns.columns)
    
    for column in u_returns.columns:
        # Clip the U-distributed samples to a small range to avoid numerical issues
        clipped_u_returns = np.clip(u_returns[column], 1e-10, 1 - 1e-10)
        
        # Convert the clipped U-distributed samples back to the original t-distribution
        original_samples[column] = fitted_models[column][-1].ppf(clipped_u_returns)
        
    return original_samples

import pandas as pd
import numpy as np
from scipy.stats import t, norm
from numpy.random import multivariate_normal

def calculate_var_Gaussian(prices, alpha=0.05, num_simulations=10000):
    # Calculate returns
    returns = prices.pct_change().dropna()

    # Transform the returns into uniformly distributed variables
    uniform_data = returns.apply(lambda x: t.cdf(x, *t.fit(x)))

    # Transform the uniform variables into normally distributed variables
    norm_data = norm.ppf(uniform_data)

    # Generate random samples from the multivariate normal distribution
    mean = np.zeros(returns.shape[1])
    cov_matrix = np.cov(norm_data, rowvar=False)
    gaussian_copula_samples = multivariate_normal(mean, cov_matrix, num_simulations)

    # Transform the Gaussian copula samples back into the original return space
    uniform_copula_samples = norm.cdf(gaussian_copula_samples)
    t_copula_samples = np.array([t.ppf(uniform_copula_samples[:, i], *t.fit(returns.iloc[:, i])) for i in range(returns.shape[1])]).T

    # Calculate the weights of each asset based on their initial value in the portfolio
    initial_asset_values = prices.iloc[-1]
    initial_portfolio_value = np.sum(initial_asset_values)
    weights = initial_asset_values / initial_portfolio_value

    # Calculate the portfolio returns for each simulation
    portfolio_returns = t_copula_samples @ weights

    # Calculate the VaR at the given confidence level
    VaR = np.percentile(portfolio_returns, alpha * 100)

    return VaR


