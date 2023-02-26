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