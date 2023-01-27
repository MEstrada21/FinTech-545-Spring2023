import scipy as scipy
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('problem1.csv')

df.head()

print(df)