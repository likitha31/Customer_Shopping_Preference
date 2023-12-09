# %%[markdown]

# Customer Shopping Preference
#
# Index
# * Load libraries
# * Data Preparation / Data Cleaning
# * Descriptive statistics
# * Data Visualization
# * Modeling
# * Hypothesis testing
# * Conclusion

# %%
# Load the libraries
import pandas as pd
import numpy as np
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import * 
import warnings 
warnings.filterwarnings("ignore")
import os



# %%
#Load the Dataset
df = pd.read_csv("shopping_trends_with_festivals.csv")
print(df.head())

#%%
print('The shape of the dataset is :', df.shape)

#%%
print('The columns in the dataset are :')
df.columns

#%%
print('The datatype of each column and null count :')
df.info

# %%[markdown]
# Descriptive statistics
# 
# Descriptive statistics is essential tools for summarizing, exploring, and understanding data.
# It provides the basis for further analysis and interpretation.
# To obtain descriptive statistics for our DataFrame, we can use the describe() method in pandas.

df.describe(include='all')
# %%[markdown]
# Add info from descriptive statistics...

# %%[markdown]
# Data cleaning 
#
# Data Cleaning involves identifying and correcting errors or inconsistencies in datasets to improve their quality and reliability. 
# Clean data is essential for accurate analysis and meaningful insights.
#
# We will perform the following data cleaning steps.
# Handling Missing Values:
# %%[markdown]

# * Missing values: 
# 
# can involve removing rows or columns with missing values, imputing missing values using statistical methods, or using more advanced techniques like machine learning-based imputation.
#
#
df.isnull().sum()
#
#
# %%[markdown]

# * Deduplication:
# 
# Identify and remove duplicate records from the dataset to avoid redundancy and ensure accuracy in analysis.
#
#
print(df.duplicated().sum())
#
#
# %%[markdown]

# * Data Transformation:
# 
# Transform data as needed for analysis, such as creating new variables, aggregating data, or applying mathematical transformations.




# %%

# %%
