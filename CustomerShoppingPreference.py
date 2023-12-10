# %%[markdown]

# # Customer Shopping Preference
#
# # Introduction
# Customer shopping preference refers to the specific choices and tendencies exhibited by individuals when making purchasing decisions. 
# Understanding customer shopping preferences is crucial for businesses as it allows them to tailor their offerings and marketing strategies to align with the desires and needs of their target audience. 
# By collecting and analyzing data related to customer shopping preferences, companies can make informed decisions, improve customer experiences, and ultimately drive higher customer satisfaction and loyalty. 
# The main goal of this project is to perform exploratory data analysis on the data collected and create models to make predictions about customer shopping preference.


# %%[markdown]

# # Index
# * Load libraries
# * Data Preparation / Data Cleaning
# * Descriptive statistics
# * EDA [Exploratory Data Analysis]
# * Modeling
# * Hypothesis testing
# * Conclusion

# %%[markdown]
# Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import * 
import warnings 
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import matplotlib.pyplot as plt



# %%[markdown]
# Load the Dataset
#
# 
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
# # EDA [Exploratory Data Analysis]
# Exploratory Data Analysis (EDA) is a critical phase in the data analysis process that involves the examination and visualization of a dataset to uncover underlying patterns, relationships, and trends. 
# By employing statistical techniques and data visualization tools, EDA helps analysts and data scientists gain a deep understanding of the data's structure and characteristics.
#
#
#
# * The Data Visualization techniques that we will perform in this project are as follows:
#   -  Maps: Map of US colored by season.
#   -  Pie chart: subscription status vs Payment Method.
#   -  Multivariant analysis: item purchased vs category vs rating review.
#   -  Multivariant analysis: Frequency of purchases vs  by subscription vs category.
#   -  correlation : Product category sales vs season



# %%[markdown]
# # Maps
#
# Map of Locations Colored by average Purchase Amount (USD
# Maps play a crucial role in visualizing spatial data and geographical patterns. 
# In the context of data analysis, maps provide an effective means to convey insights related to specific locations. 
# The provided code exemplifies this concept by generating a map that visualizes the average purchase amount in different locations. 
# Beginning with the aggregation of shopping data based on the 'Location' column, the code calculates the mean purchase amount for each location. 
# This aggregated data is then merged with a GeoDataFrame, incorporating geometrical and geographical information about U.S. states. 
# The resulting GeoDataFrame is plotted using Matplotlib and GeoPandas, where each state is color-coded based on its average purchase amount. 
# The code also includes the option to zoom into a specific geographical region, enhancing the ability to focus on particular areas of interest. 
# Through this map, one can readily discern spatial variations in average purchase amounts, providing a clear and intuitive representation of the geographical distribution of shopping trends across different locations.
# 
# 
# 
# 
shopping_data = pd.read_csv("shopping_trends_with_festivals.csv")
aggregated_data = shopping_data.groupby('Location').agg(
    AverageAmount=pd.NamedAgg(column='Purchase Amount (USD)', aggfunc='mean')  # Change 'sum' to 'mean'
).reset_index()

# Read GeoDataFrame
geo_data = gpd.read_file('cb_2018_us_state_500k.shp')

# Merge shopping data with GeoDataFrame
merged_data = geo_data.merge(aggregated_data, left_on='NAME', right_on='Location')

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plot the GeoDataFrame
merged_data.plot(column='AverageAmount', ax=ax, legend=True, cmap='viridis')  # Color by dominant season

# Set xlim and ylim to zoom into a specific region (adjust coordinates accordingly)
ax.set_xlim(-130, -65)  # Adjust these values based on the desired longitude range
ax.set_ylim(24, 50)     # Adjust these values based on the desired latitude range

plt.title('Map of Locations Colored by average Purchase Amount (USD)')
plt.show()


# %%[markdown]
# # Pie chart
#
# subscription status vs Payment Method.



# %%
# # Multivariant analysis
#
# Item purchased vs category vs rating review.


# %%
# # Multivariant analysis
#
# Frequency of purchases vs  by subscription vs category.

# %%
# # Correlation
# 
# Product category sales vs season



