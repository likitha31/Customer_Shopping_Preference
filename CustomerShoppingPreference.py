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
df = pd.read_csv("shopping_trends.csv")
df.head()



# %%
