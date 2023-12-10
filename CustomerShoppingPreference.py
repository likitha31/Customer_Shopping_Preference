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
from sklearn import preprocessing 
import seaborn as sns
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
# Data cleaning 
#
# Data Cleaning involves identifying and correcting errors or inconsistencies in datasets to improve their quality and reliability. 
# Clean data is essential for accurate analysis and meaningful insights.
#
# We will perform the following data cleaning steps.
# Handling Missing Values:
# %%[markdown]

# * Missing and NaN values: 
# 
# can involve removing rows or columns with missing values, imputing missing values using statistical methods, or using more advanced techniques like machine learning-based imputation.
#
#
# There are few NaN values in Festival column lets replace them with no festival

print("The number of missing values :", df.isnull().sum())
print("")
df['Festival'].fillna('no festival', inplace=True)
print(df.head())
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
# Descriptive statistics
# 
# Descriptive statistics is essential tools for summarizing, exploring, and understanding data.
# It provides the basis for further analysis and interpretation.
# To obtain descriptive statistics for our DataFrame, we can use the describe() method in pandas.

df.describe(include='all')

# %%[markdown]
# From descriptive statistics we can observe the following
# - The mean age is approximately 44.07, and the mean purchase amount is approximately 59.76.
# - The minimum age in the dataset is 18, and the minimum purchase amount is 20.
# - The maximum age in the dataset is 70, and the maximum purchase amount is 100.







# %%

new_df = df.copy()


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
#   -  Tree Map: item purchased vs category vs rating review.
#   -  Multivariant analysis: Frequency of purchases vs  by subscription vs category.
#   -  correlation : Product category sales vs season



# %%[markdown]
# # Maps
#
# Map of Locations Colored by average Purchase Amount (USD)
#
#
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

aggregated_data = new_df.groupby('Location').agg(
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
import pandas as pd
import plotly.express as px
#
grouped = new_df.groupby(['Subscription Status', 'Payment Method']).size().reset_index(name='Count')
fig = px.sunburst(
    grouped,
    path=['Subscription Status', 'Payment Method'],
    values='Count',
    title='Payment Method Usage by Subscription Status',
)
fig.update_traces(
    textinfo='label+percent entry',
    insidetextorientation='radial',
    hoverinfo='text+value',
)
fig.show()



# %%[markdown]
# # Tree Map
#
# Item purchased vs category vs rating review.
#
#
# This code conducts a comprehensive analysis of item purchases by category and their associated review ratings.
# Using Pandas and Plotly Express in Python, the code groups a DataFrame by 'Category' and 'Item Purchased', calculating the count and average review rating for each combination.
# The results are visualized in a treemap, where categories and specific items are represented by rectangles, with color indicating the average review rating.
# This visual representation provides an insightful overview of the distribution of purchases, allowing for easy identification of patterns and trends in item preferences across different categories. 
# The treemap serves as an effective tool for conveying complex relationships in a visually intuitive manner.
#
#
import pandas as pd
import plotly.express as px

# First, group by 'Category' and 'Item Purchased' and calculate the size and average 'Review Rating'
grouped = new_df.groupby(['Category', 'Item Purchased']).agg(
    Count=('Item Purchased', 'size'),
    AverageRating=('Review Rating', 'mean')
).reset_index()

# Then, create a treemap using Plotly Express, with the average 'Review Rating' as the color scale
fig = px.treemap(
    grouped,
    path=['Category', 'Item Purchased'],
    values='Count',
    color='AverageRating',
    color_continuous_scale='RdYlGn',  # Red to Green color scale
    title='Item Purchased by Category with Review Rating Scale'
)

fig.update_layout(coloraxis_colorbar=dict(
    title="Average\nReview Rating"
))

fig.show()  

# This will display the treemap with the color scale representing the average review rating

# %%[markdown]
# # Multivariant analysis
#
# Frequency of purchases vs  by subscription vs category.

import pandas as pd
import plotly.express as px
#
grouped = df.groupby(['Subscription Status', 'Payment Method']).size().reset_index(name='Count')
fig = px.sunburst(
    grouped,
    path=['Subscription Status', 'Payment Method'],
    values='Count',
    title='Payment Method Usage by Subscription Status',
)
fig.update_traces(
    textinfo='label+percent entry',
    insidetextorientation='radial',
    hoverinfo='text+value',
)
fig.show()

# %%[markdown]
# # Correlation

label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(new_df['Gender'])
df['Item Purchased'] = label_encoder.fit_transform(new_df['Item Purchased'])
df['Category'] = label_encoder.fit_transform(new_df['Category'])
df['Location'] = label_encoder.fit_transform(new_df['Location'])
df['Size'] = label_encoder.fit_transform(new_df['Size'])
df['Color'] = label_encoder.fit_transform(new_df['Color'])
df['Season'] = label_encoder.fit_transform(new_df['Season'])
df['Subscription Status'] = label_encoder.fit_transform(new_df['Subscription Status'])
df['Shipping Type'] = label_encoder.fit_transform(new_df['Shipping Type'])
df['Discount Applied'] = label_encoder.fit_transform(new_df['Discount Applied'])
df['Promo Code Used'] = label_encoder.fit_transform(new_df['Promo Code Used'])
df['Payment Method'] = label_encoder.fit_transform(new_df['Payment Method'])
df['Frequency of Purchases'] = label_encoder.fit_transform(new_df['Frequency of Purchases'])

plt.figure(figsize=(16,10))
ax = sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


# 
# Product category sales vs season

# %%[markdown]
# # Modeling 
# In our modeling approach, we will employ a combination of K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest algorithms.
# These modeling techniques will be instrumental in exploring the predictability of subscription status based on various independent variables.
# Our objective is to unravel the intricate relationships between different features and the subscription outcome, shedding light on which variables exert the most significant influence in predicting subscription status. 
# By leveraging these diverse modeling approaches, we aim to gain a comprehensive understanding of the interplay between various factors and the likelihood of subscription, ultimately contributing valuable insights to our predictive analytics framework.
#
# * KNN 
# * Logistic regression
# * Random Forest

# %%[markdown]
# # KNN



# %%[markdown]
# # Logistic regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Assuming 'new_df' is your original DataFrame

data = new_df.drop(['Customer ID', 'Random Date'], axis=1)

# Selecting features and target variable
X = data.drop('Subscription Status', axis=1)  # Features
y = data['Subscription Status']  # Target variable

# Convert 'No' to 0 and 'Yes' to 1 in the target variable
y_numeric = y.map({'No': 0, 'Yes': 1})

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Creating a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Creating a logistic regression pipeline with SMOTE
logreg_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),  # Add SMOTE to the pipeline
    ('classifier', LogisticRegression(random_state=42))
])

# Training the logistic regression model with SMOTE
logreg_pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = logreg_pipeline.predict(X_test)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generating a classification report
class_report = classification_report(y_test, y_pred)

# Generating a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve and AUC
y_pred_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Printing the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
print(f"AUC: {roc_auc:.2f}")

# %%[markdown]
# # Feature Selection.
# 
# Feature selection is a critical step in machine learning that involves choosing a subset of relevant features from the original set of variables to improve model performance, reduce overfitting, and enhance interpretability. 
# In the provided code, feature selection is carried out using a Random Forest classifier, a popular ensemble learning method.
# This approach leverages the ability of Random Forest to assign importance scores to features and selects those deemed most influential. 
# The final result is a subset of features that can be considered as the most relevant for predicting subscription status. 
# This process aids in building more efficient and interpretable models by focusing on the most informative variables.
#
#
#
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



# Preprocess the data: fill missing values and encode categorical variables
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(new_df), columns=df.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = data_imputed.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data_imputed[col] = label_encoder.fit_transform(data_imputed[col])

# Define the feature matrix X and the target vector y
X = data_imputed.drop('Subscription Status', axis=1)
y = label_encoder.fit_transform(data_imputed['Subscription Status'])

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the Random Forest classifier
rf.fit(X, y)

# Perform feature selection
selector = SelectFromModel(estimator=rf, prefit=True)
X_new = selector.transform(X)

# Get selected feature names
selected_features = X.columns[(selector.get_support())]
selected_features.tolist()


# %%[markdown]
# # Random Forrest with feature selection
# Random Forest is an ensemble learning algorithm that combines the predictions of multiple decision trees to enhance the overall accuracy and robustness of the model. 
# Each tree in the forest is constructed using a subset of the features and data, and the final prediction is determined by aggregating the results of individual trees. 
# In this provided code, the Random Forest algorithm is applied to a dataset for predicting the 'Subscription Status' based on selected features. 
# The data is split into training and test sets, with 80% used for training and 20% for testing. 
# A Random Forest classifier with 100 trees is initialized and trained on the training set. 
# The model is then used to predict the target variable on the test set. The code calculates the accuracy of the model and generates a classification report, providing detailed metrics such as precision, recall, and F1-score for each class. 
# This information offers a comprehensive evaluation of the Random Forest model's performance in predicting subscription status. 
# The final print statements communicate the accuracy and classification report for easy interpretation and assessment of the model's effectiveness.

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# We have selected features from the previous step. Let's use those to build the Random Forest model.
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate a classification report
class_report = classification_report(y_test, y_pred)

accuracy, class_report

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the model's performance
print(f"Accuracy of the Random Forest model: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)

# %%


# %%[markdown]
# # Random Forest without feature selection
# This is the same model as the previous one but without performing feature selection.

# First, we need to re-run the data preprocessing to ensure the context is correct for the print statements.
# Load the data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer


# Check for missing values and fill them with the mode (most frequent value) for simplicity
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(new_df), columns=df.columns)


# Encode categorical variables
le = LabelEncoder()
categorical_columns = data_imputed.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data_imputed[column] = le.fit_transform(data_imputed[column].astype(str))


# Define the feature matrix X and the target vector y
X = data_imputed.drop(['Subscription Status'], axis=1)
y = le.fit_transform(data_imputed['Subscription Status'].astype(str))


# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# Train the classifier
rf_classifier.fit(X_train, y_train)


# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)


# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the model's performance
print(f"Accuracy of the Random Forest model: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)

y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_prob)

# Print the ROC-AUC
print(f"ROC-AUC Score: {roc_auc:.2f}")



# %%
# # Cross Vlidation 

from sklearn.model_selection import cross_val_score

# ... [previous code for data loading, preprocessing, and Random Forest model setup] ...

# Define the number of folds for cross-validation
n_folds = 5

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=n_folds)

# Print the results of cross-validation
print(f"Cross-Validation Accuracy Scores for {n_folds} folds: {cv_scores}")
print(f"Average CV Accuracy Score: {cv_scores.mean():.2f}")


# %%
