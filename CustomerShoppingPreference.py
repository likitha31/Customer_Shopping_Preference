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
df = pd.read_csv("C:/Users/lkg31/OneDrive/Desktop/Customer_Shopping_Preference/shopping_trends_with_festivals.csv")
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

#%%[markdown]
# # Heatmap
# heatmap of Frequency of purchases VS Subscription Status

cross_table = pd.crosstab(df['Frequency of Purchases'], df['Subscription Status'])
sns.heatmap(cross_table, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Frequency of Purchases by Subscription Status")
plt.xlabel("Subscription Status")
plt.ylabel("Frequency of Purchases")


plt.show()

# %%[markdown]
# product category sales by Season
age_bins = [0, 18, 35, 50, float('inf')]
age_labels = ['0-18', '19-35', '36-50', '51+']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
plt.figure(figsize=(12, 8))
sns.heatmap(df.groupby(['Season', 'Category'])['Purchase Amount (USD)'].sum().unstack(), cmap='viridis', annot=True, fmt=".2f", linewidths=.5)
plt.title('Product Category Sales by Season')
plt.xlabel('Category')
plt.ylabel('Season')
plt.show()

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

aggregated_data = df.groupby('Location').agg(
    AverageAmount=pd.NamedAgg(column='Purchase Amount (USD)', aggfunc='mean')
).reset_index()

# Load the geographic data
geo_data = gpd.read_file('C:/Users/lkg31/OneDrive/Desktop/Customer_Shopping_Preference/cb_2018_us_state_500k.shp')

# Merge the aggregated shopping data with the geographic data
merged_data = geo_data.merge(aggregated_data, left_on='NAME', right_on='Location')

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot the GeoDataFrame
merged_data.plot(column='AverageAmount', ax=ax, legend=True, cmap='viridis', edgecolor='black')

# Add labels for each state
for idx, row in merged_data.iterrows():
    plt.annotate(text=row['NAME'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y), 
                 ha='center', fontsize=8)

# Set xlim and ylim to zoom into a specific region (adjust coordinates accordingly)
ax.set_xlim(-130, -65)  # Adjust these values based on the desired longitude range
ax.set_ylim(24, 50)     # Adjust these values based on the desired latitude range

# Disable grid lines
ax.grid(False)

plt.title('Map of US States Colored by Average Purchase Amount (USD)')
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

# Group the data by 'Frequency of Purchases', 'Subscription Status', and 'Category'
grouped = df.groupby(['Frequency of Purchases', 'Subscription Status', 'Category']).size().reset_index(name='Count')

# Create a sunburst chart to visualize the relationship between frequency of purchases, subscription status, and category
fig = px.sunburst(
    grouped,
    path=['Frequency of Purchases', 'Subscription Status', 'Category'],
    values='Count',
    title='Purchase Frequency by Subscription Status and Category',
    width=800,
    height=800
)

# Show the figure
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
# * Logistic regression
# * Random Forest



# %%[markdown]
# # Logistic regression
#
# The provided Python code utilizes logistic regression to predict a binary target variable, likely related to subscription status. 
# It preprocesses the data by dropping irrelevant columns, converts categorical variables, and splits the dataset for training and testing. 
# The logistic regression model is trained using a pipeline that includes preprocessing steps and oversampling with SMOTE. 
# Performance metrics such as accuracy, a classification report, confusion matrix, and AUC-ROC curve are calculated and visualized. 
# The code showcases a comprehensive approach to handling imbalanced datasets and evaluating the logistic regression model's predictive performance.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Check for missing values and fill them with the mode (most frequent value) for simplicity
mode_imputer = SimpleImputer(strategy='most_frequent')
filled_data = pd.DataFrame(mode_imputer.fit_transform(new_df), columns=new_df.columns)

# Encode categorical variables
encoder = LabelEncoder()
object_columns = filled_data.select_dtypes(include=['object']).columns
for col in object_columns:
    filled_data[col] = encoder.fit_transform(filled_data[col].astype(str))

# Define the feature matrix features_matrix and the target vector target_vector
features_matrix = filled_data.drop(['Subscription Status'], axis=1)
target_vector = encoder.fit_transform(filled_data['Subscription Status'].astype(str))

# Split the data into a training set and a testing set
features_train, features_test, target_train, target_test = train_test_split(features_matrix, target_vector, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
forest_model.fit(features_train, target_train)
predicted_train = forest_model.predict(features_train)

# Calculate accuracy on the training set
training_accuracy = accuracy_score(target_train, predicted_train)
# Make predictions on the testing set
predicted_target = forest_model.predict(features_test)

# Evaluate the model's performance
model_accuracy = accuracy_score(target_test, predicted_target)
report_classification = classification_report(target_test, predicted_target)

print(f"Training Accuracy of the Random Forest model: {training_accuracy:.2f}")
print(f"Testing Accuracy of the Random Forest model: {model_accuracy:.2f}")

# Print the model's performance
print(f"Accuracy of the Random Forest model: {model_accuracy:.2f}")
print("\nClassification Report:")
print(report_classification)

predicted_prob = forest_model.predict_proba(features_test)[:, 1]  # Probabilities for the positive class
roc_value = roc_auc_score(target_test, predicted_prob)

fpr, tpr, thresholds = roc_curve(target_test, predicted_target)
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

# %%
# # 5 Fold Cross Validation

from sklearn.model_selection import cross_val_score

# ... [previous code for data loading, preprocessing, and Random Forest model setup] ...

# Define the number of folds for cross-validation
cv_fold_count = 5

# Perform cross-validation
cv_results = cross_val_score(forest_model, features_matrix, target_vector, cv=cv_fold_count)

# Print the results of cross-validation
print(f"Cross-Validation Accuracy Scores for {cv_fold_count} folds")
print(cv_results)
print(f"Average CV Accuracy Score: {cv_results.mean():.2f}")


# %%
# # Neural Network : MLPClassifier

# Explicitly declaring X_train and y_train for the Neural Network Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Your code here
scaler = StandardScaler()
# Use the scaler object as needed


# Separating features and target variable
X = new_df.drop('Subscription Status', axis=1)
y = new_df['Subscription Status']

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Creating a column transformer with OneHotEncoder for categorical features and StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Applying the transformations
X_transformed = preprocessor.fit_transform(X)

# Splitting the dataset into training and test sets again with the properly transformed features
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Neural Network Classifier
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Predictions and Evaluations
nn_y_pred = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_y_pred)
nn_report = classification_report(y_test, nn_y_pred)
print(f"Accuracy of the Neural Network: {nn_accuracy:.2f}")
print("\nClassification Report:")
print(nn_report)

# %%
import plotly.express as px

# Data for the plot
values = [82.5, 98, 93]
labels = ['Logistic Regression', 'Random Forrest', 'Neural Network']
data = {'Labels': labels, 'Values': values}

# Creating the bar plot with Plotly Express
fig = px.bar(data, x='Labels', y='Values', title='Bar Plot of accuracy')

# Show the plot
fig.show()

# %%
