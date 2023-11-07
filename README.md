# Customer Shopping Preference

**Team 8: Saikrishna, Likhitha, Venkata Keerthana**

## Introduction
Customer shopping preference refers to the specific choices and tendencies exhibited by individuals when making purchasing decisions. Understanding customer shopping preferences is crucial for businesses as it allows them to tailor their offerings and marketing strategies to align with the desires and needs of their target audience. By collecting and analyzing data related to customer shopping preferences, companies can make informed decisions, improve customer experiences, and ultimately drive higher customer satisfaction and loyalty. The main goal of this project is to perform exploratory data analysis on the data collected and create models to make predictions about customer shopping preference.

## SMART Questions
1. How can we develop a logistic regression model to accurately predict customer shopping preferences, including subscription status, preferred payment method, and the use of discounts and promo codes, using demographic and purchase history data within a specified timeframe, in order to enhance marketing strategies and customer satisfaction?
2. What metrics will be employed to measure success or progress in solving the problem, and how will these metrics inform our decision-making process?
3. Do we have access to the necessary data sources, tools, and expertise to conduct the analysis and implement the findings within our resources?
4. Does the project address a pressing issue, such as improving customer satisfaction and revenue growth?
5. What is the expected timeline for completing the data analysis and delivering results or insights?

## Dataset / Variables 
[Link: Customer Shopping Trends Dataset (kaggle.com)](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset/data)

Our dataset, sourced from Kaggle, consists of a rich array of attributes related to customer shopping preferences. It offers essential insights to augment our understanding of the customer base. Comprising 4100 rows and 18 columns, the dataset encompasses the following variables:

- **Customer ID**: Unique identifier for each customer
- **Age**: Age of the customer
- **Gender**: Gender of the customer (Male/Female)
- **Item Purchased**: The item purchased by the customer
- **Category**: Category of the item purchased
- **Purchase Amount (USD)**: The amount of the purchase in USD
- **Location**: Location where the purchase was made
- **Size**: Size of the purchased item
- **Color**: Color of the purchased item
- **Season**: Season during which the purchase was made
- **Review Rating**: Rating given by the customer for the purchased item
- **Subscription Status**: Indicates if the customer has a subscription (Yes/No)
- **Shipping Type**: Type of shipping chosen by the customer
- **Discount Applied**: Indicates if a discount was applied to the purchase (Yes/No)
- **Promo Code Used**: Indicates if a promo code was used for the purchase (Yes/No)
- **Previous Purchases**: The total count of transactions concluded by the customer at the store, excluding the ongoing transaction
- **Payment Method**: Customer's most preferred payment method
- **Frequency of Purchases**: Frequency at which the customer makes purchases (e.g., Weekly, Fortnightly, Monthly)

## Modeling 
In this project, we will construct a logistic regression model to forecast consumer shopping preferences. Logistic regression, a statistical technique, is well-suited for analyzing datasets in which one or more independent variables determine an outcome. It proves valuable for understanding and predicting customer shopping preferences, especially when the outcome of interest is binary or categorical in nature.

