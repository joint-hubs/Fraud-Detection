# Fraud-Detection
The aim of this project is to build a model that can accurately identify fraudulent or suspicious transactions in financial datasets, which is crucial for ensuring financial security and reducing fraudulent activities.

## Dataset

This dataset contains information on transactions made by various customers with various counter parties. Each row in the dataset represents a single transaction and contains the following variables:

- `customer`: A unique identifier for the customer making the transaction
- `customer_country`: The country where the customer making the transaction is located
- `fraud_flag`: A binary variable indicating whether the transaction has been labeled as fraudulent or suspicious
- `timestamp`: The date and time of the transaction
- `counterparty`: A unique identifier for the counterparty in the transaction
- `counterparty_country`: The country where the counterparty in the transaction is located
- `transaction_type`: A categorical variable indicating the type of transaction (e.g., payment, transfer, billing)
- `currency`: The currency used in the transaction
- `amount`: The amount of money involved in the transaction (in the currency specified in the currency variable).

Download the raw file from Google Drive: https://drive.google.com/drive/folders/1qXdsZYb3NNhb3V_ABAoy0wAO0EZyIU-S

## Repository Structure

The repository contains a collection of Jupyter notebooks and Python scripts that were used to train and evaluate a machine learning model for fraud detection in financial transactions. Necessary dependencies are listed in the requirements.txt file.

The main files in the repository are:

- `src/funs.py`: This Python script contains customized functions used throughout the project.
- `src/1. Currencies.ipynb`: This notebook collects the currency conversion rates from the static table on the website and saves the data to a CSV file that is used in the solution.
- `src/2. Analysis.ipynb`: This notebook contains an initial data exploration and analysis to gain an understanding of the data. It includes steps such as identifying the number of fraud cases, performing a chi-squared test to identify variables associated with fraud, and using visualization techniques to identify factors within variables associated with fraud. One-hot encoding and ANOVA F-test techniques are also applied to the data in this notebook.
- `src/3. Tree Based Models.ipynb`: This notebook builds and explains the decision tree and XGBoost models. The notebook includes steps such as performance evaluation, feature importances, and plotting the decision tree.
- `src/4. Linear Programming Model.ipynb`: This notebook contains an algorithm based on dictionaries and performs a series of steps to find the best thresholds for expected fraud probability, standard deviation flags, and quantile flags. The notebook evaluates the model using the training and test data sets.

In addition to these files, there is also a `requirements.txt` file that lists all the necessary dependencies for running the project.

The project's workflow is shown in the following diagram:

![](img/flow_diagram.png)