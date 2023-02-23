import pandas as pd
import numpy as np
import re
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def chi2_independence(df, factor_col, fraud_col, type = "description"):
    # Variable Statistical Test of Independence
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/
    # The null hypothesis is that there is no association between the two variables, 
    # and the alternative hypothesis is that there is a significant association between them.
    
    # Summarize the data by the given factor and fraud flag
    count_group = df.groupby([factor_col, fraud_col]).size().reset_index(name='n')

    # Create a contingency table
    table = pd.pivot_table(count_group, values='n', index=fraud_col, columns=factor_col)

    # Perform the chi-square test of independence
    chi2, pval, dof, expected = chi2_contingency(table)

    if type == "description":
        # Print the results
        print("Chi-square statistic: ", chi2)
        print("p-value: ", pval)
        print("Degrees of freedom: ", dof)
        print("Expected frequencies: ")
        print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    elif type == "table":
        return chi2_contingency(table)[3]
    

def highlightTable (trxns_data, factor_col, fraud_col):
    # Outlining Factors - chi2_independence based
    # Prepare the data  
    count_group = trxns_data.groupby([factor_col, fraud_col]).size().reset_index(name='n')
    table = pd.pivot_table(count_group, values='n', index=fraud_col, columns=factor_col)
    
    # Calculate expected counts
    expected_counts = chi2_independence(trxns_data, factor_col, fraud_col, type = 'table')

    # Calculate the standardized residuals for each cell and highlight the cells with residuals greater than 2 or less than -2
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            expected = expected_counts[i][j]
            observed = table.iloc[i, j]
            residual = (observed - expected) / np.sqrt(expected)
            if np.abs(residual) > 2:
                table.iloc[i, j] = '{:.0f}*'.format(observed)
            else:
                table.iloc[i, j] = '{:.0f}'.format(observed)
    
    return table


def plotDictionary(trxns_data, colname = 'weekday', quantile_threshold = .9, count_filter = 5):
    # Variable Dictionary Analysis
    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype('category')

    # Summarize the data by a given factor
    count_group = trxns_data.assign(value=trxns_data[colname]).groupby(['value', 'fraud_flag'])['value'].count().reset_index(name='n').assign(prob=lambda x: x['n'] / x.groupby('value')['n'].transform('sum')).reset_index()
    count_group = count_group[count_group['n'] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby('value')['n'].count().reset_index(name='n_')
    count_group_names = count_group_names[count_group_names['n_'] == 2].drop(columns=['n_'])

    # Reshape the data to wide format 
    count_group_wide = count_group.pivot(index='fraud_flag', columns='value', values='n')
    count_group_wide_prob = count_group.pivot(index='fraud_flag', columns='value', values='prob')

    # Bind the data frames
    long_table = pd.concat([count_group_wide.assign(group_type='count'), count_group_wide_prob.assign(group_type='prob')], axis=0, ignore_index=False).reset_index()
    long_table = pd.melt(long_table, id_vars=['fraud_flag', 'group_type'], var_name='name', value_name='value')

    # Prepare the statistics
    table_analysis = long_table[long_table['group_type'] == 'prob'].groupby('fraud_flag').agg(
        mean=pd.NamedAgg(column='value', aggfunc=lambda x: round(np.mean(x), 4)),
        sd=pd.NamedAgg(column='value', aggfunc=np.std),
        q_t=pd.NamedAgg(column='value', aggfunc=lambda x: np.quantile(x, quantile_threshold))
    ).reset_index()

    mean_v = table_analysis.loc[table_analysis['fraud_flag'] == 'Y', 'mean'].values[0]
    sd_v = table_analysis.loc[table_analysis['fraud_flag'] == 'Y', 'sd'].values[0]
    q_v = table_analysis.loc[table_analysis['fraud_flag'] == 'Y', 'q_t'].values[0]

    result_table = long_table[(long_table['fraud_flag'] == 'Y') & (long_table['group_type'] == 'prob') & (long_table['name'].isin(count_group_names['value']))].assign(
        sd_flag=mean_v + sd_v,
        q_flag=q_v + sd_v
    )
    plot_result_table = sns.catplot(x='value', y='name', data=result_table, kind='bar', height=5, aspect=1.5, palette='Spectral')
    plot_result_table.set(xlabel='', ylabel='', title= "Fraud probability by '" + colname + "' factor")
    
    # Add text labels to the bars
    for index in range(len(result_table)):
        plot_result_table.ax.text(result_table.iloc[index]['value'], index, round(result_table.iloc[index]['value'], 4), color='black', ha="center")

    plot_result_table.ax.axvline(x=mean_v, color='k', linestyle='--')
    plot_result_table.ax.axvline(x=mean_v + 1.5*sd_v, color='r', linestyle='--')
    plot_result_table.ax.axvline(x=mean_v + 2*sd_v, color='r', linestyle='--')
    plot_result_table.ax.axvline(x=q_v + sd_v, color='r')
    plt.show()
    


def dataPreparation(all_trxns_path = "all_trxns.csv", exchange_rates_path = "exchange_rates.csv"):
    # Prepare the data
    
    # Data Collection
    # Raw Data
    all_trxns = pd.read_csv(all_trxns_path, dtype={'counterparty': str})

    # Exchange Rates -> more info in currencies.ipynb
    currency_rates = pd.read_csv(exchange_rates_path, header=None, names=["ccy", "date", "rate"])
    
    # Data Cleaning and Preprocessing

    # Transform the variables and provide additional features
    trxns_data = all_trxns.copy()
    # Convert the timestamp to datetime
    trxns_data['timestamp'] = pd.to_datetime(trxns_data['timestamp'], infer_datetime_format=True)
    # Add the date and the exchange rate
    trxns_data['date'] = trxns_data['timestamp'].dt.date
    trxns_data = trxns_data.merge(currency_rates, on=['ccy', 'date'], how='left')
    trxns_data['rate'] = np.where(trxns_data['rate'].isna(), 1, trxns_data['rate'])
    # Clean and convert the amount to EUR
    trxns_data['amount'] = trxns_data['amount'].apply(lambda x: float(re.sub('[^0-9.]', '', x)))
    trxns_data['amount_eur'] = trxns_data['amount'] / trxns_data['rate']
    # Extract the customer type from the customer id
    trxns_data['customer_type'] = trxns_data['customer'].str[0]
    # Extract the weekday, month, quarter and hour from the timestamp
    trxns_data['weekday'] = trxns_data['date'].apply(lambda x: x.strftime('%A'))
    trxns_data['month'] = trxns_data['date'].apply(lambda x: x.strftime('%B'))
    trxns_data['quarter'] = trxns_data['date'].apply(lambda x: 'Q'+str((x.month-1)//3+1))
    trxns_data['hour'] = trxns_data['timestamp'].dt.hour
    # Replace missing values in the "counterparty_country" column with "unknown"
    trxns_data['counterparty_country'] = np.where(trxns_data['counterparty_country'].isna(), "unknown", trxns_data['counterparty_country'])
    # Clean the names of counterparty countries
    trxns_data["counterparty_country"] = np.where(trxns_data["counterparty_country"].isin(["United States", "USA"]), "US", trxns_data["counterparty_country"])

    # Calculate the thresholds for the equally sized buckets of the amount in EUR
    amount_eur_quantile = np.quantile(trxns_data['amount_eur'], q=np.arange(0, 1.2, 0.2))

    # Add amount_eur buckets
    trxns_data['amount_eur_bucket'] = pd.cut(trxns_data['amount_eur'], bins=amount_eur_quantile, include_lowest=True)
    
    return trxns_data




def createMetaDictionary(trxns_data, colname = 'weekday', quantile_threshold = .9, count_filter = 5):
    # Prepare Dictionary Summary row
    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype('category')

    # Summarize the data by a given factor
    count_group = trxns_data.assign(value=trxns_data[colname]).groupby(['value', 'fraud_flag'])['value'].count().reset_index(name='n').assign(prob=lambda x: x['n'] / x.groupby('value')['n'].transform('sum')).reset_index()
    count_group = count_group[count_group['n'] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby('value')['n'].count().reset_index(name='n_')
    count_group_names = count_group_names[count_group_names['n_'] == 2].drop(columns=['n_'])

    # Reshape the data to wide format 
    count_group_wide = count_group.pivot(index='fraud_flag', columns='value', values='n')
    count_group_wide_prob = count_group.pivot(index='fraud_flag', columns='value', values='prob')

    # Bind the data frames
    long_table = pd.concat([count_group_wide.assign(group_type='count'), count_group_wide_prob.assign(group_type='prob')], axis=0, ignore_index=False).reset_index()
    long_table = pd.melt(long_table, id_vars=['fraud_flag', 'group_type'], var_name='name', value_name='value')
    
    # Prepare the statistics
    table_analysis = long_table[long_table['group_type'] == 'prob'].groupby('fraud_flag').agg(
        mean=pd.NamedAgg(column='value', aggfunc=lambda x: round(np.mean(x), 4)),
        sd=pd.NamedAgg(column='value', aggfunc=np.std),
        q_t=pd.NamedAgg(column='value', aggfunc=lambda x: np.nanquantile(x, quantile_threshold))
    ).reset_index()
    
    result_table = (
        table_analysis
        .query('fraud_flag == "Y"')
        .assign(sd_flag=lambda x: x["mean"] + x["sd"],
                q_flag=lambda x: x["q_t"],
                variable_name=colname)
        .loc[:, ["variable_name", "sd_flag", "q_flag"]]
    )
    
    return result_table


def createDictionary(trxns_data, colname = 'weekday', count_filter = 5):

    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype('category')

    # Summarize the data by a given factor
    count_group = trxns_data.assign(value=trxns_data[colname]).groupby(['value', 'fraud_flag'])['value'].count().reset_index(name='n').assign(prob=lambda x: x['n'] / x.groupby('value')['n'].transform('sum')).reset_index()
    count_group = count_group[count_group['n'] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby('value')['n'].count().reset_index(name='n_')
    count_group_names = count_group_names[count_group_names['n_'] == 2].drop(columns=['n_'])

    # Reshape the data to wide format 
    count_group_wide = count_group.pivot(index='fraud_flag', columns='value', values='n')
    count_group_wide_prob = count_group.pivot(index='fraud_flag', columns='value', values='prob')

    # Bind the data frames
    long_table = pd.concat([count_group_wide.assign(group_type='count'), count_group_wide_prob.assign(group_type='prob')], axis=0, ignore_index=False).reset_index()
    long_table = pd.melt(long_table, id_vars=['fraud_flag', 'group_type'], var_name='name', value_name='value')
    
    # Prepare result table
    result_table = long_table.loc[(long_table['fraud_flag'] == "Y") & (long_table['group_type'] == "prob") & (long_table['name'].isin(count_group_names['value'])), :].copy()
    result_table = result_table.drop(['fraud_flag', 'group_type'], axis=1)
    result_table.columns = [colname, colname+"_value"]
    
    return result_table

def evaluateModel(y_test, y_pred):
    
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix   
    cm = confusion_matrix(y_test, y_pred)

    # Create classification report
    # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
    cr = classification_report(y_test, y_pred)
    
    labels = ['0', '1']
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()
    
    print("\nAccuracy: %.2f%%" % (accuracy * 100.0),"\n")
    print(cr)