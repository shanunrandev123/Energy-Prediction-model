#%%
import os
import gc
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import kurtosis, skew, pearsonr

##visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

plt.style.use("ggplot")


dtypes = {
            'county':np.int16, 
            'is_business':np.int16, 
            'product_type':np.int16, 
            'target':np.float32, 
            'is_consumption':np.int16,
            'data_block_id':np.int16, 
            'row_id':np.int16, 
            'prediction_unit_id':np.int16
}




# %%
df_gas = pd.read_csv(r'C:\Users\Asus\Downloads\gas_prices.csv')
df_electricity = pd.read_csv(r'C:\Users\Asus\Downloads\electricity_prices.csv')
df_historical_weather = pd.read_csv(r'C:\Users\Asus\Downloads\historical_weather.csv\historical_weather.csv')
df_train = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\submission_project_viz\train.csv', parse_dates=['datetime'])
df_train['datetime'] = pd.to_datetime(df_train['datetime'], format='%Y-%m-%d %H:%M:%S')


df_client = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\submission_project_viz\client.csv')
df_forecast_weather = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\submission_project_viz\forecast_weather.csv')
weather_to_county = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\submission_project_viz\weather_station_to_county_mapping.csv')

# %%


imp_cols = ['county', 'is_business', 'is_consumption', 'product_type']

fig, axes = plt.subplots(1, len(imp_cols), figsize=(50, 10))

# Iterate through important columns and create count plots
for i, e in enumerate(imp_cols):
    sns.countplot(data=df_train, x=e, ax=axes[i])
    axes[i].set_xlabel(e, fontsize=25)
    axes[i].set_ylabel('count', fontsize=25)
    # Set numerical values with 2-digit decimal precision
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, '.2f')))

# Set the title with modified font properties
plt.suptitle('Distribution of Important Features')
plt.tight_layout()




#%%



# df_train[['target', 'is_consumption', 'is_business', 'product_type']]

# %%
def get_stats_num_col(series):
    statistics = {
        "Mean": series.mean().round(2),
        "Max": series.max().round(2),
        "Min": series.min().round(2),
        "Kurtosis": kurtosis(series).round(2),
        "Skewness": skew(series).round(2),
        "Variance": series.var().round(2),
        "Std": np.sqrt(series.var()).round(2)
    }
    
    quantiles = [25, 50, 75]
    
    percentiles = [10,20,30,40,50,60,70,80,90]
    
    quantile_values = np.percentile(series, quantiles)
    
    percentiles_values = np.percentile(series, percentiles)

    statistics.update({f"Quantile {q}": value.round(2) for q, value in zip(quantiles, quantile_values)})
    
    statistics.update({f"Percentile {q}": value.round(2) for q, value in zip(percentiles, percentiles_values)})
    
    statistics.update({"IQR": (statistics['Quantile 75'] - statistics['Quantile 25']).round(2)})
    
    return statistics


get_stats_num_col(df_train['target'].dropna())











#%%





def remove_outliers_iqr(data, column_name):
    """
    Remove outliers from the specified column of the DataFrame using the IQR method and plot boxplots before and after outlier removal.

    Parameters:
    data (DataFrame): Input DataFrame containing the data.
    column_name (str): Name of the column for which outliers are to be removed.

    Returns:
    DataFrame: DataFrame with outliers removed.
    """
    # Calculate quartiles and IQR
    q1 = data[column_name].quantile(0.25)
    q3 = data[column_name].quantile(0.75)
    iqr = q3 - q1
    
    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Remove outliers
    data_without_outliers = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    # Plot boxplots before and after outlier removal side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot before outlier removal
    sns.boxplot(data=data, y=column_name, ax=axs[0])
    axs[0].set_title('Boxplot Before Outlier Removal')
    axs[0].set_xlabel('Original Data')
    
    # Boxplot after outlier removal
    sns.boxplot(data=data_without_outliers, y=column_name, ax=axs[1])
    axs[1].set_title('Boxplot After Outlier Removal')
    axs[1].set_xlabel('Data without Outliers')
    
    plt.tight_layout()
    plt.show()
    
    return data_without_outliers

# Example usage
# Assuming 'df_client' is your DataFrame and 'eic_count' is the name of the column containing the data
# Replace 'df_client' and 'eic_count' with your actual DataFrame and column name
data_without_outliers = remove_outliers_iqr(df_client, 'installed_capacity')



# %%
plt.figure(figsize=(8,4))
plt.hist(df_train['target'], log=True, bins=50)
plt.title('Log Distribution of the column target')
plt.show()
# %%


fig, ax = plt.subplots(figsize=(10,5))

plt.hist(df_train[df_train['is_business'] == 1]['target'], bins=50, color='blue',
         log=True, alpha=0.5, label='IS BUSINESS == 1')

plt.hist(df_train[df_train['is_business'] == 0]['target'], bins=50, 
         log=True, alpha=0.8, label='IS BUSINESS == 0')

ax.set_xlabel('Target Variable')
ax.set_ylabel('Frequency')
plt.legend(loc='upper right')

plt.title('Comparison of Target Variable Distribution by IS BUSINESS')

plt.grid(True, linestyle='--', alpha=0.5)

ax.set_facecolor('#f4f4f4')

# Customize the legend box
legend = ax.legend(frameon=True)

frame = legend.get_frame()

frame.set_facecolor('white')

frame.set_edgecolor('black')
plt.show()
# %%

product_type_mapping = {
    0: "Combined", 
    1: "Fixed", 
    2: "General service", 
    3: "Spot"
}
df_train['product_type_name'] = df_train['product_type'].map(product_type_mapping)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(10,5))

plt.hist(df_train[df_train['product_type_name'] == 'Spot']['target'], bins=50, color='red',
         log=True, alpha=0.5, label='Spot')
plt.hist(df_train[df_train['product_type_name'] == 'Fixed']['target'], bins=50, color='blue',
         log=True, alpha=0.7, label='Fixed')
plt.hist(df_train[df_train['product_type_name'] == 'Combined']['target'], bins=50, color='yellow',
         log=True, alpha=0.7, label='Combined')
plt.hist(df_train[df_train['product_type_name'] == 'General service']['target'], bins=50, color='green',
         log=True, alpha=0.7, label='General service')

ax.set_xlabel('Target Variable')
ax.set_ylabel('Frequency')
plt.legend(loc='upper right')

plt.title('Comparison of Target Variable Distribution by product_type')
plt.grid(True, linestyle='--', alpha=0.5)
ax.set_facecolor('#f4f4f4')

# Customize the legend box
legend = ax.legend(frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10,5))

plt.hist(df_train[df_train['is_consumption'] == 1]['target'], bins=50, color='red',
         log=True, alpha=0.5, label='IS CONSUMPTION == 1')
plt.hist(df_train[df_train['is_consumption'] == 0]['target'], bins=50, 
         log=True, alpha=0.7, label='IS CONSUMPTION == 0')

ax.set_xlabel('Target Variable')
ax.set_ylabel('Frequency')
plt.legend(loc='upper right')

plt.title('Comparison of Target Variable Distribution by IS CONSUMPTION')
plt.grid(True, linestyle='--', alpha=0.5)
ax.set_facecolor('#f4f4f4')

# Customize the legend box
legend = ax.legend(frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.show()

# %%
print(f"There are {df_train['prediction_unit_id'].drop_duplicates().shape[0]} predictions units")
print(f"There are {df_train[['prediction_unit_id', 'is_consumption']].drop_duplicates().shape[0]} combinations of predictions units and consumption")
# %%

df_train['time'] = df_train['datetime'].dt.strftime('%H:%M:%S')


# %%
shift_days = [1,2,3,4,5,6,7]
for shift in shift_days:
    df_train['data_block_id_shifted'] = df_train['data_block_id'] + shift

    df_train = pd.merge(
        df_train,
        (
            df_train[[
                'county', 'is_business','is_consumption','product_type',
                'data_block_id_shifted', 'time', 'target']]
            .rename(columns={
                'data_block_id_shifted':'data_block_id', 
                'target':f'target_{shift}days_ago'
            })
        ),
        on = ['county', 'is_business','is_consumption','product_type', 'data_block_id', 'time'],
        how='left'
    )

    # drop the redundant column
    del df_train['data_block_id_shifted']

# %%

plt.figure(figsize=(8, 3))
sns.heatmap(df_train[['target', 'target_1days_ago', 'target_2days_ago', 'target_3days_ago',
                   'target_4days_ago', 'target_5days_ago',
                   'target_6days_ago', 'target_7days_ago']].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f", 
            linewidths=.5)


# %%

county_mapping = pd.read_json(r"C:\Users\Asus\Downloads\county_id_to_name_map (1).json",
            orient='index').reset_index()
county_mapping.columns = ['county', 'county_name']
#county_mapping['county'] = county_mapping['county'].astype(str)
county_mapping = county_mapping.set_index('county')['county_name'].to_dict()

df_train['county_name'] = df_train['county'].map(county_mapping)
df_train['county_name'].value_counts()
df_train['county_name'].value_counts().sum()
print(len(df_train))

# %%
value_counts = df_train['county_name'].value_counts(normalize=True).reset_index()

value_counts.columns = ['county', 'count']
value_counts = value_counts.sort_values(by='count', ascending=False)
value_counts['count'] = round(value_counts['count'] * 100, 1)

plt.figure(figsize=(6,4))
plt.bar(value_counts['county'], value_counts['count'], edgecolor='black')
# Add labels and title
plt.xlabel('Categories')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=60, ha='right')
plt.ylabel('Values')
plt.title('Normalized counts of counties')

# Add data values on top of the bars
for i, value in enumerate(value_counts['count']):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')

# Add a horizontal grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)




# %%

df_train["target_log1p"] = np.log1p(df_train["target"])

# Create subplots
fig, axes = plt.subplots(figsize=(18, 6), nrows=1, ncols=3)

# PDF
colors = ["blue", "orange"]  # Define colors
sns.histplot(data=df_train, x="target", hue="is_consumption", ax=axes[0], kde=True, palette=colors)
axes[0].set_title("`target` Distribution")

sns.histplot(data=df_train, x="target_log1p", hue="is_consumption", ax=axes[1], kde=True, palette=colors)
axes[1].set_title("`target_log1p` Distribution")

# CDF
sns.histplot(data=df_train, x="target_log1p", hue="is_consumption", ax=axes[2], element="step", fill=False,
             cumulative=True, stat="density", common_norm=False, palette=colors)
axes[2].set_title("`target_log1p` CDF")

plt.tight_layout()
plt.show()



# %%




# %%
# Assuming county_id2name and PRODUCT_TYPE2NAME are dictionaries


# %%


df_client = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\submission_project_viz\client.csv', infer_datetime_format=True)
print(df_client.head())

df_train = pd.merge(df_train, df_client, on=["county", "is_business", "product_type", "data_block_id"], how="left")


# %%
selected_cols = ["installed_capacity", "eic_count", "target"]

# Check if both installed_capacity and eic_count are either both null or both not null
# assert ((df_train["eic_count"].isnull() & df_train["installed_capacity"].notnull()) |
#         (df_train["eic_count"].notnull() & df_train["installed_capacity"].isnull())).all()

# Filter the data where target and installed_capacity are not null
data = df_train[(df_train["target"].notnull()) & (df_train["installed_capacity"].notnull())]

# Separate data into production and consumption
data_prod = data[data["is_consumption"] == 0]
data_cons = data[data["is_consumption"] == 1]

# Calculate correlation matrices
corr_prod = data_prod[selected_cols].corr()
corr_cons = data_cons[selected_cols].corr()

# Set index to columns
corr_prod.index = corr_prod.columns
corr_cons.index = corr_cons.columns

# Create subplots for correlation heatmaps
fig, axes = plt.subplots(figsize=(18, 5), nrows=1, ncols=2)

# Plot correlation heatmap for production
sns.heatmap(corr_prod, cmap="crest", annot=True, ax=axes[0])
axes[0].set_title("Correlation Plot of Production")

# Plot correlation heatmap for consumption
sns.heatmap(corr_cons, cmap="crest", annot=True, ax=axes[1])
axes[1].set_title("Correlation Plot of Consumption")

plt.tight_layout()
plt.show()


# %%
import math

UNIT_ID_COL = "prediction_unit_id"

n_uids, n_cols_per_row = 6, 3
line_colors = {0: (8/255, 209/255, 59/255, 0.5), 1: (245/255, 10/255, 10/255, 0.7)}
sampled_ids = np.random.choice(df_train[UNIT_ID_COL].unique(), size=n_uids, replace=False)

# Create subplots layout
fig, axes = plt.subplots(figsize=(18, 12), nrows=math.ceil(n_uids / n_cols_per_row), ncols=n_cols_per_row, sharex=True)

for i, uid in enumerate(sampled_ids):
    train_uid = df_train[df_train[UNIT_ID_COL] == uid].sort_values(by="datetime")
    for is_cons in [0, 1]:
        train_uid_ = train_uid[train_uid["is_consumption"] == is_cons]
        row, col = i // n_cols_per_row, i % n_cols_per_row
        ax = axes[row, col]
        ax.plot(train_uid_["datetime"], train_uid_["target"], label="Consumption" if is_cons else "Production", color=line_colors[is_cons])
        ax.set_title(f"Plot ({row}, {col}) - UID {uid}")
        ax.set_ylabel("target")
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        ax2.plot(train_uid_["datetime"], train_uid_["installed_capacity"], label="installed_capacity", color="black")
        ax2.set_ylabel("installed_capacity")
        ax2.legend(loc="upper right")

plt.suptitle("Sampled Electricity Sequences with `installed_capacity`", y=1.02)
plt.tight_layout()
plt.show()

# %%
df_gas.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])
# %%
fig = px.line(df_gas, x="origin_date", y=["highest_price_per_mwh", "lowest_price_per_mwh"], title="Oridin Date based Gas Price Comparison")
fig.show()


# %%

df = pd.read_csv(r'C:\Users\Asus\Downloads\enefit_project_train.csv')

selected_columns_pca = ['eic_count', 'installed_capacity',
       'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
       'cloudcover_total_fcast_mean', 'direct_solar_radiation_fcast_mean']


dfx = df[selected_columns_pca]

sns.heatmap(dfx.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# %%


# %%
