# -*- coding: utf-8 -*-



# # Project Day: Real Estate Price Prediction 🏠🏘️🪴
# 
# This notebook is a part of a final project in **The Data Master** program by xLab Digital. 


# # 1. Introduction
# 
# **Objectives** 
# 
# The goal of this report is to highlight:
# As this is the final project, I will focus on
# - Explore multiple feature engineering methods
# - Apply scikit-learn Pipeline to create pipeline for preprocessing and column transformer
# - Implement linear regression model to predict house prices using the Ames Housing Dataset


# # 2. Import Library


## import library

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from category_encoders import MEstimateEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# -- 3. Dataset --

# # 3. Dataset
# 
# Let's take a glimpse at dataset. The dataset contains information about 1460 houses with 82 features. In this case, our target value for prediction is **SalePrice** which is the property's sale price in dollars.


# **Raw Dataset**


# Loading the dataset
data = pd.read_csv("/data/notebook_files/house-price-prediction.csv")
data

# ## What information does the dataset gather for each house 🏘️?


print('All features that dataset contains:')
for column in data.columns:
    print('\t', column) # '\t' is there only for readability ( '\t' = indentation tab)

# -- 4. Exploring the Dataset 📊 --

# # 4. Exploring the Dataset 📊
# 
# First have a look at the basic information and statistics about the dataset.


data.describe()

data.info()

data

# ## How many unique values in each categorical and numeric feature?


cat = data.select_dtypes(["object"]).nunique().sort_values(axis=0)
cat= pd.DataFrame(cat, columns = ['n_unique'])
cat

num = data.select_dtypes(exclude=["object"]).nunique().sort_values(axis=0)
num= pd.DataFrame(num, columns = ['n_unique'])
num

data["Neighborhood"].value_counts()

# ## Exploring the connection between `'SalePrice'` and other features


sns.relplot(
    x="YearBuilt", y="SalePrice", data=data, height=6,
);

# mean sale price
sold_data = data.copy()

# Define a function that maps month number to quarter number
def get_quarter(month):
    return (month - 1) // 3 + 1

sold_data['quarter'] = sold_data['MoSold'].apply(get_quarter)

sold_data = sold_data[['MoSold','quarter','SalePrice','SaleCondition','YrSold']]

avg_sales = sold_data.groupby(['YrSold','MoSold','quarter']).agg(avg_price=('SalePrice', np.mean))

avg_sales

# pass custom palette:
ax = sns.lineplot(x='MoSold', 
             y='avg_price',
             hue='YrSold', 
             palette="icefire",
             data= avg_sales)

# set the x-axis tick labels
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr',
                    'May', 'June','Jul','Aug','Sep','Oct','Nov','Dec'])

# set the x and y axis labels
plt.xlabel('Month Sold')
plt.ylabel('Average Price')
plt.title('Average SalePrice by Month')

# Put the legend out of the figure
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='YrSold')
# show the plot
plt.show()

avg_sales_quarter = avg_sales.groupby(['YrSold','quarter']).agg(avg_price=('avg_price', np.mean))
avg_sales_quarter

ax = sns.lineplot(x='quarter', 
             y='avg_price',
             hue='YrSold',
             palette="icefire",
             data= avg_sales_quarter)
plt.xlabel('quarter')

# set the x-axis tick labels
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])

# set the x and y axis labels
plt.xlabel('Quarter')
plt.ylabel('Average Price')
plt.title('Average SalePrice by Quarter')


# Put the legend out of the figure
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='YrSold')

# show the plot
plt.show()

sales_con = data.groupby(['YrSold','MoSold','SaleCondition']).agg(avg_price=('SalePrice', np.mean))
sales_con

plt.suptitle('Average Sales by Sale Condition')
# Create a 3x2 subplot grid
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16,12), sharey=True)

sale_conditions = ['Family', 'Normal', 'Partial', 'Abnorml', 'AdjLand', 'Alloca']

# Set legend title
legend_title = 'YrSold'
# Iterate over the SaleConditions and plot each one in a separate subplot
for i, condition in enumerate(sale_conditions):
    
    # Get the data for this SaleCondition
    condition_data = sales_con.loc[(slice(None), slice(None), condition), :]
    
    # Plot the data in the current subplot
    ax = axes[i//2, i%2]
    sns.lineplot(x='MoSold', y='avg_price', hue='YrSold', 
                 palette="icefire",data=condition_data, ax=ax)
    
    # Set the title for the current subplot
    ax.set_title(condition)
    
# Move the legend outside the plot and set the title
plt.legend(title='YrSold', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set the y-axis label for the leftmost subplots
axes[0, 0].set_ylabel('avg_price')

# Set the x-axis label for the bottom subplots
axes[1, 0].set_xlabel('MoSold')

    
# Set the overall title for the figure
fig.suptitle('Average Sale Price by Month and Year, by Sale Condition')
plt.show()

# ### Data Insight 💡
# 
# The graph shows interesting aspects of 'SaleCondition' as follows:
# - The average 'SalePrice' of Normal sales is within 100,000 to 200,000 dollars range
# - On the other hand, the Abnormal sales show the highest peak in July,2007 which the sales price is higher than 700,000 dollars   


# filter data to only include 'Normal' and 'Abnormal' SaleConditions
sales_con_filtered = sales_con.loc[sales_con.index.get_level_values('SaleCondition').isin(['Normal', 'Abnorml'])]

# create subplot for each SaleCondition
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

# plot data for each SaleCondition
for i, cond in enumerate(['Normal', 'Abnorml']):
    # filter data to only include current SaleCondition
    data_cond = sales_con_filtered.xs(cond, level='SaleCondition')
    
    # plot line chart for each year
    sns.lineplot(x='MoSold', y='avg_price', hue='YrSold', palette='icefire', data=data_cond, ax=axes[i])
    
    # set title and adjust legend
    axes[i].set_title(f'SaleCondition: {cond}')
    axes[i].legend(title='Year Sold', loc='upper left', bbox_to_anchor=(1.02, 1))
    
# set y-axis label for the first subplot
axes[0].set_ylabel('Average Price')
    
# set overall title for the figure
fig.suptitle('Average Price by Month Sold for Normal and Abnormal SaleConditions')
    
# adjust spacing between subplots
fig.tight_layout()

# ## Explore the most famous 'Neighborhood'


from lets_plot import *
ggplot(data) + geom_bar(aes(x="Neighborhood", y="..count..")) + labs(title ="Histogram of Neighborhood")


fig, ax =plt.subplots(1,2, figsize=(16,4))
sns.distplot(data['Neighborhood'])
#sns.distplot(np.log1p(y_train), ax=ax[1])
plt.title('SalePrice Distribution')

# ### Data Insight 💡
# The 'Neighborhood' feature is the physical locations within Ames city limits. 
# - The highest house sold is in `NAmes` or Northwest Ames for **225** houses
# - On the other hand, only **2** houses were sold in `Blueste` or Bluestem
# 
# 
# Note that this data covered the year from 2006 to 2010


# ## Price distribution in Neighborhood


plt.figure(figsize=(8, 12))
sns.boxplot(x='SalePrice',y='Neighborhood', data=data, orient='h')

plt.figure(figsize=(8, 12))
sns.boxplot(x='GrLivArea',y='Neighborhood', data=data, orient='h')

from lets_plot import *
ggplot(data) + geom_bar(aes(x="GrLivArea", y="..count.."))

from lets_plot import *
ggplot(data) + geom_bar(aes(x="OverallCond", y="..count..")) + labs(title ="Histogram of OverallCond")


from lets_plot import *
ggplot(data) + geom_bar(aes(x="OverallQual", y="..count..")) + labs(title ="Histogram of OverallQual")


from lets_plot import *
ggplot(data) + geom_bar(aes(x="BedroomAbvGr", y="..count..")) + labs(title ="Histogram of Bedroom number")


from lets_plot import *
ggplot(data) + geom_bar(aes(x="TotRmsAbvGrd", y="..count..")) + labs(title ="Histogram of Total rooms above grade")

# --  5. Preprocessing --

# #  5. Preprocessing


# Before proceeding to the next step with feature engineering, I will use scikit-learn pipelines to avoid any data leakage (as such train-test contamination). Let's check for duplicated values and some outliers before partition this dataset.


# clone the dataset
df = data.copy()
# Ames Housing dataset: 1460 rows x 81 features
df.shape

# ### Check for Duplicated Values


## checking if there is any duplicate data

#df.duplicated().sum()

idsUnique = len(set(df.Id))
idsTotal = df.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total house entries")

# Drop Id column
df.drop("Id", axis = 1, inplace = True)

# ### Check for Outliers


# According to the documentation of [Ames, Iowa by Dean De Cock](https://jse.amstat.org/v19n3/decock.pdf), there are outliers present in the dataset which match with the insight that I found while exploring.
# 
# Plot `'SalePrice'` vs. `'GrLivArea'` to show those extreme values


from lets_plot import *
ggplot(df) + geom_point(aes(x="GrLivArea", y="SalePrice"))

sns.relplot(
    x="GrLivArea", y="SalePrice", data=df,  alpha=0.5,color="red"
);

# In the left plot, at the bottom right there are two very large houses that sold for a really cheap price. According to the document it's recommended to remove any house with more than 4000 square feet from the dataset. The plot after remove those values is shown in the right.


# remove house with 'GrLivArea'  > 4000
df1 = df[df.GrLivArea < 4000]

sns.relplot(
    x="GrLivArea", y="SalePrice", data=df1, alpha=0.5,color="skyblue"
);

# seprate my target from predictors
y = df1.SalePrice
X = df1.drop(['SalePrice'], axis=1)

# Partition my data into training and validation subsets
# use Train:Test = 80:20
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

print("X_train_full : " + str(X_train_full.shape))
print("X_valid_full : " + str(X_valid_full.shape))
print("y_train : " + str(y_train.shape))
print("y_valid : " + str(y_valid.shape))

## label columns based on contained data type 
# select categorical columns
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
## just in case I want to explore feature engineerings and etc.
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.select_dtypes("object")

# As **SalePrice** is the target variable for prediction. Let's explore the distribution of our target 


fig, ax =plt.subplots(1,2, figsize=(16,4))
sns.distplot(y_train, ax=ax[0])
sns.distplot(np.log1p(y_train), ax=ax[1])
plt.title('SalePrice Distribution')

# The `'SalePrice'` variable is right skewed. After log-transformation is applied, our target becomes more normally distributed as shown in the right-plot above. Also, taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally. 


# Log transformation of the taget variable
# np.log1p(x) = log(1+x)
y_log_train = np.log1p(y_train)
y_log_valid = np.log1p(y_valid)


print("y_log_train : " + str(y_log_train.shape))
print("y_log_valid : " + str(y_log_valid.shape))



sns.boxplot(x= df.GrLivArea, data=df)

plt.show()

# ## 5.1 Feature Engineering
# 
# Let's define the preprocessing steps
#   
# **Columns Transformer**
# 
# - Dealing with Missing values
# - Data transformation
# - Encoding ordering values in categorical variable e.g.`'BsmtQual'`, `'ExterCond'`


# ### Missing Values
# **Calculate percentage of missing data**
# 
# Let's explore the missing values in our full dataset. In this case, `df1` is used to shown the percentage of missing values by feature. 


# create function to calculate missing percentage
def percent_missing(df):
    percent_nan = 100* df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan>0].sort_values()
    return percent_nan

percent_nan = percent_missing(df1)
percent_nan = percent_nan.sort_values(ascending =False)
# visualize the result
fig, ax = plt.subplots(figsize=(10, 12))
sns.barplot(y=percent_nan.index,x=percent_nan, orient='h', palette="icefire")
ax.bar_label(ax.containers[0],fmt='%.1f')
plt.ylabel('Features', fontsize=15)
plt.xlabel('% missing values', fontsize=15)
plt.title('Percentage of missing data by feature', fontsize=15)
#plt.xticks(rotation=90);

# ### Data Insight 💡
# 
# In this % missing values by feature plot, it shows some interesting characteristics of the residential homes in Ames, Iowa as follows:
# - Almost every Houses(*99.5%*) has no Pool
# - *96.3%* of houses have no other features (Miscellaneous)
# - *93.8%* of houses have no Alley access
# - *80.8%* of houses have no Fence
# - 47.3 *%* of houses have no Fireplace
#   
# ALso, other features such as Garage the data description say that NA means "No Garage" therefore
# - Only *5.5%* of houses have no Garage
# - And only *2.5%* of houses have no basement
# 
# As I know about the percentage of missing data, I can further select the imputation method that suitable for the data type, which will be shown in **Data Transformation and Imputation** session. 
#  


# ### Data Correlation
# 
# I will firstly create a correlation heatmap to see how our numeric features correlated to the 'SalePrice' variable.


# Correlation map to see how features are correlated with SalePrice
# concat my train data 
concat_train = pd.concat([X_train, y_log_train], axis=1)
corrmat = concat_train .corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True, cmap="icefire")

# Store the correlation score for later use
corr_score =  concat_train.corr()['SalePrice'].sort_values(ascending=False)
corr_score


# Create a bar plot of the sorted correlations
plt.figure(figsize=(8, 10))
ax = sns.barplot(x=corr_score[1:], y=corr_score.index[1:], orient='h',palette="icefire")
plt.title('Correlation between Numeric Features and SalePrice')
plt.xlabel('Correlation')
plt.ylabel('Features')

# Annotate the bars with the corresponding correlation values
for i, v in enumerate(corr_score[1:]):
    ax.text(v + 0.01, i + 0.1, f"{v:.2f}", color='black')

plt.show()

# ### Data Insight 💡
# 
# **Top 5 Positive Correlations**
# 
# - 'OverallQual'
# - 'GrLivArea'
# - 'GarageCars'
# - 'GarageArea'
# - 'TotalBsmtSF'
# 
# These features imply that the relationships between them and 'SalePrice' are positive. For example, if the 'OverallQual' or Rates the overall material and finish of the house (1-10 scale: 1-Very Poor, 10-Very Excellence) is increasing, the 'SalePrice' will also gradually increase based on their correlation score that is almost 1, which really make sense.
# 
# **Negative Correlation**
# 
# Most of the house features seem to have positive relationship with 'SalePrice' and only a few features have negative relation such as
# - BsmtFinSF2
# - BsmtHalfBath
# - OverallCond
# - MiscVal
# - YrSold
# - LowQualFinSF
# - MSSubClass
# - KitchenAbvGr
# - EnclosedPorch
# 
# Note that as the correlation metrix only explain the relationship between two numeric features (In this case one is 'SalePrice'). Therefore, I suspect that there might be some interesting aspects between the categorical features and 'SalePrice' also. And another note is that ***Correlation does not imply causation***


# ### Data Transformation and Imputation
# 
# I will define some preprocessing steps that will contain in my pipeline
# 
# **Categorical Columns**
# - with `SimpleImputer` impute missing values with 'None' as I learnt the meaning of those NA values 
# - apply `OrdinalEncoding` in categorical data
# 
# **Numeric Columns**
# - with `SimpleImputer` impute missing values with 0 
# - apply `StandardScaler` to scale my features as they're in different scales e.g. 'LotFrontage' vs 'LotArea'


# 
# 
# My partition data right now are:
# - `X_train` : (1164, 79)
# - `X_valid` : (292, 79)
# - `y_log_train` : (1164,)
# - `y_log_valid` : (292,) 


print("Number of numerical features:",len(numerical_cols))
print("Number of categorical features:",len(categorical_cols))

# define normalcat
ord_cat = ['KitchenQual', 'FireplaceQu','HeatingQC','PoolQC','Fence','Electrical',
           'GarageQual','GarageCond','GarageFinish',
           'ExterQual','ExterCond','PavedDrive',
           'CentralAir',
           'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

# ### Create Pipeline and Preprocessing Step
# 
# At first, I try to select the threshold for my correlation. But the rmse of the validation set is not affect much and increase a little bit. Therefore, I will set my threshold to zero and drop no columns. 
# 
# In this session, I will create Pipeline for the transformation steps I mentioned above. Also, I will compose my columns transformation with `ColumnTransformer`.




# import more libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PowerTransformer, LabelEncoder 

# Specify the column that will use
# Find the columns with correlation score less than thershold
thershold = 0.0
drop_cols = list(corr_score[abs(corr_score) < thershold].index)

# Drop the columns from numerical and categorical column lists
num_cols = [col for col in numerical_cols if col not in drop_cols]
cat_cols = [col for col in categorical_cols if col not in drop_cols]

#cat1 = [col for col in cat_cols if col in ord_cat]
#cat2 = [col for col in cat_cols if col not in ord_cat]

# Preprocessing for categorical data
# ordinal data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=None)),
    ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Preprocessing for numerical data

numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant')),
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

#print(drop_cols)
print("Pipeline is setup")



# # 6. Training Models


# ### Define Model
# As I already setup my pipeline in preprocessing steps, in this session I will compose it with my seleted models as follows 
# 
# - Linear Regression (base model)
# - Random Forest 🌲
# - XGBoost


# ### 6.1 Linear Regression


## linear regression
model = LinearRegression()

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_log_train)

# Preprocessing of validation data, get predictions
train_preds = my_pipeline.predict(X_train)
preds = my_pipeline.predict(X_valid)

## Evaluate the model
print("Linear Regression")
print("Model Performance on train data")

# Evaluate model performace on train data

train_rmse = mean_squared_error(y_log_train, train_preds, squared=False)
train_mse = mean_squared_error(y_log_train, train_preds, squared=True)
train_r2 = r2_score(y_log_train, train_preds)
print("train rmse = ", train_rmse) 
#print("train mse = ", train_mse) 
#print("train r2 = ", train_r2) 
# Evaluate model performace on validation data
# Evaluate model performace on validation data
print("Model Performance on test data")
test_rmse = mean_squared_error(y_log_valid, preds, squared=False)
test_mse = mean_squared_error(y_log_valid, preds, squared=True)
test_r2 = r2_score(y_log_valid, preds)
print("test rmse = ", test_rmse) 
print("test mse = ", test_mse) 
print("test r2 = ", test_r2) 

# ### 6.2 Random Forest 🌲


## Random Forest
model = RandomForestRegressor(n_estimators=50, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_log_train)

# Preprocessing of validation data, get predictions
train_preds = my_pipeline.predict(X_train)
preds = my_pipeline.predict(X_valid)

# Evaluate the model
print("RandomForestRegressor")
# Evaluate model performace on train data
print("Model Performance on train data")
train_rmse = mean_squared_error(y_log_train, train_preds, squared=False)
train_mse = mean_squared_error(y_log_train, train_preds, squared=True)
train_r2 = r2_score(y_log_train, train_preds)
print("train rmse = ", train_rmse) 
print("train mse = ", train_mse) 
print("train r2 = ", train_r2) 
# Evaluate model performace on validation data
print("Model Performance on test data")
test_rmse = mean_squared_error(y_log_valid, preds, squared=False)
test_mse = mean_squared_error(y_log_valid, preds, squared=True)
test_r2 = r2_score(y_log_valid, preds)
print("test rmse = ", test_rmse) 
print("test mse = ", test_mse) 
print("test r2 = ", test_r2) 

# ### 6.3 XGBRegressor


## XGBRegressor
model =XGBRegressor(learning_rate=0.1,
                       n_estimators=5000,
                       max_depth=4,)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_log_train)

# Preprocessing of validation data, get predictions
train_preds = my_pipeline.predict(X_train)
preds = my_pipeline.predict(X_valid)

# Evaluate the model
print("XGBRegressor")
# Evaluate model performace on train data
print("Model Performance on train data")
train_rmse = mean_squared_error(y_log_train, train_preds, squared=False)
train_mse = mean_squared_error(y_log_train, train_preds, squared=True)
train_r2 = r2_score(y_log_train, train_preds)
print("train rmse = ", train_rmse) 
print("train mse = ", train_mse) 
print("train r2 = ", train_r2) 
# Evaluate model performace on validation data
print("Model Performance on test data")
test_rmse = mean_squared_error(y_log_valid, preds, squared=False)
test_mse = mean_squared_error(y_log_valid, preds, squared=True)
test_r2 = r2_score(y_log_valid, preds)
print("test rmse = ", test_rmse) 
print("test mse = ", test_mse) 
print("test r2 = ", test_r2) 


# # 7. Result 📌


# The performance of the **linear regression model** between the train and validation (test) sets is not much different based on Root Mean Squared Error (rmse), unlike the other two, which seem to be overfitted. As a result, I will select the linear regression model, and create some visualizations for model analysis.


## linear regression
model = LinearRegression()

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_log_train)

# Preprocessing of validation data, get predictions
train_preds = my_pipeline.predict(X_train)
preds = my_pipeline.predict(X_valid)

## Evaluate the model
print("Linear Regression")
print("Model Performance on train data")

# Evaluate model performace on train data

train_rmse = mean_squared_error(y_log_train, train_preds, squared=False)
train_mse = mean_squared_error(y_log_train, train_preds, squared=True)
train_r2 = r2_score(y_log_train, train_preds)
print("train rmse = ", train_rmse) 
#print("train mse = ", train_mse) 
#print("train r2 = ", train_r2) 
# Evaluate model performace on validation data
# Evaluate model performace on validation data
print("Model Performance on test data")
test_rmse = mean_squared_error(y_log_valid, preds, squared=False)
test_mse = mean_squared_error(y_log_valid, preds, squared=True)
test_r2 = r2_score(y_log_valid, preds)
print("test rmse = ", test_rmse) 
print("test mse = ", test_mse) 
print("test r2 = ", test_r2) 


# plot residuals
# Define the x-tick labels
xtick_positions = np.arange(10, 14.5, 0.5)
xtick_labels = [str(i) for i in xtick_positions]

plt.scatter(train_preds, train_preds - y_log_train, c = "blue", marker = "o", label = "Training data", alpha=0.5)
plt.scatter(preds, preds - y_log_valid, c = "lightgreen", marker = "o", label = "Validation data", alpha=0.5)
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.xticks(xtick_positions, xtick_labels)  # Set the x-tick positions and labels
plt.show()

# NOTE - The residuals or error between actual and predicted value seem to be normally scattered around the centerline, which means our model was able to capture most of the explanatory information


# ### Actual 'SalePrice' vs. Predicted 'SalePrice'


# Plot predictions
plt.scatter(train_preds, y_log_train, c = "blue", marker = "o", label = "Training data", alpha=0.5)
plt.scatter(preds, y_log_valid, c = "lightgreen", marker = "o", label = "Validation data", alpha=0.5)
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# ## Summary
# 
# The residuals or error between actual and predicted values seems to be normally scattered around the center line, which means our model was able to capture most of the explanatory information. My test rmse is 0.11 with R-squared = 0.92 🥰
# 
# 
# **Next Step**
# - Predict Price by Neighborhood then compare with the actual values




