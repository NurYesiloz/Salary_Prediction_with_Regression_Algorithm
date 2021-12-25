#################### Business Problem ####################
# Salary information and career statistics for 1986
# for shared baseball players' salary estimates.
# Can a machine learning project be realized?


#################### Dataset Story #######################
# This dataset is originally from StatLib at Carnegie Mellon University.
# taken from the library.
# The dataset is a sample of the data used in the 1988 ASA Graphics Section Poster Session.
# is part of it.
# Salary data originally from Sports Illustrated, April 20, 1987.
# 1986 and career statistics, Collier Books, Macmillan Publishing Company, New
# Retrieved from the 1987 Baseball Encyclopedia Update by York
# has been


#################### Task ################################
# Using data preprocessing and feature engineering techniques
# develop a salary forecasting model.


#################### Variables ###########################

# AtBat: Number of hits with a baseball bat during the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points he earned for his team in the 1986-1987 season
# RBI: Number of players a batter had jogged when he hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during a player's career
# CHits: The number of hits the player has taken throughout his career
# CHmRun: The player's most valuable hit during his career
# CRuns: Points earned by the player during his career
# CRBI: The number of players the player has made during his career
# CWalks: The number of mistakes the player has made to the opposing player during their career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with levels E and W indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate in-game
# Assists: Number of assists made by the player in the 1986-1987 season
# Errors: Player's number of errors in the 1986-1987 season
# Salary: The salary of the player in the 1986-1987 season (over thousand)
# NewLeague: A factor with A and N levels showing the player's league at the start of the 1987 season

#################### Importing Libraries ###################
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#################### Loading the Dataset ####################
def load_hitters():
    data = pd.read_csv("...hitters.csv")
    return data
df = load_hitters()

#################### Data Overview ##########################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=10)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Returns the names of categorical, numeric and categorical but cardinal variables in the data set
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


#################### Feature Engineering #####################

def hitters_feature_engineering(dataframe):
    dataframe['NEW_HitRatio'] = dataframe['Hits'] / dataframe['AtBat']
    dataframe['NEW_RunRatio'] = dataframe['HmRun'] / (dataframe['Runs'] + 0.0000001)
    dataframe['NEW_CHitRatio'] = dataframe['CHits'] / dataframe['CAtBat']
    dataframe['NEW_CRunRatio'] = dataframe['CHmRun'] / dataframe['CRuns']

    dataframe["NEW_WALKS_RATE"] = dataframe["Walks"] / dataframe["CWalks"]
    dataframe["NEW_WALKS_RATE"] = dataframe["CHits"] / dataframe["CAtBat"]
    dataframe["NEW_CRUNS_RATE"] = dataframe["CRuns"] / dataframe["Years"]

    dataframe["NEW_CHITS_RATE"] = dataframe["CHits"] / dataframe["Years"]
    Putouts_label = ["littele_helper", "medium_helper", "very_helper"]
    dataframe["NEW_PUTOUTS_CAT"] = pd.qcut(dataframe["PutOuts"], 3, labels=Putouts_label)

    dataframe['NEW_Avg_AtBat'] = dataframe['CAtBat'] / dataframe['Years']
    dataframe['NEW_Avg_Hits'] = dataframe['CHits'] / dataframe['Years']
    dataframe['NEW_Avg_HmRun'] = dataframe['CHmRun'] / dataframe['Years']
    dataframe['NEW_Avg_Runs'] = dataframe['CRuns'] / dataframe['Years']
    dataframe['NEW_Avg_RBI'] = dataframe['CRBI'] / dataframe['Years']
    dataframe['NEW_Avg_Walks'] = dataframe['CWalks'] / dataframe['Years']

    return dataframe

hitters_feature_engineering(df)
grab_col_names(df, cat_th=10, car_th=20)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
check_df(df, head=10)

#################### Outliers #################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    # Determination of the threshold value
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # Outlier detection
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    # Access outliers.
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    # Delete outliers
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
check_df(df, head=10)



for col in num_cols:
    remove_outlier(df, col)




#################### Missing Values ###########################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
df.dropna(inplace=True)
df = df[(df['Salary'].isnull() == False)]

df.isnull().any()



#################### Encoding #################################


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
binary_cols


for col in binary_cols:
    df = label_encoder(df, col)

ohe_cols = [col for col in df.columns if 17 >= df[col].nunique() > 2]
ohe_cols = [col.upper() for col in ohe_cols]
df = one_hot_encoder(df, ohe_cols)
df.head()

#################### Model #######################################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]

y = df["Salary"]
X = df.drop("Salary", axis=1)

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# Automated Hyperparameter Optimization

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}


for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

