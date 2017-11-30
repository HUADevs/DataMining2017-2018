import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


def read_data(csv_file):
    # import data from csv
    df = pd.read_csv(csv_file, na_values='?')

    # split to independent features (X) and target feature (y)
    x = df.drop('X65', axis=1)
    y = df['X65']

    return x, y


def impute_missing_values(x, strategy='most_frequent'):

    # impute missing values
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=0)
    print('Imputing missing values with strategy: {strategy}...'.format(strategy=strategy))
    imp.fit(x)
    x = pd.DataFrame(data=imp.transform(x), columns=x.columns)

    return x


def detect_and_remove_outliers(x, y, clf='LOF'):
    classifiers = {
        'LOF': LocalOutlierFactor(n_neighbors=35),
        'IF': IsolationForest(n_estimators=100),
        # 'EE': EllipticEnvelope(assume_centered=False)

    }
    est = classifiers[clf]
    print('Detecting and removing outliers using {classifier}...'.format(classifier=clf))
    if clf == 'LOF':
        outliers = est.fit_predict(x, y)
    else:
        outliers = est.fit(x, y).predict(x)
    #  keep only non-outliers (predicted values outliers=-1, non-outliers=1)
    new_X = x[outliers == 1]
    new_y = y[outliers == 1]

    return new_X, new_y


# probably not needed
def find_outliers_tukey(x):
    quartile_1 = np.percentile(x, 25)
    quartile_3 = np.percentile(x, 75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers_indices = list(x.index[(x > upper_bound) | (x < lower_bound)])
    outliers_values = list(x[outliers_indices])
    return outliers_indices, outliers_values


def normalize_data(x, scaler='Standard'):
    scalers = {
        # StandardScaler not used on sparse data but if needed with_mean=False
        'Standard': StandardScaler(copy=True, with_mean=True, with_std=True),
        # MinMaxScaler and MaxAbsScaler designed for scaling sparse data
        'MinMax': MinMaxScaler(),  # range [0,1]
        'MaxAbs': MaxAbsScaler(),  # range [-1,1]
        # cannot be fitted to sparse inputs
        'Robust': RobustScaler()

    }

    print('Scaling data using {scaler} Scaler...'.format(scaler=scaler))
    x = pd.DataFrame(data=scalers[scaler].fit_transform(x), columns=x.columns, index=x.index.values)

    return x


def select_best_features(x, y, n_features=5, method='chi2'):
    methods = {
        # Compute chi-squared stats between each non-negative feature and class
        'chi2': SelectKBest(chi2, k=n_features),
        # Compute the ANOVA F-value
        'f_classif': SelectKBest(f_classif, k=n_features),
        # measures the dependency between the variables
        # higher values mean higher dependency
        'mutual_info_classif': SelectKBest(mutual_info_classif, k=n_features)
    }

    print('Selecting features using {method}...'.format(method=method))
    x = pd.DataFrame(data=methods[method].fit_transform(x, y), columns=x.columns[methods[method].get_support()], index=x.index.values)

    return x


def plot(x, y):
    df = pd.concat([x, y], axis=1)
    print(df.head())
    sns.pairplot(df, hue='X65')
    plt.show()


def preprocessing():

    x, y = read_data(csv_file='companydata.csv')

    # # check NaN values count by column
    # print('Missing values count:')
    # print(X.isnull().sum().sort_values(ascending=False).head())
    x = impute_missing_values(x, strategy='most_frequent')  # strategies={mean, median, most_frequent}
    # # check NaN values count by column after imputation
    # print('Missing values count after imputer:')
    # print(X.isnull().sum().sort_values(ascending=False).head())

    print('Dataset\'s rows with outliers: {shape}'.format(shape=x.shape[0]))
    x, y = detect_and_remove_outliers(x, y, clf='LOF')  # clf={IF(IsolationForest), LOF(LocalOutlierFactor)}
    print('Dataset\'s rows after removing outliers: {shape}'.format(shape=x.shape[0]))

    x = normalize_data(x, scaler='MinMax')  # methods={Standard, MinMax, MaxAbs, Robust, Quantile}

    print('Dataset\'s columns before feature selection: {shape}'.format(shape=x.shape[1]))
    x = select_best_features(x, y, n_features=5, method='mutual_info_classif')  # methods={chi2, f_classif, mutual_info_classif}
    print('Dataset\'s columns after feature selection: {shape}'.format(shape=x.shape[1]))
    print(x.head())

    plot(x, y)

    # split to training and testing data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)



if __name__ == '__main__':
    preprocessing()
