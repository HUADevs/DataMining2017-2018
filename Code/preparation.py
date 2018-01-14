import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler


def read_data(csv_file):
    # import data from csv
    df = pd.read_csv(csv_file, na_values='?')
    # split to independent features (x) and target feature (y)
    x = df.drop('X65', axis=1)
    y = df['X65']

    return x, y


def impute_missing_values(x, strategy='mean'):
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
        'EE': EllipticEnvelope(assume_centered=False)
    }
    est = classifiers[clf]
    if clf == 'LOF':
        outliers = est.fit_predict(x, y)
    else:
        outliers = est.fit(x, y).predict(x)
    # keep only non-outliers (predicted values outliers=-1, non-outliers=1)
    new_x = x[outliers == 1]
    new_y = y[outliers == 1]

    return new_x, new_y


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


def select_best_features(x, y, n_features=10, method='chi2'):
    methods = {
        # Compute chi-squared stats between each non-negative feature and class
        'chi2': SelectKBest(chi2, k=n_features),
        # Compute the ANOVA F-value
        'f_classif': SelectKBest(f_classif, k=n_features),
        # measures the dependency between the variables
        # higher values mean higher dependency
        'mutual_info_classif': SelectKBest(mutual_info_classif, k=n_features)
    }

    x = pd.DataFrame(data=methods[method].fit_transform(x, y), columns=x.columns[methods[method].get_support()],
                     index=x.index.values)

    return x


def get_charts(x, y):
    print(y)
    joined = pd.concat([x, y], axis=1)
    print(joined)
    sns.pairplot(data=joined, hue='X65', diag_kws=dict(bins=5))
    plt.show()


def preprocessing(impute_values=True, remove_outliers=True, scale=True, best_features=True):
    x, y = read_data(csv_file='companydata.csv')
    if impute_values:
        x = impute_missing_values(x, strategy='mean')

    print('Dataset\'s rows with outliers: {shape}'.format(shape=x.shape[0]))
    if remove_outliers:
        x, y = detect_and_remove_outliers(x, y, clf='IF')  # clf={IF(IsolationForest), LOF(LocalOutlierFactor)}
        print('Dataset\'s rows after removing outliers: {shape}'.format(shape=x.shape[0]))

    if scale:
        x = normalize_data(x, scaler='Standard')  # check column X60 for validation
        #print(x)

    print('Dataset\'s columns before feature selection: {shape}'.format(shape=x.shape[1]))
    if best_features:
        x = select_best_features(x, y, method='mutual_info_classif')  # methods={chi2, f_classif, mutual_info_classif}
        print('Dataset\'s columns after feature selection: {shape}'.format(shape=x.shape[1]))
        print('{n} Best features: {features}'.format(n=10, features=x.columns.values))

    # print(x.head())
    # print(type(x), type(y))
    # get_charts(x, y)
    print()
    return x, y


def preprocessing_unknown():
    #x, y = read_data(csv_file='companydata.csv')
    x = pd.read_csv('test_unlabeled.csv', na_values='?')
    x = impute_missing_values(x, strategy='mean')
    #x, _ = detect_and_remove_outliers(x, y)
    x = normalize_data(x, scaler='Standard')
    return x


if __name__ == '__main__':
    preprocessing()
