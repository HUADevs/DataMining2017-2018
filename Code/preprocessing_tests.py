import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def find_outliers_tukey(x):
    quartile_1 = np.percentile(x, 25)
    quartile_3 = np.percentile(x, 75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers_indices = list(x.index[(x > upper_bound) | (x < lower_bound)])
    outliers_values = list(x[outliers_indices])
    return outliers_indices, outliers_values


def plot_histogram(x):
    plt.hist(x, color='gray', normed=True, bins=30)
    plt.title('Histogram of {var}'.format(var=x.name))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_box(x):
    plt.boxplot(x)
    plt.title('Boxplot of {var}'.format(var=x.name))
    plt.xlabel(x.name)
    plt.show()


def impute_missing_values(X):
    # check NaN values count by column
    print('Missing values count:')
    print(X.isnull().sum().sort_values(ascending=False).head())

    # impute missing values
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(X)
    X = pd.DataFrame(data=imp.transform(X), columns=X.columns)
    print('Missing values count after imputer:')
    print(X.isnull().sum().sort_values(ascending=False).head())
    return X


def read_data(csv_file):
    # import data from csv
    df = pd.read_csv(csv_file, na_values='?')

    # split to independent features (X) and target feature (y)
    X = df.drop('X65', axis=1)
    y = df['X65']

    return X, y


def detect_and_remove_outliers(X, y):
    od = IsolationForest()
    od.fit(X, y)
    print(od.predict(X))
    #  keep only non-outliers (predicted values outliers=-1, non-outliers=1)
    new_X = X[od.predict(X) == 1]

    return new_X




def preprocessing():

    X, y = read_data(csv_file='companydata.csv')

    X = impute_missing_values(X)

    X = detect_and_remove_outliers(X, y)

    print(X.shape)
    plot_histogram(X['X2'])

    # select k best attributes
    # print(X.shape)
    # X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
    # print(X_new.shape)

    # split to training and testing data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)



if __name__ == '__main__':
    preprocessing()
