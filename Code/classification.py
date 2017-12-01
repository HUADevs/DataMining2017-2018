import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, confusion_matrix

from preparation import read_data, impute_missing_values, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier


def early_classification():
    x, y = read_data(csv_file='../companydata.csv')
    x = impute_missing_values(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    print('Accuracy score = {accuracy}'.format(accuracy=accuracy_score(y_test, predictions)))
    print('Precision score = {precision}'.format(precision=precision_score(y_test, predictions)))
    print('Recall score = {recall}'.format(recall=recall_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))


def preprocess_classification():
    x, y = preprocessing()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    print('Accuracy score = {accuracy}'.format(accuracy=accuracy_score(y_test, predictions)))
    print('Precision score = {precision}'.format(precision=precision_score(y_test, predictions)))
    print('Recall score = {recall}'.format(recall=recall_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))


if __name__ == '__main__':
    print('Classification before preprocessing...')
    early_classification()
    print('Classification after preprocessing...')
    preprocess_classification()
