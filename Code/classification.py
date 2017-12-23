import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from sklearn.preprocessing import label_binarize

from Code.preparation import read_data, impute_missing_values, preprocessing, preprocessing_unknown
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, \
    classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier


def classification(pp='Y', clf='Tree', random=0, impute_values=True, remove_outliers=True, scale=True,
                   best_features=True):
    if pp == 'N':
        x, y = read_data(csv_file='companydata.csv')
        x = impute_missing_values(x)
    else:
         x,y = preprocessing(impute_values, remove_outliers, scale, best_features)
    print(np.sum(y == 0), np.sum(y== 1), end='\n')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    classifiers = {
        'Tree': DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, random_state=random,
                                       min_samples_split=40, presort=True),
        'KN': KNeighborsClassifier(n_neighbors=5),
        'RN': RadiusNeighborsClassifier(radius=1.0),
        'GP': GaussianProcessClassifier(),
        'GB': GradientBoostingClassifier(loss='exponential', n_estimators=1000, max_depth=3, random_state=0),
        'GNB': GaussianNB(),
        'MNB': MultinomialNB(),
        'BNB': BernoulliNB(),
        'RF': RandomForestClassifier(n_estimators=10),
        'ET': ExtraTreesClassifier(n_estimators=10),
        'NC': NearestCentroid(),
        'SVC': SVC(class_weight='balanced'),
        'NuSVC': NuSVC(),
        'LSVC': LinearSVC(),
        'SGDC': SGDClassifier(random_state=0),
        'DTR': DecisionTreeRegressor(random_state=0, presort=True),
        'ADA': AdaBoostClassifier(n_estimators=500, random_state=0),
        'BC': BaggingClassifier(n_estimators=50, random_state=0),
        'MLP': MLPClassifier(activation='logistic', learning_rate='adaptive'),
        'EXGB': XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=1000, objective="binary:logistic",
                              min_child_weight=1, gamma=0, max_delta_step=0,
                              scale_pos_weight=float(np.sum(y_train == 0)) / np.sum(y_train == 1), seed=0)
    }

    print(np.sum(y_train == 0), np.sum(y_train == 1), end='\n')
    est = classifiers[clf]
    if clf == 'EXGB':
        est.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='auc', verbose=False)
    else:
        est.fit(x_train, y_train)
    predictions = est.predict(x_test)
    scores(y_test, predictions, pp, clf)
    cross_val_scores(est, x, y, 10)
    roc_curve_plot(y_test, predictions)


def scores(y_test, predictions, pp, clf):
    print()
    if pp == 'Y':
        print('Scores After Preprocessing :')
    else:
        print('Scores Before Preprocessing :')
    print('Classifier = {clf}'.format(clf=clf))
    print('Accuracy score = {accuracy}'.format(accuracy=accuracy_score(y_test, predictions)))
    print('Precision score = {precision}'.format(precision=precision_score(y_test, predictions)))
    print('Recall score = {recall}'.format(recall=recall_score(y_test, predictions)))
    print('F1 Score = {f1score}'.format(f1score=f1_score(y_test, predictions)))
    print('ROC AUC = {roc_auc}'.format(roc_auc=roc_auc_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print()


def cross_val_scores(estimator, x, y, k_fold):
    cv = cross_validate(estimator, x, y, cv=k_fold, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    print('{k_fold}-fold Cross validation scores:'.format(k_fold=k_fold))
    print('Accuracy score = {accuracy}(+/-{std})'.format(accuracy=cv['test_accuracy'].mean(),
                                                         std=cv['test_accuracy'].std() * 2))
    print('Precision score = {precision}(+/-{std})'.format(precision=cv['test_precision'].mean(),
                                                           std=cv['test_precision'].std() * 2))
    print('Recall score = {recall}(+/-{std})'.format(recall=cv['test_recall'].mean(), std=cv['test_recall'].std() * 2))
    print('F1 Score = {f1score}(+/-{std})'.format(f1score=cv['test_f1'].mean(), std=cv['test_f1'].std() * 2))
    print('ROC AUC Score = {roc_auc_score}(+/-{std})'.format(roc_auc_score=cv['test_roc_auc'].mean(),
                                                             std=cv['test_roc_auc'].std() * 2))
    print()


def roc_curve_plot(y_test, y_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = label_binarize(y_test, classes=[0, 1])
    n_classes = y_test.shape[1]
    y_score = label_binarize(y_score, classes=[0, 1])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def exponentialLoss(y, pred):
    df = -y * np.exp(-y * pred)
    hess = np.exp(-y * pred)
    return df, hess

def predict_unknown():
    x_test = preprocessing_unknown()
    x, y = preprocessing(True, True, True, False)
    clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=1000, objective="binary:logistic",
                              min_child_weight=2, gamma=0, max_delta_step=0,
                              scale_pos_weight=float(np.sum(y == 0)) / np.sum(y == 1), seed=0)
    clf.fit(x, y, eval_set=[(x, y)], eval_metric='auc', verbose=False)
    predictions = clf.predict(x_test)
    pd.Series(predictions).to_csv('prediction.csv')
    prd = pd.read_csv('prediction.csv')
    # known = pd.read_csv('companydata.csv')
    # y = known['X65']
    # y.append(prd)
    # print(y)
    print(prd)


if __name__ == '__main__':
    #classification(pp='N', clf='EXGB')
    #classification(clf='EXGB', scale=True, remove_outliers=True, best_features=False)
    predict_unknown()