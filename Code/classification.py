from preparation import read_data, impute_missing_values, preprocessing
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
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
    classifiers = {
        'Tree': DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, random_state=random,
                                       min_samples_split=40, max_features=None, class_weight=None,
                                       presort=True),
        'KN': KNeighborsClassifier(n_neighbors=5),
        'RN': RadiusNeighborsClassifier(radius=1.0),
        'GP': GaussianProcessClassifier(),
        'GB': GradientBoostingClassifier(loss='exponential', learning_rate=1.0, n_estimators=100,
                                         criterion='friedman_mse', max_depth=3, random_state=0),
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
        'ADA': AdaBoostClassifier(n_estimators=100, random_state=0),
        'BC': BaggingClassifier(n_estimators=10, random_state=0),
        'MLP': MLPClassifier(activation='logistic', learning_rate='adaptive'),
        # 'EXGB': XGBClassifier()
    }

    if pp == 'N':
        x, y = read_data(csv_file='companydata.csv')
        x = impute_missing_values(x)
    else:
        x, y = preprocessing(impute_values, remove_outliers, scale, best_features)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    est = classifiers[clf]
    est.fit(x_train, y_train)
    predictions = est.predict(x_test)

    scores(y_test, predictions, pp, clf)
    cross_val_scores(est, x, y, 10)


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
    print(confusion_matrix(y_test, predictions))
    print()


def cross_val_scores(estimator, x, y, k_fold):

    cv = cross_validate(estimator, x, y, cv=k_fold, scoring=['accuracy', 'precision', 'recall', 'f1'])
    print('{k_fold}-fold Cross validation scores:'.format(k_fold=k_fold))
    print('Accuracy score = {accuracy}'.format(accuracy=cv['test_accuracy'].mean()))
    print('Precision score = {precision}'.format(precision=cv['test_precision'].mean()))
    print('Recall score = {recall}'.format(recall=cv['test_recall'].mean()))
    print('F1 Score = {f1score}'.format(f1score=cv['test_f1'].mean()))
    print()


if __name__ == '__main__':
    classification(pp='N', clf='EXGB')
    classification(clf='EXGB', scale=True, best_features=False)
