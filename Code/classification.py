from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from preparation import read_data, impute_missing_values, preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier


def classification(pp='Y', clf='Tree', random=0, impute_values=True, remove_outliers=True, scale=True, best_features=True):
    classifiers = {
        'Tree': DecisionTreeClassifier(random_state=0),
        'KN': KNeighborsClassifier(n_neighbors=5),
        'RN': RadiusNeighborsClassifier(radius=1.0),
        'GP': GaussianProcessClassifier(),
        'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'GNB': GaussianNB(),
        'MNB': MultinomialNB(),
        'BNB': BernoulliNB(),
        'RF': RandomForestClassifier(),
        'NC': NearestCentroid(),
        'SVC': SVC(),
        'NuSVC': NuSVC(),
        'LSVC': LinearSVC(),
        'SGDC': SGDClassifier(),
        'DTR': DecisionTreeRegressor(),
        'ADA': AdaBoostClassifier()
    }

    if pp == 'N':
        x, y = read_data(csv_file='../companydata.csv')
        x = impute_missing_values(x)
    else:
        x, y = preprocessing(impute_values, remove_outliers, scale, best_features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random)

    est = classifiers[clf]
    est.fit(x_train, y_train)
    predictions = est.predict(x_test)

    scores(y_test, predictions, pp, clf)


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

if __name__ == '__main__':
    classification(pp='N', clf='RF')
    classification(clf='RF')
