import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from helpers import plot_confusion_matrix


def gaussian_naive_bayes(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier = GaussianNB()

    classifier.fit(x_train, y_train)
    GaussianNB(priors=None, var_smoothing=1e-09)

    y_pred = classifier.predict(x_test)

    print('Gaussian Naive Bayes Classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='Gaussian Naive Bayes classifier')


def linear_svc_classifier(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, loss='squared_hinge', max_iter=2500,
                           multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('Linear SVC classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='Linear SVM classifier')


def sgd_classifier(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                               early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                               l1_ratio=0.5, learning_rate='optimal', loss='log', max_iter=2500,
                               n_iter_no_change=5, penalty='l2', power_t=0.9,
                               random_state=None, shuffle=False, tol=0.00001,
                               validation_fraction=0.1, verbose=0, warm_start=False)

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('SGD Classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='SGD Classifier')


def svm_classifier(json_data):
    classifier = SVC(gamma='scale', decision_function_shape='ovo')

    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('SVM classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='SVM classifier')


def decision_tree_classifier(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('Decision tree classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='Decision tree classifier')


def k_nearest_neighbors(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('K-nearest neighbors classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='K-nearest neighbors classifier')
