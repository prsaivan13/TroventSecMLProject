import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


from helpers import plot_confusion_matrix


def linear_svc_classifier(json_data):

    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    lsvcc = LinearSVC(C=0.01, tol=1e-5, max_iter=2500)

    lsvcc.fit(x_train, y_train)

    y_pred = lsvcc.predict(x_test)

    print('Linear SVC classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='Linear SVM classifier')


def sgd_classifier(json_data):

    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    sgdc = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=2500,
           n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
           random_state=None, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)

    sgdc.fit(x_train, y_train)

    y_pred = sgdc.predict(x_test)

    print('SGD Classifier')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='SGD Classifier')


def svm_classifier(json_data):
    sv_class = SVC(gamma='scale', decision_function_shape='ovo')

    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    sv_class.fit(x_train, y_train)

    y_pred = sv_class.predict(x_test)

    print('SVM classifier:')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=y.unique(), title='SVM classifier')


def lasso(json_data):
    reg = linear_model.Lasso(alpha=0.1)
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)

    print('Lasso classifier:')
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
