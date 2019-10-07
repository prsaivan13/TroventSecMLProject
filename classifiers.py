import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def svm_classifier(json_data):
    sv_class = SVC(kernel='linear') # linear/poly/rbf/sigmoid

    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    sv_class.fit(x_train, y_train)

    y_pred = sv_class.predict(x_test)

    print('SVM classifier:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def decision_tree_classifier(json_data):
    df = pd.DataFrame(data=json_data)

    x = df.drop('type', axis=1)
    y = df['type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print('Decision tree classifier:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


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
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))