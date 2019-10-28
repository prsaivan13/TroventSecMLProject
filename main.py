import json

import matplotlib.pyplot as plt


from classifiers import decision_tree_classifier, lasso, sgd_classifier, linear_svc_classifier, svm_classifier, k_nearest_neighbors

if __name__ == "__main__":
    with open('data/sampledata.json') as data:
        json_data = [json.loads(line) for line in data]

    # decision_tree_classifier(json_data=json_data) # radi
    # k_nearest_neighbors(json_data=json_data) # radi
    # svm_classifier(json_data=json_data) # radi
    # lasso(json_data=json_data) # ne radi
    # linear_svc_classifier(json_data=json_data) # radi
    # sgd_classifier(json_data=json_data) # radi

    plt.show()
