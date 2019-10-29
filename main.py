import json

import matplotlib.pyplot as plt

from classifiers import gaussian_naive_bayes, decision_tree_classifier, sgd_classifier, linear_svc_classifier, \
    svm_classifier, k_nearest_neighbors

if __name__ == "__main__":
    with open('data/sampledata.json') as data:
        json_data = [json.loads(line) for line in data]

    decision_tree_classifier(json_data=json_data)
    k_nearest_neighbors(json_data=json_data)
    svm_classifier(json_data=json_data)
    linear_svc_classifier(json_data=json_data)
    sgd_classifier(json_data=json_data)
    gaussian_naive_bayes(json_data=json_data)

    plt.show()
