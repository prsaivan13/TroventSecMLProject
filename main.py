import json

import matplotlib.pyplot as plt


from classifiers import decision_tree_classifier, svm_classifier, k_nearest_neighbors

if __name__ == "__main__":
    with open('data/sampledata.json') as data:
        json_data = [json.loads(line) for line in data]

    decision_tree_classifier(json_data=json_data)
    k_nearest_neighbors(json_data=json_data)
    # svm_classifier(json_data=json_data)

    plt.show()
