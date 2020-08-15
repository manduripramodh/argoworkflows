import pickle
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier

np.random.seed(123456)

root_path = '/training'

if __name__ == '__main__':
    pprint("Entered training step")

    with open('{}/input/train.pickle'.format(root_path), 'rb') as f:
        X_train, y_train = pickle.load(f)

    clf = RandomForestClassifier(max_depth=None, max_features=8, n_estimators=3, random_state=2)
    clf.fit(X_train, y_train)
    os.makedirs('{}/output/'.format(root_path), exist_ok=True)
    with open('{}/output/func.pickle'.format(root_path), 'wb') as f:
        pickle.dump(clf.predict_proba, f)

    with open('{}/output/model.pickle'.format(root_path), 'wb') as f:
        pickle.dump(clf, f)

    pprint("Finished training step")
