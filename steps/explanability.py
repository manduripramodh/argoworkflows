import pickle
from xai.compiler.base import Configuration, Controller
import os
from pprint import pprint
import numpy as np

np.random.seed(123456)

root_path = '/training'

if __name__ == '__main__':
    pprint("Entered explainer step")
    with open('{}/input/train.pickle'.format(root_path), 'rb') as f:
        X_train, y_train = pickle.load(f)

    with open('{}/input/model.pickle'.format(root_path), 'rb') as f:
        clf = pickle.load(f)

    with open('{}/input/func.pickle'.format(root_path), 'rb') as f:
        clf_fn = pickle.load(f)

    feature_names = X_train.columns.tolist()
    target_names_list = ['good', 'bad']
    json_config = 'basic-report-explainable.json'
    controller = Controller(config=Configuration(json_config, locals()))
    pprint(controller.config)
    controller.render()
    pprint("Compleyed explainer step")
