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
    feature_importance_algorithm = "shap"
    learning_type = os.environ['LEARNING_TYPE']
    domain_type = os.environ['DOMAIN_TYPE']
    model_interpretor_algorithm = os.environ['MODEL_INTERPRETOR_ALG']
    model_explainer_algorithm = os.environ['MODEL_EXPLAINER_ALG']
    target_names_list =['good', 'bad']
    model_interpretor_strategy = os.environ['MODEL_INTERPRETOR_STRATEGY']
    model_interpret_k_value = int(os.environ['MODEL_INTERPRETOR_K_VALUE']) 
    pprint("Learning_type = " + learning_type)
    pprint("Domain_type = " + domain_type)

    model_interpret_top_value = 8
    num_of_class = 2
    json_config = 'basic-report-explainable.json'
    controller = Controller(config=Configuration(json_config, locals()))
    pprint(controller.config)
    controller.render()
    pprint("Completed explainer step")
