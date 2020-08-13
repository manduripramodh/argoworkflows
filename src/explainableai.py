import connexion
from flask import jsonify
from connexion.exceptions import ProblemException
from error_codes import OwnErrorCode
from http import HTTPStatus
from exceptions import TrackingException
from request_utils import get_request_id, get_target

import shutil
import pickle

import warnings

from pprint import pprint
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
import os
import sys
from pprint import pprint

np.random.seed(123456)
import os
import json
import sys
import datetime
import xai
from xai.compiler.base import Configuration, Controller
from xai.explainer.explainer_factory import ExplainerFactory
import pandas as pd

from constants import (
    CONFIG_SECTIONS, CONFIGURATION, DATA_TYPES, DL_PREFIX,
    DL_XAI_PDF_PATH, EXPLAINER_PDF_NAME, FEATURE_COMPONENT,
    FEATURE_IMPORTANCE_DESCRIPTION, FEATURE_IMPORTANCE_TITLE, LABELS_EXCEPTION,
    LEARNING_TYPES, LIME_STRATEGY, METADATA, METADATA_FILE, METADATA_PATH, MODEL_COMPONENT,
    MODEL_INTERPRETOR_DESCRIPTION, MODEL_INTERPRETOR_TITLE, NO_MODEL_EXCEPTION,
    PIPELINE_EXCEPTION, TARGET_NAMES_EXCEPTION, TEXT_REGRESSION_NOT_SUPPORTED, MODEL_INTERPRETOR_INVALID_K_VALUE,
    TRAINING_DATA_COLUMN_EXCEPTION, TRAINING_DATA_EXCEPTION, PDF_WRITER, TRAINING_DATA_EMPTY_EXCEPTION,
    VALID_PREDICT_EXCEPTION, ConfigurationValues, NO_CLASS_EXCEPTION, TRAINING_DATA_LIST_EXCEPTION,
    MODEL_INTERPRETOR_INVALID_NO_OF_CLASS, MODEL_INTERPRETOR_INVALID_TOP_VALUE, FEATURE_IMPORTANCE_INVALID_THRESHOLD,
    FeatureImportanceValues, ModelInterpretorValues, Section)


def check_health():
    return 'OK', 200


def create(run):
    pprint("Entered create explanation step")
    explainableAI.generate_explanations(run)
    return 'Explainer Report Generated', 200


def inference(run):
    pprint("Entered Get inference result step")
    inference_response = explainableAI.generate_inference_result(run)
    pprint(inference_response)
    return {"response": inference_response}, HTTPStatus.OK

class ExplainableAI:
    def __init__(self):
        self.app = "NULL"

    def get_model(self):
        # TODO : Get the model data from data lake
        return self.model

    def get_Xtrain(self):
        # TODO : Get the Train data from data lake
        return self.X_train

    def get_predict_fn(self):
        # TODO : Get the Train data from data lake
        return self.predictFn

    # TODO : Training the model
    def train_data(self, isGenerateExplainerEnabled):
        currentDT = datetime.datetime.now()
        metrics_dict = {"kpi": "1"}
        pprint("Entered Explanation generation step")
        # Importing the data
        df = pd.read_csv("german_credit_data.csv", index_col=0)

        df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
        df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()
        df_age = df['Age'].values.tolist()

        # Let's look the Credit Amount column
        interval = (18, 25, 35, 60, 120)

        cats = ['Student', 'Young', 'Adult', 'Senior']
        df["Age Bucket"] = pd.cut(df.Age, interval, labels=cats)

        df_good = df[df["Risk"] == 'good']
        df_bad = df[df["Risk"] == 'bad']

        # Purpose
        df['Purpose'].replace(
            ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs',
             'vacation/others'], [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        # Sex
        df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

        # Housing
        df['Housing'].replace(['own', 'free', 'rent'], [0, 1, 2], inplace=True)

        # Saving accounts
        df['Saving accounts'] = df['Saving accounts'].fillna('empty')
        df['Saving accounts'].replace(['empty', 'little', 'moderate', 'quite rich', 'rich', ], [0, 1, 2, 3, 4],
                                      inplace=True)

        # Checking accounts
        df['Checking account'] = df['Checking account'].fillna('empty')
        df['Checking account'].replace(['empty', 'little', 'moderate', 'rich', ], [0, 1, 2, 3], inplace=True)

        # Age Bucket
        df['Age Bucket'].replace(['Student', 'Young', 'Adult', 'Senior'], [0, 1, 2, 3], inplace=True)

        # Risk
        df['Risk'].replace(['good', 'bad'], [0, 1], inplace=True)

        del df["Age"]

        df['Credit amount'] = np.log(df['Credit amount'])

        # Creating the X and y variables
        X = df.drop('Risk', 1)
        y = df["Risk"]

        # Spliting X and y into train and test version
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        clf = RandomForestClassifier(max_depth=None, max_features=8, n_estimators=3, random_state=2)
        clf.fit(X_train, y_train)

        trainingData = {
            "X_train": X_train,
            "Y_train": y_train
        }
        feature_names = X_train.columns.tolist()
        target_names_list = ['good', 'bad']
        clf_fn = clf.predict_proba
        self.model = clf
        self.predictFn = clf_fn
        self.X_train = X_train
        self.y_train = y_train
        if isGenerateExplainerEnabled == True:
            json_config = 'basic-report-explainable.json'
            controller = Controller(config=Configuration(json_config, locals()))
            pprint(controller.config)
            controller.render()

    def generate_inference_result(self, configParams):
        """
        Generate explanations by using compiler
        :param config params: List of config parameters
         """
        try:
            # TO DO -- Remove this part when explainer artifact is generated via Data Lake
            self.train_data(True)
            datasetType = configParams.get('datasetType').lower()
            explainer = ExplainerFactory.get_explainer(domain=datasetType, algorithm=xai.ALG.LIME)
            # TO DO - Fetch the Explainer Artifact from Data Lake
            explainer.load_explainer('explainer.pkl')
            config_params = configParams.get('inferenceData')
            num_features = configParams.get('numberofFeatures')

            config_dict = []
            config_dict.append(config_params)
            inferenceDf = pd.DataFrame.from_dict(config_dict)
            explanations = explainer.explain_instance(instance=inferenceDf.values[0, :], num_features=num_features)

            return explanations

        except IOError as error:  # pylint: disable=broad-except
            error_msg = {'error': f' Failed Generate the explanation'}
            pprint(f'{error_msg}: {error}')
            raise
        except Exception as error:  # pylint: disable=broad-except
            error_msg = {'error': f'Failed Generate the explanation report'}
            pprint(f'{error_msg}: {error}')
            raise

    def generate_explanations(self, configParams):
        """
        Generate explanations by using compiler
        :param config params: List of config parameters
         """
        try:
            # Convert from bytes to respective dictionaries and method
            pprint('Entered Generate Explanations')
            # Remove once we have are able to get data from data lake
            self.train_data(False)

            self.read_configuration_values(configParams)
            train_labels = self._config.labels
            X_train = self.get_Xtrain()
            predict_fn = self.get_predict_fn()
            model_data = self.get_model()
            y_train = self.y_train

            # Assumption feature names is part of Training data for Tabular
            if self._config.datasetType == DATA_TYPES['Tabular']:
                feature_names = X_train.columns.tolist()
            target_names_list = self._config.target_names
            # Create the Configuration object needed by Compiler
            config = self.create_configuration()

            controller = Controller(config=Configuration(config, locals()))
            controller.render()

        except IOError as error:  # pylint: disable=broad-except
            error_msg = {'error': f' Failed Generate the explanation'}
            pprint(f'{error_msg}: {error}')
            raise
        except Exception as error:  # pylint: disable=broad-except
            error_msg = {'error': f'Failed Generate the explanation report'}
            pprint(f'{error_msg}: {error}')
            raise

    def create_configuration(self):
        """
        Function Create Compiler configuration
        :param api: API instance
        :return dict: XAI Compiler Config
        """
        sections = []

        if self._config.feature_importance_enabled:
            feature_section = self.create_feature_importance()
            sections.append(feature_section)

        model_section = self.create_model_config()
        sections.append(model_section)
        CONFIG_SECTIONS['sections'] = sections

        config_sections = [CONFIG_SECTIONS]
        CONFIGURATION['contents'] = config_sections

        attribute = PDF_WRITER['attr']

        attribute['path'] = os.getcwd()
        CONFIGURATION['writers'].append(PDF_WRITER)
        pprint("Entered Create Model Interpretor Configuration End")
        return CONFIGURATION

    def create_feature_importance(self):
        """
        Function Create Feature Importance
        :return dict: Feature Importance Values
        """
        pprint("Entered Create Feature Importance Configuration Step")
        attributes = FeatureImportanceValues('var:model_data', 'var:X_train', 'var:feature_names',
                                             self._config.featureImportance.lower(), self._config.learningType.lower())

        feature_component_attributes = attributes._asdict()

        del feature_component_attributes['feature_names']
        FEATURE_COMPONENT['attr'] = feature_component_attributes
        section = Section(FEATURE_IMPORTANCE_TITLE, FEATURE_IMPORTANCE_DESCRIPTION,
                          FEATURE_COMPONENT)
        pprint("Entered Create Feature Importance Configuration End")
        return section._asdict()

    def create_model_config(self):
        """
        Function Create Model Interpretor JSON
        :return dict: Model Interpretor
        """

        attributes = ModelInterpretorValues(self._config.datasetType.lower(), "lime", self._config.learningType.lower(),
                                            "var:X_train",
                                            "var:y_train", "var:predict_fn", "var:feature_names",
                                            "var:target_names_list",
                                            LIME_STRATEGY[self._config.limeModelInterpreter],
                                            self._config.model_interpret_k_value,
                                            self._config.model_interpret_top_value, self._config.num_of_class)

        model_attributes = attributes._asdict()

        if self._config.learningType == LEARNING_TYPES['Regression']:
            del model_attributes['num_of_class']
            del model_attributes['labels']

        if self._config.datasetType == DATA_TYPES['Text']:
            del model_attributes['feature_names']

        MODEL_COMPONENT['attr'] = model_attributes
        section = Section(MODEL_INTERPRETOR_TITLE, MODEL_INTERPRETOR_DESCRIPTION,
                          MODEL_COMPONENT)

        return section._asdict()

    def _check_valid_datatype(self, data_value, data_type, error_message):
        """
        Check for running operator in pipeline only
        :param api: API instance
        :param data_value: data value
        :param api: Expected type of datatype
         """
        if not isinstance(data_value, data_type):
            raise ValueError(error_message)

    def read_configuration_values(self, params):
        """
        Read configuration values with checks
        :param api: API instance
         """
        try:
            config_params = params.get('additionalParamters')  # api.config.configParameters

            learning_type = params.get("learningType")
            dataset = params.get("datasetType")
            if (learning_type == LEARNING_TYPES['Regression']) and (dataset == DATA_TYPES['Text']):
                raise ValueError(TEXT_REGRESSION_NOT_SUPPORTED)

            # Setting default values for labels
            labels = []
            num_of_class = config_params.get("numberClass", 0)
            if learning_type == LEARNING_TYPES['Classification']:
                self._check_valid_datatype(num_of_class, int, MODEL_INTERPRETOR_INVALID_NO_OF_CLASS)
                if num_of_class <= 0:
                    raise ValueError(NO_CLASS_EXCEPTION)

            pprint("Entered Read Configuration Step")
            model_k_value = config_params.get("modelInterpretKValue", 5)
            model_top_value = config_params.get("modelInterpretTopValue", 15)
            threshold = config_params.get("threshold", 0.005)
            sampling_fraction = config_params.get("samplingFraction", 1.0)
            sampling_limit = config_params.get("samplingLimit", 1000)
            strategyType = params.get("learningType")
            strategyType = "Top K"
            self._config = ConfigurationValues(dataset,
                                               learning_type,
                                               "shap",
                                               params.get("learningType"),
                                               strategyType,
                                               config_params['targetNames'],
                                               labels,
                                               model_k_value,
                                               model_top_value,
                                               threshold,
                                               num_of_class,
                                               sampling_fraction,
                                               sampling_limit)
            pprint("Completed Read Configuration Step")

        except Exception as error:  # pylint: disable=broad-except
            error_msg = {'error': f' Failed to Read the configuration Values.'}
            pprint(f'{error_msg}: {error}')
            raise


explainableAI = ExplainableAI()

# if __name__ == '__main__':
# generate_explanation()
