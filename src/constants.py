from collections import namedtuple

EXPLAINER_PATH = "/shared/ml/executions"
EXPLAINER_PDF_NAME = "Explainer.pdf"
TEXT_REGRESSION_NOT_SUPPORTED = 'Text Regression is not Supported'
TARGET_NAMES_EXCEPTION = 'Target Names cannot be empty'
TRAINING_DATA_EMPTY_EXCEPTION = 'Training Data cannot be empty'
LABELS_EXCEPTION = 'Labels cannot be empty'
PIPELINE_EXCEPTION = "Unexpected graph name. Ensure that this pipeline was created via MLSM."
NO_MODEL_EXCEPTION = "There is no associated Model for the Artifact.Do pass a Model artifact"
VALID_PREDICT_EXCEPTION = "Enter valid predict function"
TRAINING_DATA_EXCEPTION = "Training Data should be a Valid Data frame object"
TRAINING_DATA_LIST_EXCEPTION = "Training Data should be a Valid list for text dataset"
TRAINING_DATA_COLUMN_EXCEPTION = "Training data should have Dataframe with columns "
NO_CLASS_EXCEPTION = "Number of Classes cannot be empty"
FEATURE_IMPORTANCE_CLASS = 'FeatureImportanceRanking'
FEATURE_IMPORTANCE_COMMENT = 'Refer the document section'
FEATURE_IMPORTANCE_TITLE = "Feature Importance Analysis"
FEATURE_IMPORTANCE_DESCRIPTION = "This section provides the analysis on feature"
MODEL_INTERPRETOR_COMMENT = "Refer the document section"
MODEL_INTERPRETOR_CLASS = "ModelInterpreter"
MODEL_INTERPRETOR_TITLE = "Model Interpreter Analysis"
MODEL_INTERPRETOR_DESCRIPTION = "Model and train data"
MODEL_INTERPRETOR_INVALID_K_VALUE = "Invalid model K value for Model Interpretor"
MODEL_INTERPRETOR_INVALID_TOP_VALUE = "Invalid model top value for Model Interpretor"
MODEL_INTERPRETOR_INVALID_NO_OF_CLASS = "Invalid number of class for Model Interpretor"
FEATURE_IMPORTANCE_INVALID_THRESHOLD = "Invalid threshold for Feature importance"

ConfigurationValues = namedtuple('ConfigurationValues',
                                 ['datasetType', 'learningType', 'featureImportance', 'feature_importance_enabled',
                                  'limeModelInterpreter', 'target_names', 'labels', 'model_interpret_k_value',
                                  'model_interpret_top_value', 'threshold', 'num_of_class','sampling_fraction',
                                  'sampling_limit'])

FeatureImportanceValues = namedtuple('FeatureImportanceValues',
                                     ['trained_model', 'train_data', 'feature_names', 'method', 'mode'])

ModelInterpretorValues = namedtuple('ModelInterpretorValues',
                                    ['domain', 'method', 'mode', 'train_data',
                                     'labels', 'predict_func', 'feature_names', 'target_names',
                                     'model_interpret_stats_type', 'model_interpret_k_value',
                                     'model_interpret_top_value', 'num_of_class'])

Section = namedtuple('Section',
                     ['title', 'desc', 'component'])

LEARNING_TYPES = {
    'Classification': 'Classification',
    'Regression': 'Regression'
}

PDF_WRITER = {"class": "Pdf",
              "attr": {"name": "Explainer"
                       }}
WRITER = []

DATA_TYPES = {
    'Tabular': 'Tabular',
    'Text': 'Text'
}

LIME_STRATEGY = {
    'Top K': 'top_k',
    'Average Score': 'average_score',
    'Average Ranking': 'average_ranking'
}

CONFIGURATION = {'content_table': True, 'name': 'Report', 'overview': True,
                 'writers': WRITER}

CONFIG_SECTIONS = {"title": "Model Interpreter",
                   "desc": "This section provides the Interpretation of model"
                   }

FEATURE_COMPONENT = {
    '_comment': FEATURE_IMPORTANCE_COMMENT,
    'class': FEATURE_IMPORTANCE_CLASS,
}

MODEL_COMPONENT = {
    '_comment': MODEL_INTERPRETOR_COMMENT,
    'class': MODEL_INTERPRETOR_CLASS,
}

METADATA = {
    'executionId': '',
    'timestamp': '',
    'artifacts': {
        'pdf': '/pdf/Explainer.pdf'
    }
}

METADATA_FILE = 'metadata.json'

METADATA_PATH = '/{}/xai/'

DL_XAI_PDF_PATH = '/{}/xai/{}/pdf/'

DL_PREFIX = 'dh-dl://DI_DATA_LAKE'