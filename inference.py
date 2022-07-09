# Logging
import logging

# Type-hints
from typing import Union, Sequence, Tuple

# Data
from dataset import IMDB_LABELS

# Classifiers
import models as nlp_models
from utils import custom_types

# Utilities
import utils.io_utils

logger = logging.getLogger(__name__)


def run_inference(
    classifier    : Union[nlp_models.LogisticRegressionSentimentClassifier, nlp_models.LSTMSentimentClassifier],
    user_input    : custom_types.DataType,
    ) -> Tuple[Sequence[str], Sequence[float]]:
    """ Runs inference on given input(s) using a trained Logistic Regression classifier. """

    # Run inference
    pred_names, pred_probs = [], []
    for user_inp in user_input:
        pred_name, pred_prob = classifier.get_single_pred(user_inp, do_print=True)
        pred_names.append(pred_name)
        pred_probs.append(pred_prob)

    return pred_names, pred_probs


def inference(args):
    """ Main inference pipeline. """

    # Get arguments
    model_type                  = args.model_type.lower()
    config                      = args.config
    feature_transformer_path    = args.feature_transformer_path
    model_path                  = args.model_path
    user_input                  = args.input

    # CONFIG
    config = utils.io_utils.load_config(config)
    try: 
        config = config["classifier"][model_type]
    except KeyError as e:
        raise KeyError(f"Configuration file does not contain configuration for model type: '{model_type}'.")

    if model_type == "sklearn":
        classifier = nlp_models.LogisticRegressionSentimentClassifier(
            label_names           = IMDB_LABELS,
            feature_transformer   = feature_transformer_path,
            model                 = model_path,
            do_remove_stopwords   = config.get("preprocess", {}).get("do_remove_stopwords", False)
        )
    elif model_type == "keras":
        classifier = nlp_models.LSTMSentimentClassifier(
            label_names           = IMDB_LABELS,
            feature_transformer   = feature_transformer_path,
            model                 = model_path,
            max_seq_len           = config.get("transform", {}).get("maxlen", 250)
        )

    # Convert single input to list
    if isinstance(user_input, str):
        user_input = [user_input]

    logger.info(f"Running inference...")
    run_inference(classifier=classifier, user_input=user_input)
