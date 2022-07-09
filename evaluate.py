# Logging
import logging

# Type-hints
import utils.custom_types

# Arrays
import numpy as np

# Dataset
import dataset as nlp_dataset

# Features
import features as nlp_features

# Models
import tensorflow.keras.models

# Metrics
import sklearn.metrics

# Utilities
import utils.io_utils

logger = logging.getLogger(__name__)


def evaluation_by_sklearn_model(
    model   : utils.custom_types.ClassifierType,
    data    : utils.custom_types.DataType,
    data_gt : utils.custom_types.DataType,
    *args, **kwargs
    ) -> float:
    """ Evaluate trained Sklearn model on given data. """

    # Calculate model accuracy score
    accuracy = model.score(data, data_gt)
    print(f"Accuracy score on data: {100*accuracy:.2f}%")

    # Get model prediction
    y_pred = model.predict(data)

    print("===== Classification Report =====")
    print(sklearn.metrics.classification_report(data_gt, y_pred))

    # Calculate TRUE vs PRED confusion matrix
    # NOTE: The rows represent the ground truth & columns the prediction, such that:
    #           (0, 0) | (0, 1)
    #   TRUE    ---------------
    #           (1, 0) | (1, 1)
    #                 PRED
    confusion_mat = sklearn.metrics.confusion_matrix(data_gt, y_pred, labels=sorted(np.unique(data_gt)))
    print("===== Confustion matrix =====")
    print(confusion_mat)

    return accuracy
    


def evaluation_by_keras_model(
    model         : utils.custom_types.ClassifierType,
    data          : utils.custom_types.DataType,
    data_gt       : utils.custom_types.DataType,
    batch_size    : int = 256,
    *args, **kwargs
    ) -> float:
    """ Evaluate trained Keras (LSTM) model on given data. """

    results = model.evaluate(data, data_gt, batch_size=batch_size, *args, **kwargs)
    loss, accuracy = results

    print(f"Accuracy score on data: {100*accuracy:.2f}%")

    return accuracy


def evaluation_by_model_type(
    model_type    : utils.custom_types.ModelType,
    model         : utils.custom_types.ClassifierType,
    data          : utils.custom_types.DataType,
    data_gt       : utils.custom_types.DataType,
    *args, **kwargs
    ) -> float:
    """
    Run evaluation on a test dataset using a model based on model type.
    'sklearn' type has the following kwargs: ().
    'keras' type has the following kwargs: ("batch_size").
    """

    if model_type.lower() == "sklearn":
        accuracy = evaluation_by_sklearn_model(model=model, data=data, data_gt=data_gt, *args, **kwargs)

    elif model_type.lower() == "keras":
        accuracy = evaluation_by_keras_model(model=model, data=data, data_gt=data_gt, *args, **kwargs)

    else:
        raise TypeError(f"Bad model type given: '{model_type}'. Supported types: {utils.custom_types.ModelType.__args__}")

    return accuracy


def evaluate(args):
    """ 
    Main evaluation pipeline.
    NOTE: Currently supporting only the following huggingface datasets for sentiment analysis:
        "imdb"
    """

    # Get arguments
    model_type                  = args.model_type.lower()
    config                      = args.config
    feature_transformer_path    = args.feature_transformer_path
    model_path                  = args.model_path

    # CONFIG
    config = utils.io_utils.load_config(config)
    try: 
        config = config["classifier"][model_type]
    except KeyError as e:
        raise KeyError(f"Configuration file does not contain configuration for model type: '{model_type}'.")

    # Load data
    df = nlp_dataset.create_dataframe(dataset_name="imdb", dataset_split="test")

    # Pre-process data
    preprocess_parameters = {}
    if model_type == "sklearn":
        preprocess_parameters["do_remove_stopwords"] = config.get("preprocess", {}).get("do_remove_stopwords", False)

    df = nlp_dataset.preprocess_dataframe(df=df, model_type=model_type, **preprocess_parameters)

    # Features
    logger.info(f"Loading feature transformer from file: '{feature_transformer_path}'...")
    feature_transformer = utils.io_utils.load_data_by_lib(feature_transformer_path, lib="pickle")

    # Model
    logger.info(f"Loading model from file: '{model_path}'...")
    if model_type == "sklearn":
        data    = df["preprocessed_text"]
        model   = utils.io_utils.load_data_by_lib(model_path, lib="pickle")

    elif model_type == "keras":        
        data    = df["review"]
        model   = tensorflow.keras.models.load_model(model_path)

    # Transform data
    transform_parameters = {}
    if model_type == "keras":
        transform_parameters["maxlen"] = config.get("transform", {}).get("maxlen", 250)

    data_features = nlp_features.transform_features(
        feature_transformer   = feature_transformer,
        data                  = data,
        **transform_parameters
    )

    # Evaluate on data
    accuracy = evaluation_by_model_type(model_type=model_type, model=model, data=data_features, data_gt=df["label"])
