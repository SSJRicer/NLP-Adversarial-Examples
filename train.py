# Logging
import logging

# IO
from pathlib import Path

# Type-hints
from typing import Sequence
import utils.custom_types

# Datasets
import pandas as pd
import dataset as nlp_dataset

# Features
import sklearn.feature_extraction.text
import features as nlp_features

# Models
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV
import models as nlp_models
import tensorflow.keras.callbacks

# Evaluate
import evaluate as nlp_evaluate

# Utilities
import utils.io_utils

# Control GPU memory usage (make it dynamic)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logger = logging.getLogger(__name__)


def train_sklearn_model(
    model       : utils.custom_types.ClassifierType,
    X_train     : utils.custom_types.DataType,
    y_train     : utils.custom_types.DataType,
    X_test      : utils.custom_types.DataType,
    y_test      : utils.custom_types.DataType,
    save_path   : utils.custom_types.PathType = None,
    *args, **kwargs
    ):
    """ Trains a Logistic Regression model. """

    if "gridsearch" in kwargs:
        grid_params = kwargs["gridsearch"]
        logger.info(f"Performing grid search with parameters: {grid_params}...")

        grid = GridSearchCV(
            sklearn.linear_model.LogisticRegression(),
            param_grid    = grid_params,
            cv            = 5,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_

        logger.info(f"Best cross-validation score: {grid.best_score_:.2f}")
        logger.info(f"Best parameters: {grid.best_params_}")
        logger.info(f"Best estimator: {model}")
    else:
        model.fit(X_train, y_train)

    if save_path is not None:
        test_accuracy = model.score(X_test, y_test)
        utils.io_utils.save_data_by_lib(model, save_path.format(test_accuracy=test_accuracy), lib="pickle")


def train_keras_model(
    model         : utils.custom_types.ClassifierType,
    X_train       : utils.custom_types.DataType,
    y_train       : utils.custom_types.DataType,
    X_test        : utils.custom_types.DataType,
    y_test        : utils.custom_types.DataType,
    batch_size    : int = 128,
    epochs        : int = 10,
    callbacks     : Sequence[tensorflow.keras.callbacks.Callback] = None,
    *args, **kwargs
    ):
    """ Trains a LSTM model. """

    history = model.fit(
        X_train, y_train,
        batch_size        = batch_size,
        epochs            = epochs,
        callbacks         = callbacks,
        validation_data   = (X_test, y_test),
        *args, **kwargs
    )


def train_model(model_type: utils.custom_types.ModelType, *args, **kwargs):
    """
    Trains a model based on model type.
    'sklearn' type has the following kwargs: ().
    'keras' type has the following kwargs: ("batch_size", "epochs").
    NOTE: Check each model's documentation for more arguments.
    """

    logger.info(f"Training '{model_type}' model...")

    if model_type == "sklearn":
        train_sklearn_model(*args, **kwargs)

    elif model_type == "keras":
        train_keras_model(*args, **kwargs)

    else:
        raise TypeError(f"Bad model type given: '{model_type}'. Supported types: {utils.custom_types.ModelType.__args__}")


def train(args):
    """ Main training pipeline. """

    # Get model type
    model_type = args.model_type.lower()

    # CONFIG
    config = utils.io_utils.load_config(args.config)
    try: 
        config = config["classifier"][model_type]
    except KeyError as e:
        raise KeyError(f"Configuration file does not contain configuration for model type: '{model_type}'.")

    # DATASET
    dataset_path = config["dataset"].get("path", None)

    # Load from file
    if dataset_path is not None and Path(dataset_path).exists():
        logger.info(f"Loading dataset from file: '{dataset_path}'...")
        df = pd.read_pickle(dataset_path)

    # Create & pre-process
    else:
        dataset_name    = config["dataset"].get("name", None)
        df              = nlp_dataset.create_dataframe(dataset_name=dataset_name, dataset_split=None)

        # Get pre-processing configuration parameters
        preprocess_parameters                 = config.get("preprocess", {})
        preprocess_parameters["save_path"]    = dataset_path

        df = nlp_dataset.preprocess_dataframe(df, model_type = model_type, **preprocess_parameters)

    if model_type == "sklearn":
        data = df["preprocessed_text"]
    elif model_type == "keras":
        data = df["review"]
        # data = df["preprocessed_text"]

    # FEATURES
    # Get features configuration parameters
    feature_parameters                = config.get("features", {})
    transformer_path                  = args.feature_transformer_path
    feature_parameters["save_path"]   = transformer_path

    # Load from file
    if transformer_path is not None and Path(transformer_path).exists():
        logger.info(f"Loading feature transformer from file: '{transformer_path}'...")
        feature_transformer = utils.io_utils.load_data_by_lib(transformer_path, lib="pickle")
    else:
        # Take out transformer type
        transformer_type = feature_parameters.pop("type")

        if model_type == "sklearn":
            # Set stop words
            stop_words = feature_parameters.get("stop_words", "nltk")
            feature_parameters["stop_words"] = None
            if stop_words == "nltk":
                feature_parameters["stop_words"] = [w for w in nlp_dataset.STOPWORDS if w not in ["no", "not", "nor"]]
            elif stop_words == "sklearn":
                feature_parameters["stop_words"] = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
            else:
                logger.warning(f"Unsupported library given: '{stop_words}'. Supporting only: 'nltk' and 'sklearn'. Defaulting to None...")
        
        # Create & fit transformer
        feature_transformer = nlp_features.create_and_fit_feature_transform(
            transformer_by_model_type   = f"{model_type}_{transformer_type}",
            data                        = data,
            **feature_parameters
        )

    # Get feature transform configuration parameters
    transform_parameters    = config.get("transform", {})

    features_path           = transform_parameters.get("save_path", None)

    # Load from file
    if features_path is not None and Path(features_path).exists():
        logger.info(f"Loading features from file: '{features_path}'...")
        if model_type == "sklearn":
            features = pd.read_pickle(features_path)
        else:
            features = utils.io_utils.load_data_by_lib(features_path, lib="pickle")
    else:
        # Transform data into features
        features = nlp_features.transform_features(
            feature_transformer   = feature_transformer,
            data                  = data,
            **transform_parameters
        )

    # Split data
    logger.info("Splitting dataset to 'train' & 'test'...")
    X_train, y_train    = features[:25000], df["label"][:25000].to_numpy()
    X_test, y_test      = features[25000:], df["label"][25000:].to_numpy()

    # MODEL
    # Get model configuration parameters
    model_parameters = config.get("model", {})
    
    # Get model build configuration parameters
    model_build_parameters = model_parameters.get("build", {})
    if model_type == "keras":
        model_build_parameters["embedding_input_dim"] = len(feature_transformer.word_index) + 1

    # Build model
    model = nlp_models.build_model(model_type=model_type, **model_build_parameters)

    # Get model train configuration parameters
    model_train_parameters = model_parameters.get("train", {})

    if model_type == "sklearn":
        model_train_parameters["save_path"] = args.model_path

    elif model_type == "keras":
        callbacks = [
            tensorflow.keras.callbacks.ModelCheckpoint(
                    filepath          = args.model_path,
                    monitor           = "val_loss",
                    save_best_only    = True
            ),
            tensorflow.keras.callbacks.EarlyStopping(
                monitor                 = "val_loss",
                min_delta               = 1e-2,
                patience                = 3,
                restore_best_weights    = True
            )
        ]
        model_train_parameters["callbacks"] = callbacks

    # Train model
    train_model(
        model_type    = model_type,
        model         = model,
        X_train       = X_train,
        y_train       = y_train,
        X_test        = X_test,
        y_test        = y_test,
        **model_train_parameters
    )

    logger.info("SUCCESS! Finished training.")
    # classifier_path = args.classifier_path

    # if model_type == "sklearn":
    #     classifier = nlp_models.LogisticRegressionSentimentClassifier(
    #         label_names           = nlp_dataset.IMDB_LABELS,
    #         feature_transformer   = feature_transformer,
    #         model                 = model,
    #         do_remove_stopwords   = config.get("preprocess", {}).get("do_remove_stopwords", False)
    #     )
    # elif model_type == "keras":
    #     classifier = nlp_models.LSTMSentimentClassifier(
    #         label_names           = nlp_dataset.IMDB_LABELS,
    #         feature_transformer   = feature_transformer,
    #         model                 = model,
    #         max_seq_len           = transform_parameters.get("maxlen", 250)
    #     )

    # classifier.save(classifier_path)

    if args.evaluate:
        evaluation_parameters = {}
        if model_type == "keras":
            evaluation_parameters["batch_size"] = model_train_parameters.get("batch_size", 256)

        logger.info("Running evaluation on TRAIN data...")
        train_accuracy = nlp_evaluate.evaluation_by_model_type(
            model_type    = model_type,
            model         = model,
            data          = X_train,
            data_gt       = y_train
        )

        logger.info("Running evaluation on TEST data...")
        test_accuracy = nlp_evaluate.evaluation_by_model_type(
            model_type    = model_type,
            model         = model,
            data          = X_test,
            data_gt       = y_test
        )
