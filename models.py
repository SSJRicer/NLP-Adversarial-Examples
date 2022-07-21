# Logging
import logging

# Type-hints
from typing import Sequence, Union, Iterable
from utils import custom_types
from utils.custom_types import ModelType, PathType, SklearnFeatureTransformerType, KerasTokenizer
from tensorflow.keras.models import Sequential

# Text
import re

# Datasets
import numpy as np
import dataset as nlp_dataset

# Models
from sklearn.linear_model import LogisticRegression
import tensorflow.keras as keras

# Utilities
import utils.io_utils
from utils.log_utils import TextColors

logger = logging.getLogger(__name__)

TC = TextColors()


def build_sklearn_model(C: float = 30, max_iter: int = 200, *args, **kwargs):
    """ Create a logistic regression model. """

    model = LogisticRegression(C=C, max_iter=max_iter, *args, **kwargs)

    return model


def build_keras_model(
    embedding_input_dim       : int,
    lstm_input_dim            : int = 128,
    lstm_dropout              : float = 0.5,
    lstm_recurrent_dropout    : float = 0.2,
    loss                      : str = "binary_crossentropy",
    optimizer                 : str = "adam",
    metrics                   : Sequence[str] = ["accuracy"]
    ):
    """ Create & compile a LSTM model. """

    # Create model
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(embedding_input_dim, lstm_input_dim))
    model.add(keras.layers.LSTM(lstm_input_dim, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        loss        = loss,
        optimizer   = optimizer,
        metrics     = metrics
    )

    return model


def build_model(model_type: ModelType, *args, **kwargs):
    """
    Create a model based on model type.
    'sklearn' type has the following kwargs: ("C", "max_iter").
    'keras' type has the following kwargs: ("embedding_input_dim",
        "lstm_input_dim", "lstm_dropout", "lstm_recurrent_dropout",
        "loss", "optimizer", "metrics").
    NOTE: Check each model's documentation for more arguments.
    """

    if model_type.lower() == "sklearn":
        model = build_sklearn_model(*args, **kwargs)

    elif model_type.lower() == "keras":
        model = build_keras_model(*args, **kwargs)

    else:
        raise TypeError(f"Bad model type given: '{model_type}'. Supported types: {utils.custom_types.ModelType.__args__}")

    return model


class CustomClassifier:
    """
    Base class for Sklearn & Keras (LSTM) classifiers.
    """

    def get_prob(self, input_batch: Iterable[str]):
        """ Get prediction probabilities for an input batch. """

        raise NotImplementedError()

    def get_pred(self, input_batch: Iterable[str]):
        """ Get prediction labels for an input batch. """

        return self.get_prob(input_batch).argmax(axis=1)
    
    def get_single_pred(self, input_single: str, do_print: bool = False):
        """ Get prediction probability and label (name) for a single input. """

        # Get probabilties ([neg_prob, pos_prob])
        pred_probs        = self.get_prob([input_single])[0]

        # Get top probability label (0 or 1)
        top_pred_label    = pred_probs.argmax()

        # Get top probability
        top_pred_prob     = pred_probs[top_pred_label]
        
        # Get top label name
        top_pred_name     = self.label_names[top_pred_label]

        if do_print:
            pred_color = TC.BGREEN if top_pred_label else TC.BRED

            print(f"REVIEW: \"{input_single}\"")
            print(f"PREDICTION: {pred_color}{top_pred_name}{TC.RESET} ({100*top_pred_prob:.2f}%)")
        
        return top_pred_name, top_pred_prob

    def save(self, output_path: custom_types.PathType):
        """ Save trained classifier. """

        utils.io_utils.save_data_by_lib(self, output_path, lib="pickle")


class LogisticRegressionSentimentClassifier(CustomClassifier):
    """
    Sklearn Logistic Regression classifier for sentiment analysis.
    """

    def __init__(
        self,
        label_names           : Sequence[str],
        feature_transformer   : Union[SklearnFeatureTransformerType, PathType],
        model                 : Union[LogisticRegression, PathType],
        do_remove_stopwords   : bool = False
        ):

        # Pre-processing
        self.do_remove_stopwords = do_remove_stopwords

        # Data
        self.label_names = label_names
        
        # Features
        if isinstance(feature_transformer, PathType.__args__):
            logger.info(f"Loading feature transformer from file: '{feature_transformer}'...")
            feature_transformer = utils.io_utils.load_data_by_lib(feature_transformer, lib="pickle")
        self.feature_transformer = feature_transformer

        # Model
        if isinstance(model, PathType.__args__):
            logger.info(f"Loading model from file: '{model}'...")
            model = utils.io_utils.load_data_by_lib(model, lib="pickle")
        self.model = model
    
    def get_prob(self, input_batch: Iterable[str]):
        """ Get prediction probabilities for an input batch. """

        batch_probs = []

        for input_text in input_batch:
            # Pre-process
            preprocessed_text   = nlp_dataset.clean_text(input_text)
            if self.do_remove_stopwords:
                preprocessed_text = nlp_dataset.remove_stopwords(text=preprocessed_text, stopwords=nlp_dataset.STOPWORDS)

            # Extract/select features
            features            = self.feature_transformer.transform((preprocessed_text,))

            # Run inference
            probs               = self.model.predict_proba(features)

            batch_probs.append(probs[0])

        return np.array(batch_probs)


class LSTMSentimentClassifier(CustomClassifier):
    """
    Tensorflow/Keras LSTM classifier for sentiment analysis.
    """

    def __init__(
        self,
        label_names           : Sequence[str],
        feature_transformer   : Union[KerasTokenizer, PathType],
        max_seq_len           : int,
        model                 : Union[Sequential, PathType],
        ):

        # Data
        self.label_names = label_names

        # Features
        self.feature_transformer = feature_transformer
        self.max_seq_len = max_seq_len

        # Model
        self.model = model

        # Features
        if isinstance(feature_transformer, PathType.__args__):
            logger.info(f"Loading feature transformer from file: '{feature_transformer}'...")
            feature_transformer = utils.io_utils.load_data_by_lib(feature_transformer, lib="pickle")
        self.feature_transformer = feature_transformer

        # Model
        if isinstance(model, PathType.__args__):
            logger.info(f"Loading model from file: '{model}'...")
            model = keras.models.load_model(model)
        self.model = model
    
    def get_prob(self, input_batch: Iterable[str]):
        """ Get prediction probabilities for an input batch. """

        batch_probs = []

        # Pre-process
        # Remove ANSI (color) sequences
        input_batch           = [re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", '', text) for text in input_batch]

        # Transform features
        input_word_freq_seq   = self.feature_transformer.texts_to_sequences(input_batch)
        model_inp             = keras.preprocessing.sequence.pad_sequences(input_word_freq_seq, maxlen=self.max_seq_len)

        # Get (positive label) prediction
        inp_probs = self.model.predict(model_inp)

        for pos_pred_prob in inp_probs:
            # Calculate negative prediction probability
            pos_pred_prob = pos_pred_prob[0]
            neg_pred_prob = 1.0 - pos_pred_prob

            batch_probs.append([neg_pred_prob, pos_pred_prob])

        return np.array(batch_probs)
