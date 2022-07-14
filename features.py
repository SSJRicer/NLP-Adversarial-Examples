# Logging
import logging

# Type-hints
from typing import Sequence, Union
import utils.custom_types

# Arrays
import numpy as np

# Datasets
import pandas as pd

# Features
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

# Utilities
import utils.io_utils

logger = logging.getLogger(__name__)


def create_and_fit_feature_transform(
    transformer_by_model_type   : utils.custom_types.TransformerByModelType,
    data                        : Union[pd. DataFrame, Sequence[str]],
    save_path                   : utils.custom_types.PathType = None,
    *args, **kwargs
    ) -> utils.custom_types.FeatureTransformerType:
    """
    Instantiates feature transformer for model by type.
    'sklearn' transformers have the following kwargs: ("max_features", "ngram_range", "stop_words").
    'keras' transformers have the following kwargs: ("num_words").
    NOTE: Check each transformer's documentation for more arguments.
    """

    logger.debug(f"Instantiating & fitting feature transformer for '{transformer_by_model_type.lower()}' model...")

    if transformer_by_model_type.lower() == "sklearn_bow":
        feature_transformer = SklearnCountVectorizer(*args, **kwargs)
        feature_transformer.fit(data)
    
    elif transformer_by_model_type.lower() == "sklearn_tfidf":
        feature_transformer = SklearnTfidfVectorizer(*args, **kwargs)
        feature_transformer.fit(data)

    elif transformer_by_model_type.lower() == "keras_tokenizer":
        feature_transformer = KerasTokenizer(*args, **kwargs)
        feature_transformer.fit_on_texts(data)

    else:
        raise ValueError(f"Bad transformer by model type given: '{transformer_by_model_type}'. Supported types: {utils.custom_types.TransformerByModelType.__args__}")

    if save_path is not None:
        utils.io_utils.save_data_by_lib(feature_transformer, save_path, lib="pickle")

    return feature_transformer


def transform_features(
    feature_transformer   : utils.custom_types.FeatureTransformerType,
    data                  : Union[pd. DataFrame, pd.Series, Sequence[str]],
    save_path             : utils.custom_types.PathType = None,
    *args, **kwargs
    ) -> utils.custom_types.DataType:
    """ 
    Transform text into features.
    "keras" transformer takes a "maxlen" argument.
    """

    logger.debug(f"Transforming features...")

    if isinstance(feature_transformer, KerasTokenizer):
        features = feature_transformer.texts_to_sequences(data)
        features = keras_pad_sequences(features, **kwargs)
        features = np.array(features)

    elif isinstance(feature_transformer, (SklearnCountVectorizer, SklearnTfidfVectorizer)):
        features = feature_transformer.transform(data)
        features = pd.DataFrame(features.toarray(), columns=feature_transformer.get_feature_names_out())

    else:
        raise TypeError(f"Bad feature transformer type given: '{feature_transformer.__class__}' . Supported types: {utils.custom_types.FeatureTransformerType.__args__}")

    if save_path is not None:
        if isinstance(features, pd.DataFrame):
            features.to_pickle(save_path)
        else:
            utils.io_utils.save_data_by_lib(features, save_path, lib="pickle")

    return features
