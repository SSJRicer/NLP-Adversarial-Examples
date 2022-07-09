# System
import os

# Type-hints
from typing import Union, Literal, Iterable

# Data
import pandas as pd
import numpy as np

# Features
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer

# Models
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential

# IO
PathType                        = Union[str, os.PathLike]

# Data
DataType                        = Union[pd.DataFrame, np.ndarray, Iterable]

# Models
ModelType               = Literal["sklearn", "keras"]
ClassifierType          = Union[LogisticRegression, Sequential]

# Features
TransformerByModelType          = Literal["sklearn_bow", "sklearn_tfidf", "keras_tokenizer"]
SklearnFeatureTransformerType   = Union[SklearnCountVectorizer, SklearnTfidfVectorizer]
FeatureTransformerType          = Union[SklearnCountVectorizer, SklearnTfidfVectorizer, KerasTokenizer]
