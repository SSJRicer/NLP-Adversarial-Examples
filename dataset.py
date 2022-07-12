# Logging
import logging

# REGEX
import re

# Type-hints
from typing import Sequence, Literal, Union
import utils.custom_types
from nltk.tokenize.api import TokenizerI
from nltk.stem.api import StemmerI

# Datasets
import datasets as huggingface_datasets
import pandas as pd

# Pre-processing
import nltk
import nltk.corpus
from nltk import WordPunctTokenizer
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

IMDB_LABELS   = ["NEGATIVE", "POSITIVE"]
STOPWORDS     = nltk.corpus.stopwords.words("english")


# ========================================================================================= #
#                             SKLEARN PRE-PREPROCESSING FUNCS                               #
# ========================================================================================= #

def clean_text(text: str, do_lower: bool = True) -> str:
    """ Cleans text for pre-processing. """

    # Lowercase text
    if do_lower:
        text = text.lower()

    # Remove ANSI (color) sequences
    text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", '', text)

    # Remove HTML components (namely, '<br />')
    # text = re.sub("<br />", ' ', text)
    text = re.sub(r"<.*?>", ' ', text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", ' ', text)

    # Remove Emojis
    text = re.sub(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", ' ', text, flags=re.UNICODE
    )

    # Remove non-alphanumeric characters
    text = re.sub("[^a-zA-Z0-9]", ' ', text)

    # Remove whitespaces
    text = " ".join(text.split())

    return text


def remove_stopwords(
    text        : str,
    stopwords   : Sequence[str],
    tokenizer   : TokenizerI = None,
    white_list  : Sequence[str] = []
    ) -> str:
    """
    Removes stop words from text (excluding white-listed words).

    Args:
        text -          Text to remove stop words from.
        stopwords -     Sequence of stop words.
        tokenizer -     Tokenizer for splitting the text.
        white_list -    Words to include.
    
    Returns:
        filtered_text - Filtered text.
    """

    # Set default tokenizer
    if tokenizer is None:
        tokenizer = WordPunctTokenizer()

    # Split text into tokens
    tokens            = [token.strip() for token in tokenizer.tokenize(text)]

    # Filter out stop words (excluding white-listed words)
    filtered_tokens   = [token for token in tokens if token not in stopwords or token in white_list]

    # Re-construct text
    filtered_text     = ' '.join(filtered_tokens)

    return filtered_text


def tokenize_and_stem_text(
    df          : pd.DataFrame,
    tokenizer   : TokenizerI = None,
    stemmer     : StemmerI = None,
    do_stem     : bool = False
    ) -> pd.DataFrame:
    """ Tokenize & stem pre-processed text. """

    # Set default tokenizer
    if tokenizer is None:
        tokenizer = WordPunctTokenizer()

    text_col_idx = df.columns.get_loc("preprocessed_text")

    # Tokenize text in training
    logger.debug("Tokenizing pre-processed text...")
    tokened_text = [tokenizer.tokenize(text) for text in df["preprocessed_text"]]
    df.insert(text_col_idx + 1, column="tokened", value=tokened_text)

    if do_stem:
        # Set default stemmer
        if stemmer is None:
            stemmer = PorterStemmer()

        # Create word stems
        logger.debug("Stemming tokened text...")
        stemmed_tokens = []
        for i in range(len(tokened_text)):
            stems = [stemmer.stem(token) for token in tokened_text[i]]
            stems = ' '.join(stems)
            stemmed_tokens.append(stems)

        df.insert(text_col_idx + 1, column="stemmed", value=stemmed_tokens)

    return df


def preprocess_for_sklearn_model(
    df                    : pd.DataFrame,
    tokenizer             : Union[TokenizerI, str] = None,
    stemmer               : Union[StemmerI, str] = None,
    do_remove_stopwords   : bool = False,
    do_stem               : bool = False
    ) -> pd.DataFrame:
    """ Pre-process dataframe prior to training a SKLearn model. """

    # Set default tokenizer
    if tokenizer is None:
        tokenizer = WordPunctTokenizer()
    if isinstance(tokenizer, str):
        try:
            tokenizer = getattr(nltk.tokenize, tokenizer)()
        except AttributeError as e:
            logger.warning(f"Bad tokenizer given: '{tokenizer}'. Supporting only 'nltk.tokenize' Tokenizer classes. Defaulting to 'WordPunctTokenizer'...")
            tokenizer = WordPunctTokenizer()

    # Set default stemmer
    if stemmer is None:
        stemmer = PorterStemmer()
    if isinstance(stemmer, str):
        try:
            stemmer = getattr(nltk.stem, stemmer)()
        except AttributeError as e:
            logger.warning(f"Bad stemmer given: '{stemmer}'. Supporting only 'nltk.stem' Stemmer classes. Defaulting to 'PorterStemmer'...")
            stemmer = PorterStemmer()

    # Clean text
    logger.debug(f"Cleaning {len(df)} reviews...")
    preprocessed_text = df["review"].apply(clean_text)

    # Insert pre-processed text to dataframe
    df.insert(1, column="preprocessed_text", value=preprocessed_text)

    # Remove stopwords
    if do_remove_stopwords:
        logger.debug(f"Removing stop words...")
        df["preprocessed_text"] = df["preprocessed_text"].apply(
            remove_stopwords, args=(STOPWORDS, tokenizer)
        )

    # Create tokenized & stemmed columns
    df = tokenize_and_stem_text(df=df, tokenizer=tokenizer, stemmer=stemmer, do_stem=do_stem)

    # Add label name column
    logger.debug("Adding 'label_name' column...")
    df["label_name"] = [IMDB_LABELS[label] for label in df.label]

    return df


# ========================================================================================= #
#                             KERAS PRE-PREPROCESSING FUNCS                                 #
# ========================================================================================= #

def preprocess_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Pre-process dataframe prior to training a Keras model.

    Args:
        df - Dataset dataframe.

    Returns:
        df - Dataset dataframe containing pre-processed text.
    """

    # TODO: Explore pre-processing options

    return df

# ========================================================================================= #
#                                        DATAFRAMES                                         #
# ========================================================================================= #

def create_dataframe(dataset_name: str, dataset_split: Literal["train", "test"] = None) -> pd.DataFrame:
    """ Creates a dataframe from a HuggingFace dataset. """

    logger.debug(f"Loading HuggingFace dataset '{dataset_name}' with split: '{dataset_split}'...")

    # Load the dataset
    dataset = huggingface_datasets.load_dataset(dataset_name)

    if dataset_split is not None:
        dataset = dataset[dataset_split]
    else:
        logger.debug(f"Concatenating 'train' and 'test' splits...")
        dataset_train   = dataset["train"]
        dataset_test    = dataset["test"]
        dataset         = huggingface_datasets.concatenate_datasets([dataset_train, dataset_test])

    # Create dataframe
    df              = pd.DataFrame()
    df["review"]    = dataset["text"]
    df["label"]     = dataset["label"]

    return df

def preprocess_dataframe(
    df            : pd.DataFrame,
    model_type    : utils.custom_types.ModelType,
    save_path     : utils.custom_types.PathType = None,
    *args, **kwargs
    ) -> pd.DataFrame:
    """
    Pre-process dataframe based on model type, each having its own key arguments.
    'sklearn' type has the following kwargs: ("tokenizer", "stemmer", "do_remove_stopwords").
    'keras' type has the following kwargs: ().

    Args:
        df -            Dataframe to pre-process.
        model_type -    Different model types have different pre-processing functions.
        save_path -     [OPTIONAL] Path to save pre-processed dataframe.

    Returns:
        df -            Pre-processed dataframe.
    """

    logger.debug(f"Pre-processing dataset for '{model_type.upper()}' model...")

    if model_type.lower() == "sklearn":
        df = preprocess_for_sklearn_model(df, **kwargs)

    elif model_type.lower() == "keras":
        df = preprocess_for_keras_model(df, **kwargs)

    else:
        raise TypeError(f"Bad model type given: '{model_type}'. Supported types: {utils.custom_types.ModelType.__args__}")

    logger.debug("Finished pre-processing dataset!")

    if save_path is not None:
        logger.debug(f"Saving pre-processed dataframe to '{save_path}'...")
        df.to_pickle(save_path)

    return df
