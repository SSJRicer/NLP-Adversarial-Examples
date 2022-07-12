# Type-hints
from typing import Tuple, Sequence
from matplotlib.figure import Figure

# Arrays
import numpy as np

# Dataframes
import pandas as pd

# Text
import wordcloud

# Plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def combine_two_figures(
    fig1    : Figure,
    fig2    : Figure,
    title   : str,
    dpi     : int = 200,
    ):
    """ Combine two figures into one. """

    # Get figures' canvas
    c1 = fig1.canvas
    c2 = fig2.canvas

    # Draw canvas
    c1.draw()
    c2.draw()

    # Get axes data
    a1    = np.array(c1.buffer_rgba())
    a2    = np.array(c2.buffer_rgba())
    a     = np.hstack((a1, a2))

    # Display combined figure
    fig, ax = plt.subplots(figsize=(2000 / dpi, 1000 / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)
    plt.suptitle(title, y=0.85)

    plt.show()


def target_label_distribution(
    df            : pd.DataFrame,
    target_col    : str,
    figsize       : Tuple[float, float] = None,
    title         : str = "Label distribution",
    ):
    """ Plot a dataframe's target column (label) distribution. """

    # Data target/label distribution
    label_distribution = dict(df[target_col].value_counts())

    fig = plt.figure(figsize=figsize)
    plt.title(title)
    bars = plt.bar([str(v) for v in label_distribution.keys()], label_distribution.values())
    fig.axes[0].bar_label(bars)

    plt.show()

    return fig


def pos_vs_negative_distribution(
    df_pos    : pd.DataFrame,
    df_neg    : pd.DataFrame,
    figsize   : Tuple[float, float] = None,
    title     : str = "Positive vs Negative distribution",
    ):
    """ Positive vs Negative tokened pre-processed text distribution """

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.suptitle(title)
    sns.distplot(df_pos.tokened_len, ax=axes[0], color="Blue"), axes[0].set_title("Positive")
    sns.distplot(df_neg.tokened_len, ax=axes[1], color="Red"),  axes[1].set_title("Negative")

    plt.show()


def gen_wordcloud(
    text            : str,
    stopwords       : Sequence[str],
    max_words       : int = 1000,
    max_font_size   : int = 256,
    random_state    : int = 42,
    *args, **kwargs
    ):
    """ Generate Wordcloud for text. """

    wc = wordcloud.WordCloud(
        stopwords       = stopwords,
        max_words       = max_words,
        max_font_size   = max_font_size,
        random_state    = random_state,
        *args, **kwargs
    )
    wc.generate(text)

    return wc

def show_wordcloud(
    wc        : wordcloud.WordCloud,
    figsize   : Tuple[float, float] = None,
    title     : str = "WordCloud",
    ):
    """ Display text's wordcloud. """

    # Display it
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(wc)

    plt.show()


def show_duo_wordcloud(
    wc1       : wordcloud.WordCloud,
    wc2       : wordcloud.WordCloud,
    figsize   : Tuple[float, float] = None,
    title1    : str = "1",
    title2    : str = "2",
    suptitle  : str = "WordCloud",
    ):
    """ Display text's wordcloud. """

    # Display it
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(wc1), axes[0].set_title(title1)
    axes[1].imshow(wc2), axes[1].set_title(title2)
    plt.suptitle(suptitle, y=0.75)
    plt.tight_layout()

    plt.show()


def plot_user_ratings(
    ratings_pos   : Sequence[float],
    ratings_neg   : Sequence[float],
    title         : str = "User ratings (X/10)",
    ):
    """ Plot positive vs negative user ratings (found in reviews). """

    fig = plt.figure(figsize=(15, 5))
    sns.distplot(ratings_pos, color="green")
    sns.distplot(ratings_neg, color="red")
    plt.xlabel("Rating")
    plt.legend(["Positive", "Negative"])
    plt.suptitle(title)

    plt.show()


def plot_tokens_by_freq(
    tokens    : Sequence[str],
    freqs     : Sequence[int],
    figsize   : Tuple[float, float] = (14, 10),
    title     : str = "Top {topk} words by frequency",
    ):
    """ Plots tokens by frequency. """

    fig = plt.figure(figsize=figsize)
    sns.barplot(x=freqs, y=tokens)
    plt.title(title.format(topk=len(tokens)))

    plt.show()
