import pandas as pd

import matplotlib.pyplot as plt


def target_label_distribution(df: pd.DataFrame, target_col: str):
    """ Plot a dataframe's target column (label) distribution. """

    # Data target/label distribution
    label_distribution = dict(df[target_col].value_counts())

    fig = plt.figure()
    plt.title("Label distribution")
    plt.bar([str(v) for v in label_distribution.keys()], label_distribution.values())

    return fig
