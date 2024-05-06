"""
module that mostly uses dataframe-based manipulations

"""
import pandas as pd

from utils.hf_utils import sentiment_analysis, namecalling_analysis
from utils.utils import logging_setup

__author__ = 'jwiebe1017'
__version__ = '1.0.0'
__credits__ = ['stackoverflow', 'me, myself, and I', 'you I guess?']

log = logging_setup(__name__, False)


def perform_ml_analysis(comments_df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
    """
    given a dataframe from the YouTube commentThread,
    analyze the comments text for sentiment and name calling using HF models

    :param pd.DataFrame comments_df: dataframe containing comments as strings
    :param str col: default, 'text', column to analyze
    :return pd.DataFrame: original dataframe with Sentiment and NameCalling columns added
    """
    return comments_df.assign(
        **{
            "Sentiment": comments_df[col].apply(sentiment_analysis),
            "NameCalling": comments_df[col].apply(namecalling_analysis),

        })
