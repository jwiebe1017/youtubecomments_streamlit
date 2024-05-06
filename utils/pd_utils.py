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

def perform_ml_analysis(comments_df: pd.DataFrame) -> pd.DataFrame:
    return comments_df.assign(
                **{
                    "Sentiment": comments_df.text.apply(sentiment_analysis),
                    "NameCalling": comments_df.text.apply(namecalling_analysis),

                })
