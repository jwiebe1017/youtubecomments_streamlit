"""
module that mostly uses huggingface based functions
"""
import numpy as np
from scipy.special import softmax

from utils.config import sentiment_tokenizer, sentiment_model, sentiment_config, \
    namecalling_classifier
from utils.utils import logging_setup

__author__ = 'jwiebe1017'
__version__ = '1.0.0'
__credits__ = ['stackoverflow', 'me, myself, and I', 'you I guess?']

log = logging_setup(__name__, False)


def preprocess(text):
    """

    :param text:
    :return:
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t  # replace usernames
        t = 'http' if t.startswith('http') else t  # replace link refs
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text: str) -> str:
    """

    :param text:
    :return:
    """
    text = preprocess(text)
    try:
        encoded_input = sentiment_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        output = sentiment_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        return sentiment_config.id2label[ranking[0]]
    except RuntimeError:
        log.info(f'could not process {text} in Sentiment Analysis')
        return 'Could not extract'


def namecalling_analysis(text: str):
    """

    :param text:
    :return:
    """
    try:
        output = namecalling_classifier(text)
        return output[0]['label']
    except RuntimeError:
        log.info(f'could not process {text} in Sentiment Analysis')
        return 'Could not extract'
