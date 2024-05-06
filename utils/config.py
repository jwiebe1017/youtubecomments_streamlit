"""
project configs + inits for injection

"""
import pathlib
from typing import Tuple, Any

import googleapiclient.discovery
import streamlit as st
import yaml
import keyring
import streamlit as st
from keyring.errors import NoKeyringError
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    TextClassificationPipeline

from utils.utils import logging_setup

__author__ = 'jwiebe1017'
__version__ = '1.0.0'
__credits__ = ['stackoverflow', 'me, myself, and I', 'you I guess?']

log = logging_setup(__name__, False)


@st.cache_resource
def init_sent_model(model_name: str) -> Tuple[Any, Any, Any]:
    """
    cache model if available. If not, download model and save locally -> re-init from local

    :param str model_name: huggingface model name, e.g. 'user/model_name'
    :return tuple: (tokenizer, config, model) from autoloaders
    """
    try:
        return AutoTokenizer.from_pretrained(model_name), \
               AutoConfig.from_pretrained(model_name), \
               AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        log.warning(e)
        p = pathlib.Path(model_name)
        model_ = pathlib.Path(*p.parts[2:])
        tokenizer = AutoTokenizer.from_pretrained(model_)
        config = AutoConfig.from_pretrained(model_)
        model = AutoModelForSequenceClassification.from_pretrained(model_)
        # save things out
        tokenizer.save_pretrained(model_name)
        model.save_pretrained(model_name)
        config.save_pretrained(model_name)

        # re-init from local
        return AutoTokenizer.from_pretrained(model_name), \
               AutoConfig.from_pretrained(model_name), \
               AutoModelForSequenceClassification.from_pretrained(model_name)


@st.cache_resource
def init_nc_model(model_name: str) -> TextClassificationPipeline:
    """
    cache model if available. If not, download model and save locally -> re-init from local

    :param str model_name: huggingface model name, e.g. 'user/model_name'
    :return TextClassificationPipeline: pipeline of loaded tokenizer and model
    """
    try:
        return TextClassificationPipeline(
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            model=AutoModelForSequenceClassification.from_pretrained(model_name))
    except Exception as e:
        log.warning(e)
        p = pathlib.Path(model_name)
        model_ = pathlib.Path(*p.parts[2:])
        tokenizer = AutoTokenizer.from_pretrained(model_)
        model = AutoModelForSequenceClassification.from_pretrained(model_)
        # save things out
        tokenizer.save_pretrained(model_name)
        model.save_pretrained(model_name)

        # re-init from local
        return TextClassificationPipeline(
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            model=AutoModelForSequenceClassification.from_pretrained(model_name))


with open('./utils/config.yaml') as cfg:
    data = yaml.safe_load(cfg)

# local runs have the key in keyring
try:
    google_api_key = keyring.get_password('local_user', 'google_api_key_youtubecomments')
# ideally this is not just hanging in config as a  string...
except NoKeyringError:
    try:
        google_api_key = data['GOOGLE_API_KEY']
    except TypeError:
        # allow user to just throw it in via app
        google_api_key = st.text_input('Please Paste Google API Key:')


youtube_client = googleapiclient.discovery.build(
    serviceName="youtube",
    version="v3",
    developerKey=google_api_key
)

# define model locations (local ./models/...)
SENTIMENT_MODEL = data['SENTIMENT_MODEL']
NAMECALLING_MODEL = data['NAMECALLING_MODEL']

# cache their inits
sentiment_tokenizer, sentiment_config, sentiment_model = init_sent_model(SENTIMENT_MODEL)
namecalling_classifier = init_nc_model(NAMECALLING_MODEL)
