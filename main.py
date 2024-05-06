"""
main ideas from:
https://github.com/analyticswithadam/Python/blob/main/YouTube_Comments_Advanced.ipynb

Allows users to enter url to YouTube video - and then can add inference from models to the text
"""
import streamlit as st

from utils.pd_utils import perform_ml_analysis
from utils.utils import logging_setup
from utils.yt_utils import get_yt_id, getcomments, request_youtube_videodetails

__author__ = 'jwiebe1017'
__version__ = '1.0.0'
__credits__ = ['stackoverflow', 'me, myself, and I', 'you I guess?']


if __name__ == '__main__':
    log = logging_setup(__name__, False)

    st.title('YouTube Comment Analysis')
    yt_url = st.sidebar.text_input("Please insert your YouTube URL to analyze")
    log.info(f'pulling ID from {yt_url}')

    yt_id = get_yt_id(yt_url)
    if yt_url:
        st.subheader(request_youtube_videodetails(yt_id)['items'][0]['snippet']['title'])
        comments_df = getcomments(yt_id)
        sent_comments_df = comments_df.copy()
        if st.sidebar.checkbox('Perform ML Analysis', key='ml_analysis'):
            log.info('ML analysis requested')
            sent_comments_df = perform_ml_analysis(sent_comments_df)

        if not sent_comments_df.equals(comments_df):
            comments_df = sent_comments_df.copy()


        filt_cols = st.sidebar.multiselect('Filter to Columns:', comments_df.columns.values)
        filt_posnegneut = st.sidebar.multiselect(
            'Filter Sentiment:',
            comments_df.Sentiment.unique()
        ) if 'Sentiment' in comments_df.columns.values else None
        filt_namecalling = st.sidebar.multiselect(
            'Filter NameCalling:',
            comments_df.NameCalling.unique()
        ) if 'NameCalling' in comments_df.columns.values else None

        if filt_cols:
            comments_df = comments_df[filt_cols]

        if filt_posnegneut:
            mask = comments_df.Sentiment.isin(filt_posnegneut)
            comments_df = comments_df[mask]

        if filt_namecalling:
            mask = comments_df.NameCalling.isin(filt_posnegneut)
            comments_df = comments_df[mask]

        log.info('display data_editor')
        st.data_editor(
            comments_df,
        )
