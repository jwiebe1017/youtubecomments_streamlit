"""
module for YouTube api type functionalities
"""
from typing import Union
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st

from utils.config import youtube_client
from utils.utils import logging_setup

__author__ = 'jwiebe1017'
__version__ = '1.0.0'
__credits__ = ['stackoverflow', 'me, myself, and I', 'you I guess?']


log = logging_setup(__name__, False)


def get_yt_id(url: str) -> Union[str | None]:
    """
    Parses the url and looks for the ID based on specific patterns (as of Q2 2024).
    Returns the video ID if it matches - see examples below of working urls.

    Examples:
     - http://youtu.be/SA2iWivDJiE
     - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
     - http://www.youtube.com/embed/SA2iWivDJiE
     - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    credit: users/7732434/elijah

    :param str url: YouTube url akin to the examples above
    :return str | None: video ID if url fits formatting
    """

    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube_client.com', 'youtube_client.com',
                          'music.youtube_client.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/': return query.path.split('/')[2]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]


@st.cache_data
def request_youtube_videodetails(video_id: str) -> dict:
    """
    given a video ID, gather information relating to the video
    such as: title, description, author, etc.
    :param str video_id: YouTube video ID
    :return dict: video details
    """
    req = youtube_client.videos().list(
        part="snippet",
        id=video_id
    )
    vid_deets = req.execute()
    return vid_deets


@st.cache_data
def request_youtube_comment(video_id: str, page_num: int | None = None) -> dict:
    """
    given a video ID, gather information relating to the video comments section
    such as: comment, date, public, author
    :param str video_id: YouTube video ID
    :param int | None page_num: muted, results page number to use (if greater than 1 page)
    :return dict: details of comments contained in the video
    """
    yt_request = youtube_client.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
    )
    if page_num:
        yt_request.pageToken = page_num

    vid_comments = yt_request.execute()
    return vid_comments


def getcomments(video_id: str) -> pd.DataFrame:
    """
    given a YouTube video ID, parse comments with the client
    and return the results in a DataFrame

    credit: github.com/analyticswithadam
    :param video_id:
    :return:
    """
    comments = []

    # Execute the request.
    response = request_youtube_comment(
        video_id=video_id
    )

    # Get the comments from the response.
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
            comment['videoId'],
            public
        ])

    while True:
        try:
            next_page_token = response['next_page_token']
        except KeyError:
            break
        # Execute the next request.
        response = request_youtube_comment(
            video_id=video_id,
            page_num=next_page_token
        )
        # Get the comments from the next response.
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                comment['videoId'],
                public
            ])

    df2 = pd.DataFrame(
        comments,
        columns=['author', 'updated_at', 'like_count', 'text', 'video_id', 'public']
    )

    # reorder the cols for clarity
    return df2[[
        'public',
        'author',
        'text',
        'like_count',
        'updated_at',
        'video_id'
    ]]
