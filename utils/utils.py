import re
from googleapiclient.discovery import build
import urllib.request
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain_community.vectorstores import DocArrayInMemorySearch


def get_api_key(api_key_path):
    """
    Reads the YouTube Data API key from a text file.

    Args:
        api_key_path: The path to the text file containing the API key.

    Returns:
        The API key as a string.
    """
    try:
        with open(api_key_path, 'r') as file:
            api_key = file.read().strip()
        return api_key
    except Exception as e:
        print(f"An error occurred while trying to read the key file: {e}")
        return None


def download_html(url):
    with urllib.request.urlopen(url) as response:
        html = response.read().decode()
    return html


def extract_channel_id_and_name(html_content):
    """Extracts the channel ID and name from a YouTube channel URL.

    Args:
        html_content: The HTML content of the YouTube channel page.

    Returns:
        A tuple containing the channel ID and channel name.
    """

    id_pattern = r'<link rel="canonical" href="https://www.youtube.com/channel/(.*?)"'
    title_pattern = r"<title>(.*?) - YouTube</title>"

    id_match = re.findall(id_pattern, html_content)
    title_match = re.findall(title_pattern, html_content)

    channel_id = id_match[0] if id_match else None
    channel_name = title_match[0] if title_match else None
    return channel_id, channel_name


def get_channel_videos(channel_id, api_key):
    """
    Retrieves a list of videos from a YouTube channel using the YouTube Data API.

    Args:
        channel_id: The ID of the YouTube channel.
        api_key: The YouTube Data API key.

    Returns:
        A list of video dictionaries containing video details.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    # Get the Uploads playlist ID

    res = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    next_page_token = None

    # Retrieve videos from the playlist
    while True:
        res = youtube.playlistItems().list(playlistId=playlist_id,
                                           part='snippet',
                                           maxResults=50,
                                           pageToken=next_page_token).execute()

        videos += res['items']
        next_page_token = res.get('nextPageToken')

        if next_page_token is None:
            break

    return videos


def get_video_transcript(video_id):
    """
    Retrieves the transcript of a YouTube video using the YouTubeTranscriptApi library.

    Args:
        video_id: The ID of the YouTube video.

    Returns:
        The full transcript of the video as a string.
    """
    try:
        # Attempt to fetch the transcript for the given video ID
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Concatenate all text items in the transcript list to form the full transcript
        transcript = "\n".join([item['text'] for item in transcript_list])
        return transcript
    except TranscriptsDisabled:
        # Transcripts are disabled for this video
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        # No transcript was found for this video
        return "No transcript found for this video."


def add_channel_videos(channel_id, api_key, num_vids=None, df=None):
    """
    Retrieves video details from a YouTube channel and appends them to a DataFrame.

    Args:
        channel_id: The ID of the YouTube channel.
        api_key: The YouTube Data API key.
        num_vids: The number of videos to retrieve from the channel.
        df: The DataFrame to append the video details to.

    Returns:
        The DataFrame containing the video details.
    """
    if df is None:
        df = pd.DataFrame()

    videos = get_channel_videos(channel_id, api_key)
    if num_vids is None:
        num_vids = len(videos)

    print(f"Processing {num_vids} videos from channel ID: {channel_id}")
    for video in videos[:num_vids]:
        video_id = video['snippet']['resourceId']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_description_ini = video['snippet']['description']
        video_transcript_ini = get_video_transcript(video_id)

        # Clean and summarize the description and transcript
        video_description = clean_description(video_description_ini)
        video_transcript = clean_transcript(video_transcript_ini)

        video_name = video['snippet']['title']
        channel_name = video['snippet']['channelTitle']
        df = df._append({'video_url': video_url,
                         'video_description': video_description,
                         'video_transcript': video_transcript,
                         'video_name': video_name,
                         'channel_name': channel_name,
                         }, ignore_index=True)
    return df


def clean_description(description):
    """
    Removes non-important information from a video description, using an existing template.

    Args:
        description: The video description to clean.

    Returns:
        The cleaned video description.
    """
    template = """
    Please summarize the video description by removing any non-important information such as timestamps, 
    links, or any other information that is not relevant to the main content of the video. 
    Start your summary with the phrase "The video summary based on the description is:" 
    and provide the summary right after.

    Descripition: {description}
    """

    prompt = PromptTemplate.from_template(template)
    prompt.format(description=description)

    chat_llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1", api_key="not-needed")
    chain = prompt | chat_llm
    response = chain.invoke({"description": description})
    clean_response = process_description(response.content)
    return clean_response


def process_description(response):
    prefix = "The video summary based on the description is:"
    if prefix in response:
        # Remove the prefix and return the summary part
        summary_text = response.split(prefix, 1)[1].strip()
        summary_cleaned = summary_text.strip('*').strip()
        return summary_cleaned
    else:
        # Handle cases where the model does not follow the format
        return "Summary extraction error: Unexpected response format."


def clean_transcript(transcript):
    """
    Removes non-important information from a video transcript, using an existing template.

    Args:
        transcript: The video transcript to clean.

    Returns:
        The cleaned video transcript.
    """
    template = """
    Please summarize the video transcript. 
    Start your summary with the phrase "The video summary based on the transcript is:" 
    and provide the summary right after.

    Transcript: {transcript}
    """

    prompt = PromptTemplate.from_template(template)
    prompt.format(transcript=transcript)

    chat_llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1", api_key="not-needed")
    chain = prompt | chat_llm

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)
    text_chunks = text_splitter.split_text(transcript)

    responses = []
    for chunk in text_chunks:
        time_start = time.time()
        response = chain.invoke({"transcript": chunk})
        clean_response = response.content
        responses.append(clean_response)
        time_end = time.time()
        print(f"Chunk processed in {time_end - time_start:.2f} seconds")

    return process_transcript(responses)


def process_transcript(responses):
    """
    Processes the responses from the model and combines them into a single cleaned transcript.

    Args:
        responses: A list of responses from the model.

    Returns:
        The combined cleaned transcript.
    """
    prefix = "The video summary based on the transcript is:"
    summary_parts = []
    for response in responses:
        if prefix in response:
            # Remove the prefix and return the summary part
            summary_text = response.split(prefix, 1)[1].strip()
            summary_cleaned = summary_text.strip('*').strip()
            summary_parts.append(summary_cleaned)
        else:
            # Handle cases where the model does not follow the format
            summary_parts.append("Summary extraction error: Unexpected response format.")
    return " ".join(summary_parts)
