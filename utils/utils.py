import re
import tqdm
from googleapiclient.discovery import build
import urllib.request
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import torch


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
    for video in tqdm.tqdm(videos[:num_vids]):
        start = time.time()
        new_row, num_chunks = add_video_details(video)
        df = df._append(new_row, ignore_index=True)
        end = time.time()
        print(f"Processed video in {end - start:.2f} seconds. Transcript split into {num_chunks} chunks.")
    return df


def add_video_details(video: dict):
    """
    Adds video details to a DataFrame.

    Args:
        video: A dictionary containing details of a video.

    Returns:
        The DataFrame containing the video details.
    """
    video_id = video['snippet']['resourceId']['videoId']
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_description = clean_description(video['snippet']['description'])
    video_transcript, num_chunks = clean_transcript(get_video_transcript(video_id))
    embedding = weighted_embed_text(video_description, video_transcript)

    # Clean and summarize the description and transcript
    new_row = {'video_url': video_url,
               'video_description': video_description,
               'video_transcript': video_transcript,
               'video_name': video['snippet']['title'],
               'channel_name': video['snippet']['channelTitle'],
               'embedding': embedding
               }
    return new_row, num_chunks


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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8192, chunk_overlap=512)
    text_chunks = text_splitter.split_text(transcript)

    responses = []
    for chunk in text_chunks:
        response = chain.invoke({"transcript": chunk})
        clean_response = response.content
        responses.append(clean_response)

    return process_transcript(responses), len(text_chunks)


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


def get_embedding(text: str, model: Optional[SentenceTransformer] = None) -> list[float]:
    """
    Get the embedding of a text using a SentenceTransformer model. If no model is explicitly provided,
    the function will initialize and use the default 'sentence-transformers/all-MiniLM-L6-v2' model.

    Args:
        text (str): The text to get the embedding for.
        model (Optional[SentenceTransformer], optional): The SentenceTransformer model to use for embedding.
            If not specified, the function initializes the 'all-MiniLM-L6-v2' model by default.

    Returns:
        list[float]: The embedding of the text as a list of floats.

    Example:
        # Without specifying a model; uses the default model
        embedding = get_embedding("This is a sample text.")
        print(embedding)

        # With a specified model
        custom_model = SentenceTransformer('your-custom-model-name')
        embedding = get_embedding("This is a sample text.", model=custom_model)
        print(embedding)
    """
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model.encode(text)


def get_similarity_score(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Get the similarity score between two embeddings using cosine similarity.

    Args:
        embedding1: The first embedding.
        embedding2: The second embedding.

    Returns:
        The similarity score between the two embeddings.
    """
    if not embedding1 or not embedding2:
        print("Attempted to get similarity score for empty embeddings.")
        return 0.0

    similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

    return similarity_score


def weighted_embed_text(transcript: str, description: str, weight_transcript: float = 0.7,
                        weight_description: float = 0.3) -> list[float]:
    """
    Generate a weighted embedding for a video by separately embedding its transcript and description, then combining these embeddings using specified weights.

    This function allows for the emphasis of certain parts of the video's content over others in the resulting embedding. By adjusting the weights for the transcript and the description, users can tailor the embedding to better suit their needs for comparing video content to search queries or other texts.

    Args:
        transcript (str): The transcript of the video to be embedded.
        description (str): The description of the video to be embedded.
        weight_transcript (float, optional): The weight to be applied to the transcript's embedding. Defaults to 0.7.
        weight_description (float, optional): The weight to be applied to the description's embedding. Defaults to 0.3.

    Returns:
        list[float]: The weighted embedding of the video's transcript and description as a list of floats. This embedding represents a single vector that combines the information from both the transcript and the description, adjusted by their respective weights.

    Note:
        The function assumes the existence of a previously defined `embed_text` function that takes a string as input and returns its embedding as a list of floats. It is also assumed that the embeddings generated by `embed_text` are of a fixed size and directly comparable (i.e., they are produced by the same model or process).

    Example:
        # Example usage for a video with a specific transcript and description
        transcript = "Here is the video transcript."
        description = "Here is the video description."
        weighted_embedding = weighted_embed_text(transcript, description, 0.7, 0.3)
        print(weighted_embedding)  # This will print the weighted embedding vector as a list of floats.
    """
    transcript_embedding = get_embedding(transcript)
    description_embedding = get_embedding(description)

    # Applying weights
    weighted_embedding = (transcript_embedding * weight_transcript) + (description_embedding * weight_description)
    return weighted_embedding.tolist()
