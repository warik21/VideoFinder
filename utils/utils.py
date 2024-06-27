import re
import tqdm
from googleapiclient.discovery import build
import numpy as np
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
from typing import Optional, Callable
import torch
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain_community
import logging
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer, AutoModel


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


def get_channel_id_and_name(html_content):
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


def get_channel_videos(channel_id, api_key, num_vids=None, max_retries=5, backoff_factor=1):
    """
    Retrieves a list of videos from a YouTube channel using the YouTube Data API with retry mechanism.

    Args:
        channel_id: The ID of the YouTube channel.
        api_key: The YouTube Data API key.
        num_vids: The number of videos to retrieve from the channel.
        max_retries: Maximum number of retries for the request in case of failure.
        backoff_factor: Factor for exponential backoff.

    Returns:
        A list of video dictionaries containing video details.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    res = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    videos = []
    next_page_token = None
    remaining_videos = num_vids

    while remaining_videos is None or remaining_videos > 0:
        max_results = min(remaining_videos, 50) if remaining_videos is not None else 50

        for retry in range(max_retries):
            try:
                res = youtube.playlistItems().list(playlistId=playlist_id,
                                           part='snippet',
                                           maxResults=max_results,
                                           pageToken=next_page_token).execute()

                videos += res['items']
                next_page_token = res.get('nextPageToken')
                break
            except HttpError as e:
                if e.resp.status in [500, 503]:  # Internal Server Error or Service Unavailable
                    logging.warning(f"Attempt {retry + 1} failed: {e}")
                    time.sleep(backoff_factor * (2 ** retry))
                else:
                    raise  # Reraise if it's not a 500 or 503 error
        else:
            raise Exception("Max retries exceeded when fetching videos")

        videos += res['items']
        next_page_token = res.get('nextPageToken')
        remaining_videos = remaining_videos - len(res['items']) if remaining_videos is not None else None

        if next_page_token is None or (remaining_videos is not None and remaining_videos <= 0):
            break

    return videos[:num_vids] if num_vids is not None else videos


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
    except TranscriptsDisabled:
        # Transcripts are disabled for this video
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        # No transcript was found for this video
        return "No transcript found for this video."
    except Exception as e:
        # Catch any other exceptions
        logging.error(f"An error occurred while retrieving the transcript for the video {video_id}: {e}")
        return "An error occurred while retrieving the transcript for this video."
    # Concatenate all text items in the transcript list to form the full transcript
    transcript = "\n".join([item['text'] for item in transcript_list])
    return transcript


def add_channel_videos(channel_id, api_key, model: langchain_community.llms.ollama.Ollama,
                       num_vids=None, df=None):
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

    videos = get_channel_videos(channel_id, api_key, num_vids=num_vids)
    if num_vids is None:
        num_vids = len(videos)

    print(f"Processing {num_vids} videos from channel ID: {channel_id}")
    for video in tqdm.tqdm(videos[:num_vids]):
        start = time.time()
        new_row = add_video_details(video, model)
        df = df._append(new_row, ignore_index=True)
        end = time.time()
        print(f"Processed video in {end - start:.2f} seconds.")
    return df


def add_video_details(video: dict, model: langchain_community.llms.ollama.Ollama) -> dict:
    """
    Adds video details to a dictionary, processing the video's description and transcript.

    Args:
        video: A dictionary containing details of a video.

    Returns:
        A tuple containing the dictionary with the video details and the number of transcript chunks processed.
    """
    # Extract video details
    video_id = video['snippet']['resourceId']['videoId']
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_name = video['snippet']['title']
    channel_name = video['snippet']['channelTitle']
    
    # Clean and summarize the description
    clean_description = clean_text(video['snippet']['description'])
    summarized_description = summarize_text(clean_description, "description", model)
    
    # Retrieve and process transcript
    clean_transcript = clean_text(get_video_transcript(video_id))
    summarized_transcript = summarize_text(clean_transcript, "transcript", model)
    
    # Prepare the video details
    new_row = {
        'Channel': channel_name,
        'URL': video_url,
        'Title': video_name,
        'Description': summarized_description,
        'Transcript': summarized_transcript
        }

    return new_row


def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove social media handles
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove common non-content phrases
    phrases_to_remove = ["Subscribe to my channel", "Follow me on"]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")
    return text


def summarize_text(text: str, text_type: str, 
                   model: langchain_community.llms.ollama.Ollama) -> str:
    """
    Summarizes the given text based on its type (description or transcript).

    Args:
        text: The video description or transcript to summarize.
        text_type: "description" or "transcript" indicating the type of text.

    Returns:
        The summarized text.
    """
    prefix = f"The video summary based on the {text_type} is:"    
    template = """
    Please summarize the video {text_type}. 
    Start your summary with the phrase "The video summary based on the {text_type} is:" 
    and provide the summary right after.

    text :{text}
    """
    responses = []
    parser = StrOutputParser()
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | parser
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=8192, chunk_overlap=512).split_text(text)
    for chunk in chunks:
        responses.append(chain.invoke({"text": chunk, "text_type": text_type}))
    response = " ".join(responses)
    
    return process_summary(response, prefix)


def process_summary(response: str, prefix: str) -> str:
    """
    Processes the summary response to remove all occurrences of a specified prefix and additional formatting.

    Args:
        response: The full response string from the summarization.
        prefix: The prefix that should be removed from the response.

    Returns:
        The cleaned summary text. If the prefix is not found even once, it returns an error message.
    """
    if prefix in response:
        summary_text = response.replace(prefix, "").strip()
        summary_text = summary_text.strip('*').strip()
        return summary_text
    else:
        return "Summary extraction error: Unexpected response format."


def get_embedding(text: str, encoding_model: SentenceTransformer = None,
                  device: str = "cuda:0") -> list[float]:
    """
    Get the embedding of a text using a SentenceTransformer model. If no model is explicitly provided,
    the function will initialize and use the default 'sentence-transformers/all-MiniLM-L6-v2' model.

    Args:
        text (str): The text to get the embedding for.
        encoding_model (Optional[SentenceTransformer], optional): The SentenceTransformer model to use for embedding.
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
    if encoding_model is None:
        encoding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    encoding_model.to(device)

    return encoding_model.encode(text).reshape(1, -1)


def get_joint_embedding(embedding_function: Callable, row: pd.Series, weights: list) -> np.ndarray:
    """
    Get a joint embedding for a dataframe row using the provided embedding function and weights.

    Args:
        embedding_function (function): Function that takes text input and returns its embedding.
        row (pd.Series): Row from a dataframe containing 'Title', 'Description', and 'Transcript' columns.
        weights (list): List of weights for combining embeddings from different text fields.

    Returns:
        np.ndarray: Joint embedding.

    Example:
        # Example usage for getting a joint embedding
        bert_embedding_func = get_embedding_function("bert-base-uncased")
        row = {'Title': "Example title", 'Description': "Example description", 'Transcript': "Example transcript"}
        weights = [0.4, 0.3, 0.3]
        joint_embedding = get_joint_embedding(bert_embedding_func, row, weights)
        print(joint_embedding)  # This will print the joint embedding vector as a numpy array.
    """
    joint_embedding = np.zeros_like(embedding_function(""))

    for text_field, weight in zip(['Title', 'Description', 'Transcript'], weights):
        text = row[text_field]
        embedding = embedding_function(text)
        joint_embedding += embedding * weight

    return joint_embedding


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


def generate_prompt(video_title: str, video_description: str, video_transcript: str, 
                    model: Optional[ChatOpenAI] = None) -> str:
    """
    Generate a set of prompts based on the video description and transcript for evaluating a language model.

    This function generates prompts that can be used to evaluate a language model's performance on tasks related to the video content. The prompts are designed to test the model's ability to understand and respond to queries or requests based on the video description and transcript.

    Args:
        video_title (str): The title of the video
        video_description (str): The description of the video
        video_transcript (str): The transcript of the video

    Returns:
        str: A list of prompts generated based on the video description and transcript.

    Example:
        title = "Video Title"
        description = "This is the video description."
        transcript = "This is the video transcript."
        prompts = generate_prompt(title, description, transcript)
        print(prompts)  # This will print a list of prompts based on the video content.
    """
    # Define the template for generating prompts
    template = """
    Imagine you're exploring a digital library of videos and are interested in topics covered in the video below. 
    Using the title, description and transcript provided, craft a general user-like query that someone might use when searching 
    for a video on a broader topic. Start your query with 'A video about' and ensure it captures the overarching themes 
    or general ideas rather than specific details. Aim for a query that could help someone decide if this video aligns 
    with their general interest in a subject.

    Title: {title}
    Description: {description}
    Transcript: {transcript}
    """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | model | StrOutputParser()

    generated_prompt = chain.invoke({"transcript": video_transcript,
                                     "description": video_description,
                                     "title": video_title})

    return generated_prompt


def initialize_embedding_model(model_name: Optional[str]='all-MiniLM-L6-v2'):
    global model_embedding
    model_embedding = SentenceTransformer(model_name)


def initialize_model(model_name: Optional[str] = 'gemma:instruct'):
    """
    Initialize the Ollama model for generating predictions.

    Args:
        model_name: str - the name of the model to use for predictions
        
    Returns:
        None

    Example:
        initialize_model('gemma:7b-instruct')
    """
    global model
    model = Ollama(model=model_name)


def get_model():
    """
    Get the initialized model.
    
    Returns:
        The initialized model
    """
    return model
    

def generate_predictions(similarities_df: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.DataFrame(columns=['video_id', 'hf'])
    video_names = similarities_df.index.tolist()

    for i in range(len(similarities_df.index)):
        hf = 0

        similarities_hf = similarities_df.iloc[i]['similarity_hf']

        closest_index_hf = np.argmax(similarities_hf)

        if video_names[closest_index_hf] == video_names[i]:
            hf = 1

        predictions.loc[i] = [video_names[i], hf]

    return predictions


def get_embedding_function(model_name, device="cuda:0") -> Callable[[str], list]:
    """
    Get an embedding function for a given model name.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        function: A function that takes text input and returns its embedding.

    Example:
        embedding_function = get_embedding_function('sentence-transformers/all-MiniLM-L6-v2')
        embedding = embedding_function("This is a sample text.")
        print(embedding)
    """
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    # Define the embedding function
    def embedding_function(text):
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []
        model.encode(text).reshape(1, -1)     
        
    return embedding_function