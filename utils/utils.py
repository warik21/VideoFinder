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
from typing import Optional
import torch
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import DistilBertModel, DistilBertTokenizer


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
        new_row = add_video_details(video)
        df = df._append(new_row, ignore_index=True)
        end = time.time()
        print(f"Processed video in {end - start:.2f} seconds.")
    return df


def add_video_details(video: dict) -> dict:
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
    summarized_description = summarize_text(clean_description, "description")
    
    # Retrieve and process transcript
    clean_transcript = clean_text(get_video_transcript(video_id))
    summarized_transcript = summarize_text(clean_transcript, "transcript")
    
    # Assuming weighted_embed_text is updated or replaced to handle embeddings
    embedding = get_joint_embedding(video_name, clean_description, clean_transcript)

    # Prepare the video details
    new_row = {
        'video_url': video_url,
        'video_description': summarized_description,
        'video_transcript': summarized_transcript,
        'video_name': video_name,
        'channel_name': channel_name,
        'embedding': embedding
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


def summarize_text(text: str, text_type: str) -> str:
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


def get_embedding(text: str, encoding_model: Optional[SentenceTransformer] = None) -> list[float]:
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoding_model.to(device)

    return encoding_model.encode(text).reshape(1, -1)


def get_joint_embedding(name: str, description: str, transcript: str, encoding_model=None) -> list[float]:
    """
    Generate a joint embedding for a video by combining the embeddings of its name, description, and transcript.

    Args:
        name (str): The name of the video.
        description (str): The description of the video.
        transcript (str): The transcript of the video.
        model (Optional[SentenceTransformer]): SentenceTransformer model to use for embedding.

    Returns:
        list[float]: The joint embedding of the video's name, description, and transcript as a list of floats.

    Example:
        # Example usage for a video with a specific name, description, and transcript
        name = "Video Name"
        description = "Video Description"
        transcript = "Video Transcript"
        joint_embedding = get_joint_embedding(name, description, transcript)
        print(joint_embedding)  # This will print the joint embedding vector as a list of floats.
    """
    if encoding_model is None:
        encoding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    name_embedding = get_embedding(name, encoding_model)
    description_embedding = get_embedding(description, encoding_model)
    transcript_embedding = get_embedding(transcript, encoding_model)

    # Combine the embeddings
    joint_embedding = 0.4 * name_embedding + 0.3 * description_embedding + 0.3 * transcript_embedding
    return joint_embedding.reshape(1, -1)


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
                    chat_llm: ChatOpenAI = None, max_length: int = 20) -> str:
    """
    Generate a set of prompts based on the video description and transcript for evaluating a language model.

    This function generates prompts that can be used to evaluate a language model's performance on tasks related to the video content. The prompts are designed to test the model's ability to understand and respond to queries or requests based on the video description and transcript.

    Args:
        video_title (str): The title of the video
        video_description (str): The description of the video
        video_transcript (str): The transcript of the video
        max_length (int, optional): The maximum length of each prompt. Defaults to 100.

    Returns:
        list[str]: A list of prompts generated based on the video description and transcript.

    Example:
        # Example usage for generating prompts
        description = "This is the video description."
        transcript = "This is the video transcript."
        prompts = generate_prompt(description, transcript, 100)
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

    prompt = PromptTemplate.from_template(template).format(title=video_title, 
                                                           transcript=video_transcript, 
                                                           description=video_description)

    if chat_llm is None:
        chat_llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1",
                          api_key="not-needed", max_tokens=max_length)
        
    chain = prompt | chat_llm

    generated_prompt = chain.invoke({"transcript": video_transcript,
                                     "description": video_description,
                                     "title": video_title})

    return generated_prompt.content


def initialize_embedding_model(model_name: Optional[str]='all-MiniLM-L6-v2'):
    global model_embedding
    model_embedding = SentenceTransformer(model_name)


def initialize_model(model_name: Optional[str]='gemma:2b-instruct'):
    global model
    model = Ollama(model=model_name)
    # global embeddings
    # embeddings = OllamaEmbeddings(model)
    

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