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


def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove social media handles
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove common non-content phrases (customize as needed)
    phrases_to_remove = ["Subscribe to my channel", "Follow me on"]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")
    return text


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

    return model.encode(text).reshape(1, -1)


def get_embedding_bert(text: str, model: Optional[DistilBertModel] = None,
                       tokenizer: Optional[DistilBertTokenizer] = None) -> list[float]:
    """
    Get the embedding of a text using a DistilBERT model. If no model is explicitly provided,
    the function will initialize and use the default 'distilbert-base-uncased' model.

    Args:
        text (str): The text to get the embedding for.
        model (Optional[DistilBertModel], optional): The DistilBERT model to use for embedding.
            If not specified, the function initializes the 'distilbert-base-uncased' model by default.
        tokenizer (Optional[DistilBertTokenizer], optional): The tokenizer to use for encoding the text.

    Returns:
        list[float]: The embedding of the text as a list of floats.

    Example:
        # Without specifying a model; uses the default model
        embedding = get_embedding_bert("This is a sample text.")
        print(embedding)

        # With a specified model
        custom_model = DistilBertModel.from_pretrained('your-custom-model-name')
        embedding = get_embedding_bert("This is a sample text.", model=custom_model)
        print(embedding)
    """
    if model is None:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    # Get the embeddings for the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    return embeddings.reshape(1, -1)


def get_joint_embedding_bert(name, description, transcript, model=None, tokenizer=None):
    """
    Generate a joint embedding for a video by combining the embeddings of its name, description, and transcript.

    Args:
        name (str): The name of the video.
        description (str): The description of the video.
        transcript (str): The transcript of the video.
        model (Optional[SentenceTransformer]): SentenceTransformer model to use for embedding.
        tokenizer (Optional[DistilBertTokenizer]): Tokenizer to use for encoding.

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
    if model is None:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    name_embedding = get_embedding_bert(name, model, tokenizer)
    description_embedding = get_embedding_bert(description, model, tokenizer)
    transcript_embedding = get_embedding_bert(transcript, model, tokenizer)

    # Combine the embeddings
    # TODO: add weights
    joint_embedding = 0.4 * name_embedding + 0.3 * description_embedding + 0.3 * transcript_embedding
    # TODO: Figure out why it returns a nested list [embedding] instead of just the embedding
    return joint_embedding[0].reshape(1,-1)


def get_mean_embedding(text, batch_size=32, model=None, tokenizer=None, device=None):
    """
    Generate a mean embedding for the given text using DistilBERT.

    Parameters:
    - text (str): Input text to encode.
    - batch_size (int): Size of batches to process the text.
    - model (Optional[DistilBertModel]): DistilBERT model to use for encoding.
    - tokenizer (Optional[DistilBertTokenizer]): Tokenizer to use for encoding.
    - device (Optional[str]): Device to use for computation (e.g., 'cuda' or 'cpu').

    Returns:
    - numpy.ndarray: The mean embedding of the input text.
    """
    if model is None:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Ensure the model and computation are on the same device (CPU or GPU)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Move tokens to the same device as model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Initialize a container for all embeddings
    all_embeddings = []

    # Process text in batches
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        batch_inputs = {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}

        with torch.no_grad():
            outputs = model(**batch_inputs)

        # Extract embeddings and perform mean pooling
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings and compute the overall mean
    all_embeddings = torch.cat(all_embeddings, dim=0)
    mean_embedding = all_embeddings.mean(dim=0).cpu().numpy()

    return mean_embedding.reshape(1, -1)


def get_joint_embedding(name: str, description: str, transcript: str, model=None) -> list[float]:
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
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    name_embedding = get_embedding(name, model)
    description_embedding = get_embedding(description, model)
    transcript_embedding = get_embedding(transcript, model)

    # Combine the embeddings
    joint_embedding = 0.4 * name_embedding + 0.3 * description_embedding + 0.3 * transcript_embedding
    return joint_embedding.reshape(1, -1)


def get_joint_mean_embedding(name: str, description: str, transcript: str, model=None,
                             tokenizer=None, device=None) -> list[float]:
    """
    Generate a joint mean embedding for a video by combining the mean embeddings of its name, description, and transcript.

    Args:
        name (str): The name of the video.
        description (str): The description of the video.
        transcript (str): The transcript of the video.
        model (Optional[DistilBertModel]): DistilBERT model to use for encoding.
        tokenizer (Optional[DistilBertTokenizer]): Tokenizer to use for encoding.
        device (Optional[str]): Device to use for computation (e.g., 'cuda' or 'cpu').

    Returns:
        list[float]: The joint mean embedding of the video's name, description, and transcript as a list of floats.

    Example:
        # Example usage for a video with a specific name, description, and transcript
        name = "Video Name"
        description = "Video Description"
        transcript = "Video Transcript"
        joint_embedding = get_joint_mean_embedding(name, description, transcript)
        print(joint_embedding)  # This will print the joint mean embedding vector as a list of floats.
    """
    if model is None:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    name_embedding = get_mean_embedding(name, model=model, tokenizer=tokenizer, device=device)
    description_embedding = get_mean_embedding(description, model=model, tokenizer=tokenizer, device=device)
    transcript_embedding = get_mean_embedding(transcript, model=model, tokenizer=tokenizer, device=device)

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


def generate_prompt(video_title: str, video_description: str, video_transcript: str,
                    max_length: int = 20) -> str:
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

    prompt = PromptTemplate.from_template(template)
    prompt.format(title=video_title, transcript=video_transcript, description=video_description)

    chat_llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1",
                          api_key="not-needed", max_tokens=max_length)
    chain = prompt | chat_llm

    generated_prompt = chain.invoke({"transcript": video_transcript,
                                     "description": video_description,
                                     "title": video_title})

    return generated_prompt.content
