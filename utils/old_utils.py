### This file is meant for old functions which will soon be deprecated from the project.

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