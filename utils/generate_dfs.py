import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import sys
from utils.utils import *
from utils.old_utils import *
import langchain_community


def calculate_similarities(df: pd.DataFrame, video_embeddings_func, 
                           cue_embeddings_func) -> tuple[pd.DataFrame, int]:
    """
    Calculate similarities for each cue to all video embeddings.

    Args:
        df: pd.DataFrame - DataFrame with videos data
        video_embeddings_func: function - Function to get video embeddings
        cue_embeddings_func: function - Function to get cue embeddings

    Returns:
        pd.DataFrame - DataFrame with similarities for video embeddings and cues
        int - Number of correct matches
    """
    correct_matches = 0
    similarities_df = pd.DataFrame()

    for i in tqdm.tqdm(range(len(df))):
        cue = df.iloc[i]['cue']
        video_name = df.iloc[i]['Title']
        
        video_embedding = video_embeddings_func(video_name)
        cue_embedding = cue_embeddings_func(cue)
        
        similarity = cosine_similarity([video_embedding], [cue_embedding])[0][0]

        similarities_df = similarities_df.append({
            'Title': video_name,
            'cue': cue,
            'similarity': similarity
        }, ignore_index=True)

        if np.argmax(similarity) == i:
            correct_matches += 1

    return similarities_df, correct_matches

def generate_similarities_df(df: pd.DataFrame, path: str, video_embeddings_func, 
                             cue_embeddings_func) -> tuple[pd.DataFrame, int]:
    """
    Compare each cue to all video embeddings and save the results to a CSV file.

    Args:
        df: pd.DataFrame - DataFrame with videos data
        path: str - Path to save the new CSV file
        video_embeddings_func: function - Function to get video embeddings
        cue_embeddings_func: function - Function to get cue embeddings

    Returns:
        pd.DataFrame - DataFrame with similarities for video embeddings and cues
        int - Number of correct matches
    """
    similarities_df, correct_matches = calculate_similarities(df, video_embeddings_func, 
                                                              cue_embeddings_func)
    similarities_df.to_csv(path, index=False)

    return similarities_df, correct_matches


def generate_videos_df(channels: list[str], path: str,  model: Optional[langchain_community.llms.ollama.Ollama] = None,
                       num_videos=None, return_df=False) -> pd.DataFrame:
    """
    Generate a dataframe with videos data from a list of channels
    
    Args:
        channels: list[str] - list of channel urls
        path: str - path to save the new csv file
        num_videos: int - number of videos to download from each channel
        
    Returns:
        None
    
    Example:
        generate_df(channels=['https://www.youtube.com/@TwoMinutePapers'], path='csvs/testing_df.csv', num_videos=10)
    """
    api_key = get_api_key(os.path.join('../', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))
    df = pd.DataFrame()

    if model is None:
        print("Model not provided, using default model - gemma:instruct")
        initialize_model()
        model = get_model()

    for channel in channels:
        channel_id, channel_name = get_channel_id_and_name(download_html(channel))
        print(f"Processing channel: {channel_name}")
        df = add_channel_videos(channel_id, api_key, df=df, num_vids=num_videos, model=model)

    df.to_csv(path, index=False)
    if return_df:
        return df


def generate_embeddings_hf_df(df: pd.DataFrame, path: str, 
                              model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
                              ) -> pd.DataFrame:
    """
    Generate embeddings for each video in the dataframe and save them to a new csv file

    Args:
        df: pd.DataFrame - dataframe with videos data
        path: str - path to save the new csv file

    Returns:
        embeddings_df: pd.DataFrame - dataframe with video names and embeddings
    """
    # Calculate embeddings for each video
    video_names = []

    embeddings_df = pd.DataFrame(columns=['video_name', 'title_embedding', 'desc_embedding', 
                                          'transcript_embedding', 'prompt_embedding', 'label'])
    # Populate lists with video names and embeddings
    for i in tqdm.tqdm(range(len(df.index))):
        video_name = df.iloc[i]['video_name']
        video_description = df.iloc[i]['video_description']
        video_transcript = df.iloc[i]['video_transcript']
        prompt_text = df.iloc[i]['prompt']

        video_names.append(video_name)
        title_embedding = get_embedding(video_name, model)
        desc_embedding = get_embedding(video_description, model)
        transcript_embedding = get_embedding(video_transcript, model)
        prompt_embedding = get_embedding(prompt_text, model)

        # Save embeddings to new df
        embeddings_df.loc[i] = [video_names[i], title_embedding, desc_embedding, transcript_embedding, prompt_embedding, i]

    embeddings_df.to_csv(path, index=False)
    return embeddings_df