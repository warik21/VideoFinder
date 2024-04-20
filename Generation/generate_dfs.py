import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../../'))
from utils.utils import *
from utils.old_utils import *


def generate_embeddings_df(df: pd.DataFrame, path: str) -> tuple[list, list, list]:
    """
    Generate embeddings for each video in the dataframe and save them to a new csv file

    Args:
        df: pd.DataFrame - dataframe with videos data
        path: str - path to save the new csv file

    Returns:
        None
    """
    # Calculate embeddings for each video
    video_names = []
    video_embeddings_bert = []
    video_embeddings_mean_bert = []
    video_embeddings_hf = []

    embeddings_df = pd.DataFrame()
    # Populate lists with video names and embeddings
    for i in tqdm.tqdm(range(len(df.index))):
        video_name = df.iloc[i]['video_name']
        video_description = df.iloc[i]['video_description']
        video_transcript = df.iloc[i]['video_transcript']

        video_names.append(video_name)
        video_embeddings_bert.append(
            get_joint_embedding_bert(name=video_name, description=video_description, transcript=video_transcript))
        video_embeddings_mean_bert.append(
            get_joint_mean_embedding(name=video_name, description=video_description, transcript=video_transcript))
        video_embeddings_hf.append(
            get_joint_embedding(name=video_name, description=video_description, transcript=video_transcript))

        # Save embeddings to new df
        embeddings_df = embeddings_df._append({'video_name': video_name,
                                               'embedding_bert': video_embeddings_bert[-1],
                                               'embedding_mean_bert': video_embeddings_mean_bert[-1],
                                               'embedding_hf': video_embeddings_hf[-1]}, ignore_index=True)
    embeddings_df.to_csv(path, index=False)
    return video_embeddings_bert, video_embeddings_mean_bert, video_embeddings_hf, video_names


def generate_similarities_df(df: pd.DataFrame, path: str, video_embeddings_bert: list, video_embeddings_mean_bert: list,
                             video_embeddings_hf: list, video_names: list) -> tuple[pd.DataFrame, int, int, int]:
    """
    Compare each prompt to all video embeddings

    Args:
        df: pd.DataFrame - dataframe with videos data
        path: str - path to save the new csv file
        video_embeddings_bert: list - list of embeddings for each video
        video_embeddings_mean_bert: list - list of mean embeddings for each video
        video_embeddings_hf: list - list of embeddings for each video
        video_names: list - list of video names

    Returns:
        pd.DataFrame - dataframe with similarities
        int - number of correct matches for bert embeddings
        int - number of correct matches for hf embeddings
        int - number of correct matches for mean bert embeddings
        """
    correct_matches_bert = 0
    correct_matches_hf = 0
    correct_matches_mean_bert = 0

    # define similarities df:
    similarities_df = pd.DataFrame()
    # Compare each prompt to all video embeddings
    print("Calculating similarities...")
    for i in tqdm.tqdm(range(len(df))):
        prompt = df.iloc[i]['prompt']

        prompt_embedding_bert = get_embedding_bert(prompt)
        prompt_embedding_hf = get_embedding(prompt)
        prompt_embedding_mean_bert = get_mean_embedding(prompt)

        # Calculate similarities for each embedding type
        similarities_bert = [cosine_similarity(embed, prompt_embedding_bert)[0][0] for embed in video_embeddings_bert]
        similarities_hf = [cosine_similarity(embed, prompt_embedding_hf)[0][0] for embed in video_embeddings_hf]
        similarities_mean_bert = [cosine_similarity(embed, prompt_embedding_mean_bert)[0][0] for embed in
                                  video_embeddings_mean_bert]

        # Find the index of the most similar video
        closest_index_bert = np.argmax(similarities_bert)
        closest_index_hf = np.argmax(similarities_hf)
        closest_index_mean_bert = np.argmax(similarities_mean_bert)

        # Check if the closest video is the same as the current video
        if video_names[closest_index_bert] == video_names[i]:
            correct_matches_bert += 1
        if video_names[closest_index_hf] == video_names[i]:
            correct_matches_hf += 1
        if video_names[closest_index_mean_bert] == video_names[i]:
            correct_matches_mean_bert += 1

        # Save similarities to new df
        similarities_df = similarities_df._append({'video_name': video_names[i],
                                                   'prompt': prompt,
                                                   'similarity_bert': similarities_bert,
                                                   'similarity_hf': similarities_hf,
                                                   'similarity_mean_bert': similarities_mean_bert}, ignore_index=True)

    similarities_df.to_csv(path, index=False)

    correct_matches_dict = {'bert': correct_matches_bert,
                            'hf': correct_matches_hf,
                            'mean_bert': correct_matches_mean_bert}

    return similarities_df, correct_matches_dict


def generate_df(channels: list[str], path: str, num_videos=None) -> None:
    """
    Generate a dataframe with videos data from a list of channels
    
    Args:
        channels: list[str] - list of channel urls
        path: str - path to save the new csv file
        
    Returns:
        None
    """
    # api_key
    api_key = get_api_key(os.path.join('../', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))

    # Example Usage
    channels = ["https://www.youtube.com/@TwoMinutePapers"]
    df = pd.DataFrame()

    for channel in channels:
        channel_id, channel_name = extract_channel_id_and_name(download_html(channel))
        print(f"Processing channel: {channel_name}")
        df = add_channel_videos(channel_id, api_key, df=df, num_vids=num_videos)

    df.to_csv(path, index=False)


def generate_embeddings_hf_df(df: pd.DataFrame, path: str, model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')) -> pd.DataFrame:
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