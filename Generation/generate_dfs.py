import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../../'))
from utils.utils import *


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
