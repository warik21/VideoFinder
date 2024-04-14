import pandas as pd
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from Generation.generate_dfs import *
import os

df = pd.read_csv('videos_df.csv')
# subset for testing
df = df.head(5)

# Generate embeddings for each video
video_embeddings_bert, video_embeddings_mean_bert, video_embeddings_hf, video_names = generate_embeddings_df(df, 'embeddings2.csv')

# define similarities df:
sm_df, correct_matches_dict = generate_similarities_df(df, 'similarities2.csv', video_embeddings_bert,
                                                       video_embeddings_mean_bert, video_embeddings_hf, video_names)

print(f"Accuracy Bert: {correct_matches_dict['bert'] / len(df) * 100:.2f}%")
print(f"Accuracy HF: {correct_matches_dict['hf'] / len(df) * 100:.2f}%")
print(f"Accuracy Mean Bert: {correct_matches_dict['mean_bert'] / len(df) * 100:.2f}%")
