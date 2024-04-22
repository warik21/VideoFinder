import pandas as pd
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


# df = generate_df(channels=['https://www.youtube.com/@TwoMinutePapers', 'https://www.youtube.com/@TED'], 
#                  path='csvs/testing_df.csv', num_videos=5)


MODEL = 'gemma:2b-instruct'
model = initialize_model(MODEL)

api_key = get_api_key(os.path.join('../', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))
channels=['https://www.youtube.com/@TwoMinutePapers', 'https://www.youtube.com/@TED']

df = pd.DataFrame()
for channel in channels:
    channel_id, channel_name = get_channel_id_and_name(download_html(channel))
    print(f"Processing channel: {channel_name}")
    df = add_channel_videos(channel_id, api_key, df=df, num_vids=5)

df.to_csv('csvs/testing_df.csv', index=False)