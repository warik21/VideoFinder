from utils.utils import *
from utils.generate_dfs import generate_videos_df, generate_similarities_df
import os
import pandas as pd

api_key = get_api_key(os.path.join('..', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
channels = ['https://www.youtube.com/@TwoMinutePapers', 'https://www.youtube.com/@bigthink', 'https://www.youtube.com/@TED']

initialize_model()
model = get_model()

videos_df = generate_videos_df(channels, 'csvs/videos_df.csv', num_videos=2, model=model, return_df=True)

# use generate_prompt on every row: 
videos_df['cue'] = videos_df.apply(lambda x: generate_prompt(x['Title'], x['Description'], x['Transcript'], model), axis=1)
video_names = videos_df['Title'].tolist()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

generate_similarities_df(videos_df, 'csvs/similarities.csv', model=model, return_df=True)