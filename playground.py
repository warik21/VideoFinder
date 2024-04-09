from utils.utils import *
import os
import pandas as pd
import time
import numpy as np
from ast import literal_eval


df = pd.read_csv('videos_df3.csv')
# subset first 5 videos
df = df.head(5)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# initiate a text prompt to compare to the embeddings of the videos
inp = input("What do you want to watch?")

# encode the input
start_time = time.time()
inp_embedding = get_embedding(inp).reshape(1, -1)
end_time = time.time()
execution_time = end_time - start_time
print(f"The execution time of get_embedding function is: {execution_time} seconds")

# create a search object
existing_embeddings = df['embedding'].apply(literal_eval)
existing_embeddings = [np.asarray(existing).reshape(1, -1) for existing in existing_embeddings]

similarities = [cosine_similarity(existing_embedding, inp_embedding) for existing_embedding in existing_embeddings]

closest_video = df.iloc[np.argmax(similarities)]

print(f"Closest video to your input is: {closest_video['video_name']}, Here is a link: {closest_video['video_url']}")
