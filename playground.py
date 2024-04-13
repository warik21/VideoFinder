from utils.utils import *
import pandas as pd
import time
import os
import numpy as np
from ast import literal_eval
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from utils.utils import clean_transcript

df = pd.read_csv('videos_df.csv')

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for i in range(len(df.index)):
    video_name = df.iloc[i]['video_name']
    video_description = df.iloc[i]['video_description']
    video_transcript = df.iloc[i]['video_transcript']

    # Make these include embedding everything
    embedding_bert = get_joint_embedding_bert(video_transcript)
    mean_embedding_bert = get_mean_embedding(video_transcript)
    embedding_hf = get_embedding(video_transcript)





for transcript in test_transcripts:
    start = time.time()
    mean_embedding = get_mean_embedding(text=transcript, model=model,
                                        tokenizer=tokenizer, device=device)
    print(f"Time taken to get mean embedding: {time.time() - start}")

for transcript in test_transcripts:
    start = time.time()
    # clean_transcript = clean_transcript(transcript)
    embedding = get_embedding(transcript)
    print(f"Time taken to clean transcript: {time.time() - start}")


accuracy = correct_matches / total_prompts
print(f"Accuracy: {accuracy * 100:.2f}%")

# create a search object
existing_embeddings = df['embedding'].apply(literal_eval)
existing_embeddings = [np.asarray(existing).reshape(1, -1) for existing in existing_embeddings]

similarities = [cosine_similarity(existing_embedding, inp_embedding) for existing_embedding in existing_embeddings]

closest_video = df.iloc[np.argmax(similarities)]

print(f"Closest video to your input is: {closest_video['video_name']}, Here is a link: {closest_video['video_url']}")
