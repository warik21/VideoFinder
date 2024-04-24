import pandas as pd
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from utils.generate_dfs import generate_df, generate_embeddings_hf_df
from utils.training import VideoDataset, initialize_video_model, train_BCE, train_pairwise
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

initialize_embedding_model('all-MiniLM-L6-v2')
initialize_model('gemma:2b-instruct')
embedding_dim = 384

videos_df = pd.read_csv('csvs/videos_df.csv')
similarities_df = pd.read_csv('csvs/similarities.csv')

predictions = pd.DataFrame(columns=['video_id', 'hf'])
video_names = similarities_df.index.tolist()

#TODO : make this a function, which takes in the similarities df and returns the accuracy
for i in range(len(similarities_df.index)):
    hf = 0

    similarities_hf = similarities_df.iloc[i]['similarity_hf']

    closest_index_hf = np.argmax(similarities_hf)

    if video_names[closest_index_hf] == video_names[i]:
        hf = 1

    predictions.loc[i] = [video_names[i], hf]

print(f"Accuracy HF: {predictions['hf'].sum() / len(similarities_df.index) * 100:.2f}%")

embeddings_df = generate_embeddings_hf_df(videos_df, 'csvs/embeddings_hf.csv')

title_embeds = torch.tensor(embeddings_df['title_embedding'], dtype=torch.float32).squeeze()
desc_embeds = torch.tensor(embeddings_df['desc_embedding'], dtype=torch.float32).squeeze()
trans_embeds = torch.tensor(embeddings_df['transcript_embedding'], dtype=torch.float32).squeeze()
prompt_embeds = torch.tensor(embeddings_df['prompt_embedding'], dtype=torch.float32).squeeze()
relevant_indices = torch.tensor(embeddings_df['label'], dtype=torch.long)

video_dataset = VideoDataset(title_embeds, desc_embeds, trans_embeds, prompt_embeds, relevant_indices)
batch_size = 32
video_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Baseline
video_recommending_model = initialize_video_model(embedding_dim)
predictions = pd.DataFrame(columns=['video_id', 'hf'])
cosine_similarities = video_recommending_model(title_embeds, desc_embeds, trans_embeds, prompt_embeds, relevant_indices)
video_names = similarities_df.index.tolist()
true_index = list(range(len(video_names)))
predicted_index = torch.argmax(cosine_similarities, dim=1).tolist()
print(f'Accuracy: {sum([1 for i, j in zip(true_index, predicted_index) if i == j]) / len(true_index) * 100:.2f}%')

# Train using BCE loss
video_recommending_model = initialize_video_model(embedding_dim)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(video_recommending_model.parameters(), lr=0.001)  # You can adjust the learning rate as needed
num_epochs = 100
train_BCE(video_recommending_model, optimizer, loss_function, video_dataloader, num_epochs)
video_recommending_model.eval()  # Set the model to evaluation mode
predictions = pd.DataFrame(columns=['video_id', 'hf'])
cosine_similarities = video_recommending_model(title_embeds, desc_embeds, trans_embeds, prompt_embeds, relevant_indices)
video_names = similarities_df.index.tolist()
true_index = list(range(len(video_names)))
predicted_index = torch.argmax(cosine_similarities, dim=1).tolist()
print(f'Accuracy: {sum([1 for i, j in zip(true_index, predicted_index) if i == j]) / len(true_index) * 100:.2f}%')


# Train using Pairwise ranking loss
num_epochs = 100
video_recommending_model = initialize_video_model(embedding_dim)
# Call the train function
optimizer = optim.Adam(video_recommending_model.parameters(), lr=1e-3)
train_pairwise(video_recommending_model, optimizer, video_dataloader, num_epochs)
video_recommending_model.eval()  # Set the model to evaluation mode
predictions = pd.DataFrame(columns=['video_id', 'hf'])
cosine_similarities = video_recommending_model(title_embeds, desc_embeds, trans_embeds, prompt_embeds, relevant_indices)
video_names = similarities_df.index.tolist()
true_index = list(range(len(video_names)))
predicted_index = torch.argmax(cosine_similarities, dim=1).tolist()
print(f'Accuracy: {sum([1 for i, j in zip(true_index, predicted_index) if i == j]) / len(true_index) * 100:.2f}%')
