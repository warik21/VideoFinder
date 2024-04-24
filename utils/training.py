import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
from utils.utils import *
from utils.generate_dfs import *


class VideoRecommendationModel(nn.Module):
    def __init__(self, embed_dim):
        super(VideoRecommendationModel, self).__init__()
        self.embed_dim = embed_dim
        initial_weights = torch.tensor([0.3, 0.4, 0.3]).view(3, 1)  # Create a 3x1 tensor with the desired values
        self.weights = nn.Parameter(initial_weights)

    def forward(self, title_embeds, desc_embeds, trans_embeds, prompt_embed, relevant_indices):
        # Stack embeddings and ensure they are of shape [num_videos, 3, embed_dim]
        stacked_embeds = torch.stack([title_embeds, desc_embeds, trans_embeds], dim=1)

        # Apply softmax to weights to get normalized weights of shape [3, 1]
        weights = F.softmax(self.weights, dim=0)

        # Perform weighted sum by broadcasting weights across the embedding dimension
        weighted_embeds = stacked_embeds * weights.expand_as(stacked_embeds)

        # Sum across the second dimension (the one holding the three embedding types) to combine them
        joint_video_embeds = torch.sum(weighted_embeds, dim=1)

        # Normalize embeddings
        joint_video_embeds = F.normalize(joint_video_embeds, p=2, dim=1)
        prompt_embed = F.normalize(prompt_embed, p=2, dim=1)

        # Calculate cosine similarities
        cosine_similarities = torch.mm(joint_video_embeds, prompt_embed.t()).squeeze(1)  # Shape: [num_videos]

        # Filter cosine similarities based on relevant indices
        # relevant_cosine_similarities = cosine_similarities[relevant_indices]

        return cosine_similarities


class VideoDataset(Dataset):
    def __init__(self, title_embeds, desc_embeds, trans_embeds, prompt_embeds, relevant_indices):
        """
        Initializes the dataset.
        
        :param title_embeds: Tensor of shape [num_samples, embed_dim] - embeddings for the titles
        :param desc_embeds: Tensor of shape [num_samples, embed_dim] - embeddings for the descriptions
        :param trans_embeds: Tensor of shape [num_samples, embed_dim] - embeddings for the transcripts
        :param prompt_embeds: Tensor of shape [num_samples, embed_dim] - embeddings for the prompts
        :param relevant_indices: Tensor of shape [num_samples] - index of the relevant video for each prompt
        """
        self.title_embeds = title_embeds
        self.desc_embeds = desc_embeds
        self.trans_embeds = trans_embeds
        self.prompt_embeds = prompt_embeds
        self.relevant_indices = relevant_indices

    def __len__(self):
        return len(self.prompt_embeds)

    def __getitem__(self, idx):
        # Return the embeddings along with the index of the relevant video
        return (self.title_embeds[idx], self.desc_embeds[idx], self.trans_embeds[idx],
                self.prompt_embeds[idx], self.relevant_indices[idx])


#TODO: Make this a part of the VideoRecommendationModel class
def initialize_video_model(embed_dim):
    model = VideoRecommendationModel(embed_dim=embed_dim)
    return model

def train_BCE(vid_model, optimizer, loss_function, dataloader, epochs):
    vid_model.train()  # Set the model to training mode

    for epoch in range(epochs):
        total_loss = 0
        for title_batch, desc_batch, trans_batch, prompt_batch, indices_batch in dataloader:
            current_batch_size = title_batch.shape[0]
            optimizer.zero_grad()  # Clear gradients

            # Forward pass: Compute the cosine similarities
            cosine_similarities = vid_model(title_batch, desc_batch, trans_batch, prompt_batch, list(range(current_batch_size)))  # Shape: [batch_size, batch_size]
            
            # Create a target matrix where only the diagonal elements are 1
            targets = torch.eye(current_batch_size, dtype=torch.float32)

            # Compute loss
            loss = loss_function(cosine_similarities, targets)

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        # Optionally print average loss per epoch
        if (epoch + 1) % 10 == 0:
            average_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}')


def train_pairwise(vid_model, optimizer, dataloader, epochs):
    vid_model.train()  # Set the model to training mode
    margin = 1.0  # Margin for ranking loss
    criterion = nn.MarginRankingLoss(margin=margin)

    for epoch in range(epochs):
        total_loss = 0
        for title_batch, desc_batch, trans_batch, prompt_batch, indices_batch in dataloader:
            current_batch_size = title_batch.shape[0]
            optimizer.zero_grad()  # Clear gradients

            # Forward pass: Compute the cosine similarities
            cosine_similarities = vid_model(title_batch, desc_batch, trans_batch, prompt_batch, list(range(current_batch_size)))  # Shape: [batch_size, batch_size]
            
            # Compute the pairwise ranking loss
            loss = 0
            for i in range(current_batch_size):
                # Positive examples are the diagonal elements
                positive_examples = cosine_similarities[i, i].expand(current_batch_size - 1)
                # Negative examples are all non-diagonal elements in the row
                negative_examples = torch.cat([cosine_similarities[i, :i], cosine_similarities[i, i+1:]])
                # Targets: 1's indicating positive examples should have higher scores
                targets = torch.ones(current_batch_size - 1)
                loss += criterion(positive_examples, negative_examples, targets)

            loss /= current_batch_size  # Average the loss over the number of comparisons in the batch

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        # Optionally print average loss per epoch
        if (epoch + 1) % 10 == 0:
            average_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}')