import pandas as pd
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from utils.generate_dfs import generate_df

MODEL = 'gemma:2b-instruct'
model = initialize_model(MODEL)
channels=['https://www.youtube.com/@TwoMinutePapers', 'https://www.youtube.com/@TED']

generate_df(channels=channels, path='csvs/testing_df.csv', num_videos=5)
