import pandas as pd
from utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from Generation.generate_dfs import *
import os

df = generate_df(channels=['https://www.youtube.com/@TwoMinutePapers'], path='csvs/testing_df.csv', num_videos=10)