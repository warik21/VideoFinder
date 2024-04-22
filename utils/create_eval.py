import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../../'))
from utils import generate_prompt

df = pd.read_csv('csvs/videos_df.csv')

prompts = []

for i in range(len(df.index)):
    print(f"Video {i + 1}: {df.iloc[i]['video_name']}")
    generated_prompt = generate_prompt(video_description=df.iloc[i]['video_description'],
                                       video_title=df.iloc[i]['video_name'],
                                       video_transcript=df.iloc[i]['video_transcript'])
    print(f"Prompt: {generated_prompt}")
    prompts.append(generated_prompt)

df['prompt'] = prompts
df.to_csv('../videos_df.csv', index=False)
