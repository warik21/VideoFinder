from utils.utils import *
import os
import pandas as pd


# api_key
api_key = get_api_key(os.path.join('..', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))

# Example Usage
channels = ["https://www.youtube.com/@juliensimonfr", "https://www.youtube.com/@sentdex"]
df = pd.DataFrame()

for channel in channels:
    channel_id, channel_name = extract_channel_id_and_name(download_html(channel))
    print(f"Processing channel: {channel_name}")
    df = add_channel_videos(channel_id, api_key, df=df)

df.to_csv('videos_df.csv', index=False)
