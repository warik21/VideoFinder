from utils.utils import *
import os
import pandas as pd


api_key = get_api_key(os.path.join('..', 'keys', 'VideoFinder', 'YouTubeAPIKey.txt'))

# Example Usage
# channels = ["https://www.youtube.com/@juliensimonfr", "https://www.youtube.com/@sentdex"]
channels = ["https://www.youtube.com/@juliensimonfr"]
df = pd.DataFrame()

for channel in channels:
    channel_id, channel_name = extract_channel_id_and_name(download_html(channel))
    print(f"Processing channel: {channel_name}")
    df = add_channel_videos(channel_id, api_key, num_vids=5, df=df)

description = df['video_description'][0]
transcript = df['video_transcript'][0]
clean_description = clean_description(description)
print(clean_description)
clean_transcript = clean_transcript(transcript)
print(clean_transcript)
