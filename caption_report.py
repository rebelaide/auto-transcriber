import os
import sys
# Import the transcriber function from your caption.py
from caption import transcribe_video

def process_bad_videos(video_list):
    """
    Takes a list of dictionaries with video details:
    [{'title': 'Lecture 1', 'url': 'https://youtu.be/...', 'status': 'MISSING'}]
    """
    print(f"🔄 Starting auto-captioning for {len(video_list)} videos...")
    
    for video in video_list:
        url = video.get('url')
        title = video.get('title', 'Unknown Video')
        
        print(f"\n🎥 Processing: {title}")
        print(f"🔗 URL: {url}")
        
        # Run the transcriber
        # We use 'medium' model for balance of speed/accuracy on Colab
        try:
            transcribe_video(
                input_path=url,
                model_size="medium",
                initial_prompt="Hello, welcome. This is a sentence with proper punctuation, capitalization, and grammar."
            )
            print(f"✅ Successfully captioned: {title}")
        except Exception as e:
            print(f"❌ Failed to caption {title}: {e}")

    print("\n✨ All captioning tasks completed.")