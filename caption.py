from __future__ import print_function
import re
import requests
import os
import shutil
from bs4 import BeautifulSoup
from google.colab import userdata, drive
import gspread
from gspread_dataframe import set_with_dataframe
import pandas as pd

# --- IMPORT TRANSCRIBER ---
try:
    from caption import transcribe_video
except ImportError:
    print("⚠️ Warning: caption.py not found. Auto-transcription disabled.")
    def transcribe_video(*args, **kwargs): return []

# --------------------------------------------------------------
# 1️⃣ CONSTANTS
# --------------------------------------------------------------
CANVAS_API_URL   = userdata.get('CANVAS_API_URL')
CANVAS_API_KEY   = userdata.get('CANVAS_API_KEY')
YOUTUBE_API_KEY  = userdata.get('YOUTUBE_API_KEY')

YT_CAPTION_URL = "https://www.googleapis.com/youtube/v3/captions"
YT_VIDEO_URL   = "https://www.googleapis.com/youtube/v3/videos"
YT_PATTERN = (
    r'(?:https?://)?(?:[0-9A-Z-]+\.)?(?:youtube|youtu|youtube-nocookie)\.'
    r'(?:com|be)/(?:watch\?v=|watch\?.+&v=|embed/|v/|.+\?v=)?([^&=\n%\?]{11})'
)
LIB_MEDIA_URLS = ["fod.infobase.com", "search.alexanderstreet.com", "kanopystreaming-com", "hosted.panopto.com"]

# ----------------------------------------------------------------------
# 2️⃣ HELPER FUNCTIONS
# ----------------------------------------------------------------------

def get_youtube_id(url):
    match = re.search(YT_PATTERN, url, re.IGNORECASE)
    return match.group(1) if match else None

def check_youtube_captions(video_id):
    if not YOUTUBE_API_KEY: return ("API Key Missing", "00:00")
    
    params_cap = {"part": "snippet", "videoId": video_id, "key": YOUTUBE_API_KEY}
    params_vid = {"part": "contentDetails", "id": video_id, "key": YOUTUBE_API_KEY}

    try:
        # Check Captions
        resp_cap = requests.get(YT_CAPTION_URL, params=params_cap)
        data_cap = resp_cap.json()
        has_manual = any(i["snippet"]["trackKind"] == "standard" for i in data_cap.get("items", []))
        has_auto = any(i["snippet"]["trackKind"] == "ASR" for i in data_cap.get("items", []))
        
        status = "Professional Captions" if has_manual else "Automatic Captions" if has_auto else "No Captions"

        # Check Duration
        resp_vid = requests.get(YT_VIDEO_URL, params=params_vid)
        data_vid = resp_vid.json()
        duration_str = "00:00"
        if "items" in data_vid and len(data_vid["items"]) > 0:
            iso_dur = data_vid["items"][0]["contentDetails"]["duration"]
            match = re.search(r'PT(\d+H)?(\d+M)?(\d+S)?', iso_dur)
            if match:
                h = int((match.group(1) or "0H")[:-1])
                m = int((match.group(2) or "0M")[:-1])
                s = int((match.group(3) or "0S")[:-1])
                duration_str = f"{h:02}:{m:02}:{s:02}" if h > 0 else f"{m:02}:{s:02}"

        return status, duration_str
    except Exception as e:
        return (f"Error: {str(e)}", "00:00")

def process_page_content(html_content):
    found_media = []
    if not html_content: return found_media
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Check iframes
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        yt_id = get_youtube_id(src)
        if yt_id:
            found_media.append({"type": "YouTube", "id": yt_id, "url": f"https://www.youtube.com/watch?v={yt_id}"})
            continue
        for lib in LIB_MEDIA_URLS:
            if lib in src:
                found_media.append({"type": "Library Media", "url": src, "id": None})
                break
                
    # Check links
    for link in soup.find_all("a"):
        href = link.get("href", "")
        yt_id = get_youtube_id(href)
        if yt_id:
            found_media.append({"type": "YouTube", "id": yt_id, "url": f"https://www.youtube.com/watch?v={yt_id}"})
            
    return found_media

def setup_drive_folder(folder_name):
    """Mounts Drive and creates a folder for transcripts."""
    print("📂 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    base_path = "/content/drive/My Drive"
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"🆕 Created Drive Folder: {folder_path}")
        except FileExistsError:
            print(f"📂 Using existing Drive Folder: {folder_path}")
    else:
        print(f"📂 Using existing Drive Folder: {folder_path}")
        
    return folder_path

# ----------------------------------------------------------------------
# 3️⃣ MAIN LOGIC
# ----------------------------------------------------------------------

def run_caption_report(course_input, generate_vtt=False):
    # Extract Course ID
    try:
        course_id = re.search(r'courses/(\d+)', str(course_input)).group(1) if "http" in str(course_input) else str(course_input)
    except:
        print("❌ Invalid Course ID/URL"); return

    # Connect to APIs
    from canvasapi import Canvas
    canvas = Canvas(CANVAS_API_URL, CANVAS_API_KEY)
    try:
        course = canvas.get_course(course_id)
        print(f"✅ Found Course: {course.name}")
    except Exception as e:
        print(f"❌ Canvas Error: {e}"); return

    # Google Auth
    try:
        from google.colab import auth
        auth.authenticate_user()
        import google.auth
        creds, _ = google.auth.default()
        gc = gspread.authorize(creds)
    except Exception as e:
        print(f"⚠️ Auth Warning: {e}")

    # Setup Drive Folder
    drive_folder_name = f"{course.name} Transcripts"
    drive_path = setup_drive_folder(drive_folder_name)
    
    # Scan Content
    print("📦 Scanning Course Content...")
    report_data = []
    
    # Helper to scan items
    def scan_items(items, location_prefix, get_body_fn):
        for item in items:
            body = get_body_fn(item)
            for m in process_page_content(body):
                m['location'] = f"{location_prefix}: {item.title if hasattr(item, 'title') else item.name}"
                report_data.append(m)

    scan_items(course.get_pages(), "Page", lambda x: course.get_page(x.url).body)
    scan_items(course.get_assignments(), "Assignment", lambda x: x.description)
    scan_items(course.get_discussion_topics(), "Discussion", lambda x: x.message)

    print(f"found {len(report_data)} media items. Checking status...")

    # Process Videos
    final_rows = []
    
    for item in report_data:
        status = "Unknown"
        duration = "00:00"
        transcript_created = "No" # New Column Value
        
        if item['type'] == 'YouTube':
            status, duration = check_youtube_captions(item['id'])
            
            # --- AUTO-TRANSCRIPTION TRIGGER ---
            if status in ["No Captions", "Automatic Captions", "Non-English", "Unknown"]:
                print(f"\n⚠️  Fixing: {item['url']} ({status})")
                try:
                    # Run transcriber
                    output_files = transcribe_video(
                        input_path=item['url'],
                        model_size="medium",
                        initial_prompt="Hello, welcome. This is a sentence with proper punctuation.",
                        generate_vtt=generate_vtt # Pass user preference
                    )
                    
                    if output_files:
                        transcript_created = "Yes"
                        # Move files to Drive
                        for f_path in output_files:
                            if os.path.exists(f_path):
                                dest = os.path.join(drive_path, os.path.basename(f_path))
                                shutil.move(f_path, dest)
                                print(f"💾 Saved to Drive: {os.path.basename(f_path)}")
                        status = f"{status} (AUTO-FIXED)"
                    else:
                        transcript_created = "Failed (Download Error)"
                        
                except Exception as e:
                    print(f"❌ Transcription failed: {e}")
                    transcript_created = "Error"
            # ----------------------------------

        elif item['type'] == 'Library Media':
            status = "Check Manually (Library)"
            
        final_rows.append({
            "Media Type": item['type'],
            "URL": item['url'],
            "Location": item['location'],
            "Caption Status": status,
            "Duration": duration,
            "Auto Transcript Created": transcript_created
        })

    # Save Report
    df = pd.DataFrame(final_rows)
    sheet_title = f"VAST Report - {course.name}"
    
    try:
        try: sh = gc.open(sheet_title)
        except: sh = gc.create(sheet_title)
            
        ws = sh.sheet1
        ws.clear()
        set_with_dataframe(ws, df)
        
        # Add Drive Info at bottom
        last_row = len(df) + 4
        ws.update_cell(last_row, 1, "Transcripts Saved To:")
        ws.update_cell(last_row, 2, f"/content/drive/My Drive/{drive_folder_name}")
        
        # Add helper link text
        ws.update_cell(last_row + 1, 1, "Access via Google Drive using folder name above.")
        
        print(f"✅ Report saved: {sh.url}")
        
    except Exception as e:
        print(f"❌ Sheet Save Failed: {e}")
        df.to_csv(f"vast_report.csv", index=False)
        from google.colab import files; files.download("vast_report.csv")

    return df