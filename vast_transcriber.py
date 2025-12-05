import os
import whisper
import yt_dlp
import time
import threading
import sys
import re

# --- HELPER FUNCTIONS ---

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def show_progress_timer(stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        sys.stdout.write(f"\r⏳ Transcribing... {mins:02}:{secs:02}")
        sys.stdout.flush()
        time.sleep(1)
    print()

def download_youtube_audio(url, audio_quality='192'):
    print(f"--- Downloading: {url} ---")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': audio_quality,
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            final_filename = os.path.splitext(filename)[0] + ".mp3"
            return final_filename
    except Exception as e:
        print(f"❌ Download Error: {e}")
        return None

def build_dcmp_captions(word_segments, max_chars=32, max_lines=2):
    if not word_segments: return []
    captions = []
    current_lines = [""]
    current_start = word_segments[0]['start']
    current_end = word_segments[0]['end']
    
    force_new_block = False
    force_new_line = False

    for i, word_obj in enumerate(word_segments):
        word = word_obj['word'].strip()
        start = word_obj['start']
        end = word_obj['end']
        
        if force_new_block:
            captions.append({"start": current_start, "end": current_end, "lines": current_lines})
            current_lines = [""]
            current_start = start
            current_end = end
            force_new_block = False
            force_new_line = False 
        elif force_new_line:
            if len(current_lines) < max_lines: current_lines.append("")
            else:
                captions.append({"start": current_start, "end": current_end, "lines": current_lines})
                current_lines = [""]
                current_start = start
            force_new_line = False

        line_idx = len(current_lines) - 1
        sep = " " if len(current_lines[line_idx]) > 0 else ""
        if len(current_lines[line_idx]) + len(sep) + len(word) <= max_chars:
            current_lines[line_idx] += sep + word
            current_end = end 
        else:
            if len(current_lines) < max_lines:
                current_lines.append(word)
                current_end = end
            else:
                captions.append({"start": current_start, "end": current_end, "lines": current_lines})
                current_lines = [word]
                current_start = start
                current_end = end

        if len(current_lines[-1]) > 15:
            if word.endswith(('.', '?', '!')): force_new_block = True
            elif word.endswith((',', ';', ':')): force_new_line = True

    captions.append({"start": current_start, "end": current_end, "lines": current_lines})
    
    # Orphan Fix
    if len(captions) > 1:
        last_text = " ".join(captions[-1]['lines'])
        if len(last_text) < 15 and (len(captions[-2]['lines'][-1]) + len(last_text) < (max_chars + 15)):
            captions[-2]['lines'][-1] += " " + last_text
            captions[-2]['end'] = captions[-1]['end']
            captions.pop()
    return captions

def write_pro_vtt(captions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\nKind: captions\nLanguage: en\n\n")
        for caption in captions:
            f.write(f"{format_timestamp(caption['start'])} --> {format_timestamp(caption['end'])}\n")
            f.write(f"{chr(10).join(caption['lines'])}\n\n")

# --- MAIN TRANSCRIBE FUNCTION ---

def transcribe_video(input_path, model_size="base", audio_quality='192', max_chars=32, max_lines=2, initial_prompt=None, generate_vtt=False):
    """
    Returns list of generated file paths (e.g. ['video.txt', 'video.vtt'])
    """
    downloaded_file = None
    generated_files = []

    # 1. Download
    if input_path.startswith("http"):
        downloaded_file = download_youtube_audio(input_path, audio_quality)
        if not downloaded_file: return []
        local_file_path = downloaded_file
    else:
        local_file_path = input_path
        if not os.path.exists(local_file_path): return []

    base_name = os.path.splitext(local_file_path)[0]

    try:
        # 2. Load Model
        # (Optimized: In Colab, model usually stays loaded in memory if we re-call this)
        print(f"--- Loading Model ({model_size}) ---")
        model = whisper.load_model(model_size)
        
        if not initial_prompt:
            initial_prompt = "Hello, welcome. This is a sentence with proper punctuation."

        # 3. Transcribe
        print("--- Transcribing ---")
        stop_timer = threading.Event()
        timer_thread = threading.Thread(target=show_progress_timer, args=(stop_timer,))
        timer_thread.start()

        try:
            result = model.transcribe(
                local_file_path, 
                fp16=False, 
                word_timestamps=True, 
                initial_prompt=initial_prompt,
                condition_on_previous_text=False
            )
        finally:
            stop_timer.set()
            timer_thread.join()
        
        # 4. Generate Outputs
        
        # ALWAYS TXT
        txt_output = f"{base_name}.txt"
        with open(txt_output, "w", encoding="utf-8") as f:
            f.write(result['text'].strip())
        generated_files.append(txt_output)
        print(f"✅ Transcript created: {os.path.basename(txt_output)}")

        # OPTIONAL VTT
        if generate_vtt:
            vtt_output = f"{base_name}.vtt"
            all_words = []
            for segment in result["segments"]:
                if "words" in segment: all_words.extend(segment["words"])
            
            pro_captions = build_dcmp_captions(all_words, max_chars, max_lines)
            write_pro_vtt(pro_captions, vtt_output)
            generated_files.append(vtt_output)
            print(f"✅ VTT created: {os.path.basename(vtt_output)}")

        return generated_files

    except Exception as e:
        print(f"❌ Error: {e}")
        return []
        
    finally:
        if downloaded_file and os.path.exists(downloaded_file):
            try: os.remove(downloaded_file)
            except: pass