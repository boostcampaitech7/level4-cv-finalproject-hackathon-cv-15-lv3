import os
import subprocess

def download_videos_and_audio(file_path, video_folder="video", audio_folder="audio"):
    try:
        # Create the folders if they don't exist
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)

        # Read the file with URLs
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
        
        if not urls:
            print("No URLs found in the file.")
            return
        
        # Iterate through URLs and download each video and audio
        for url in urls:
            # Extract video ID from URL
            video_id = url.split("v=")[-1]
            
            print(f"Processing video ID: {video_id}")

            # File names based on video ID
            video_filename = os.path.join(video_folder, f"video_{video_id}.%(ext)s")
            audio_filename = os.path.join(audio_folder, f"audio_{video_id}.%(ext)s")

            # Download video
            print(f"Downloading video: {url}")
            subprocess.run(["yt-dlp", "-o", video_filename, url], check=True)

            # Extract audio only
            print(f"Extracting audio: {url}")
            subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_filename, url], check=True)
        
        print("All downloads and extractions completed.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
input_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/dataset/movie_urls.txt"  # Replace with the name of your URL file
video_folder = "video"  # Folder to store videos
audio_folder = "audio"  # Folder to store audio files
download_videos_and_audio(input_file, video_folder, audio_folder)