import os
import subprocess

def download_videos_and_audio(file_path, video_folder="video", audio_folder="audio"):
    try:
        # Create folders if they don't exist
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)

        # Read URLs from the input file
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]

        if not urls:
            print("No URLs found in the file.")
            return

        for url in urls:
            # Extract video ID
            video_id = url.split("v=")[-1]
            print(f"Processing video ID: {video_id}")

            # Define output filenames
            video_filename = os.path.join(video_folder, f"{video_id}.%(ext)s")
            audio_filename = os.path.join(audio_folder, f"{video_id}.%(ext)s")

            try:
                # Download video and force MP4 format
                print(f"Downloading video: {url}")
                subprocess.run(
                    ["yt-dlp", "--ignore-errors", "--geo-bypass", "--merge-output-format", "mp4", "-o", video_filename, url],
                    check=True
                )

                # Extract audio as WAV
                print(f"Extracting audio: {url}")
                subprocess.run(
                    ["yt-dlp", "--ignore-errors", "-x", "--audio-format", "wav", "-o", audio_filename, url],
                    check=True
                )

            except subprocess.CalledProcessError as e:
                print(f"Error processing video ID {video_id}: {e}")

        print("All downloads and extractions completed.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
input_file = "/Users/imch/workspace/level4-cv-finalproject-hackathon-cv-15-lv3/pre-processing/Movieclips_urls.txt"  # Replace with your URL file path
video_folder = "video"
audio_folder = "audio"
download_videos_and_audio(input_file, video_folder, audio_folder)