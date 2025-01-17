import os
import subprocess
import argparse


def download_videos_audios(file_path, video_folder, audio_folder):
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

# Todo: Add a main function to run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos and audio from YouTube URLs.")
    parser.add_argument("--category_name", type=str, help="The category name for the videos and audios.")
    args = parser.parse_args()

    category_name = args.category_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, f"category-{category_name}-info/{category_name}_urls.txt") # Todo: if you want to use a different file, change this path
    video_folder = os.path.join(script_dir, f"data/{category_name}/videos")
    audio_folder = os.path.join(script_dir, f"data/{category_name}/audios")
    
    download_videos_audios(input_file, video_folder, audio_folder)

