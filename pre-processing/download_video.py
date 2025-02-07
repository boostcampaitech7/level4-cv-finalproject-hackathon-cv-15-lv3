import os
import subprocess
import argparse

def download_videos(file_path, video_folder):
    try:
        # Create folder if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)

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

            # Define output filename
            video_filename = os.path.join(video_folder, f"{video_id}.%(ext)s")

            try:
                # Download video and force MP4 format
                print(f"Downloading video: {url}")
                subprocess.run(
                    ["yt-dlp", "--ignore-errors", "--geo-bypass", "--merge-output-format", "mp4", "-o", video_filename, url],
                    check=True
                )

            except subprocess.CalledProcessError as e:
                print(f"Error processing video ID {video_id}: {e}")

        print("All video downloads completed.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Todo: Add a main function to run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos from YouTube URLs.")
    parser.add_argument("--category_name", type=str, help="The category name for the videos.")
    args = parser.parse_args()

    category_name = args.category_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = "/Users/imch/workspace/level4-cv-finalproject-hackathon-cv-15-lv3/pre-processing/YouTube-8M/etc_urls.txt" # Todo: if you want to use a different file, change this path
    video_folder = os.path.join(script_dir, f"data/etc_videos")
    
    download_videos(input_file, video_folder)
