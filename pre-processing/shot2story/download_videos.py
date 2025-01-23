import pandas as pd
import os
import argparse
import concurrent.futures
from tqdm import tqdm  # 터미널에 진행률 표시

# yt-dlp version 2025.01.12 (ensure it's installed and updated)

def ytb_save(id, save_fp):
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        id (str): YouTube video ID.
        save_fp (str): File path to save the video.
    """
    os.system(f"yt-dlp -f 'bestvideo[ext=mp4][height<=360]+bestaudio/best[ext=mp4][height<=360]' "
              f"-o '{save_fp}/{id}.mp4' https://www.youtube.com/watch?v={id}")


def main(args):
    """
    Main function to read metadata and download missing videos.
    
    Args:
        args: Command line arguments.
    """
    # Ensure video directory exists
    video_dir = os.path.join(args.data_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    # Read metadata
    df = pd.read_csv(os.path.join(args.data_dir, 'annotations/20k_meta.csv'))
    print(f"Total entries in metadata: {len(df)}")

    # Remove duplicate YouTube IDs
    df = df.drop_duplicates(subset='youtube_id', keep='first')

    # Identify videos to download
    ids_todo = []
    save_fps = []
    for _, row in df.iterrows():
        video_fp = os.path.join(video_dir, f"{row['youtube_id']}.mp4")
        if not os.path.isfile(video_fp):
            ids_todo.append(row['youtube_id'])
            save_fps.append(video_dir)

    total_videos = len(ids_todo)
    print(f"Spawning {total_videos} download jobs...")
    
    # Download videos with progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
        with tqdm(total=total_videos, desc="Downloading videos", unit="video") as pbar:
            futures = []
            for url, fp in zip(ids_todo, save_fps):
                future = executor.submit(ytb_save, url, fp)
                futures.append(future)
            
            # Update progress bar as downloads complete
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Downloader')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of parallel download processes.')
    args = parser.parse_args()

    main(args)