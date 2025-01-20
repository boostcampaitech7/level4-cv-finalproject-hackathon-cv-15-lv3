import os
import cv2
import json
import logging
import shutil
import csv
import argparse
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from scenedetect.scene_detector import FlashFilter

logging.basicConfig(filename="process.log", level=logging.INFO)

def filter_long_scenes(scene_list, max_frames):
    """Filter out scenes longer than max_frames."""
    return [(start, end) for start, end in scene_list if (end.get_frames() - start.get_frames()) <= max_frames]

def save_json(annotation_path, file_name, scene_data):
    """Save scene data as a single JSON file."""
    os.makedirs(annotation_path, exist_ok=True)
    json_path = os.path.join(annotation_path, f"{file_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)

def save_csv(annotation_path, file_name, scene_data):
    """Save scene data as a single CSV file."""
    os.makedirs(annotation_path, exist_ok=True)
    csv_path = os.path.join(annotation_path, f"{file_name}.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=scene_data[0].keys())
        writer.writeheader()
        writer.writerows(scene_data)

def split_audio_clip(audio_path, output_audio_path, start_time, end_time):
    """Split audio using ffmpeg based on start and end times."""
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-ss", f"{start_time:.2f}",
        "-to", f"{end_time:.2f}",
        "-c", "copy",
        output_audio_path,
        "-y"
    ]
    os.system(" ".join(cmd))

def create_clip_videos(input_video_path, output_dir_path, min_scene_len_seconds, max_scene_len_seconds):
    """Create clip videos and return scene list."""
    os.makedirs(output_dir_path, exist_ok=True)

    # Get FPS
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        logging.error(f"Failed to retrieve FPS for {input_video_path}")
        return []
    cap.release()

    min_scene_len_frames = int(min_scene_len_seconds * fps)
    max_scene_len_frames = int(max_scene_len_seconds * fps)

    scene_list = detect(input_video_path, ContentDetector(min_scene_len=min_scene_len_frames, filter_mode=FlashFilter.Mode.MERGE))
    return filter_long_scenes(scene_list, max_scene_len_frames), fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos and audio from YouTube URLs.")
    parser.add_argument("--category_name", type=str, help="The category name for the videos and audios.")
    parser.add_argument("--num_videos", type=int, help="The number of videos to process. Default is -1 (process all videos).",default=-1)
    parser.add_argument("--min_length", type=int, help="The number of videos to process. Default is -1 (process all videos).",default=3)
    parser.add_argument("--max_length", type=int, help="The number of videos to process. Default is -1 (process all videos).",default=15)

    args = parser.parse_args()

    category_name = args.category_name
    min_length = args.min_length
    max_length = args.max_length

    root_video_path = f"./data/YouTube-8M-{category_name}/videos"
    root_audio_path = f"./data/YouTube-8M-{category_name}/audios"
    annotation_path = f"./data/YouTube-8M-{category_name}/annotations"
    video_files = sorted(os.listdir(root_video_path))

    if args.num_videos != -1:
        if args.num_videos < -1:
            raise ValueError("--num_videos must be -1 (for all videos) or a positive integer.")
        video_files = video_files[:args.num_videos]

    scene_number = 1
    scene_data = []

    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        logging.info(f"Processing video ID: {video_id}")

        input_video_path = os.path.join(root_video_path, video_file)
        output_video_dir = os.path.join(f"./data/YouTube-8M-{category_name}/clip_videos/", video_id)
        output_audio_dir = os.path.join(f"./data/YouTube-8M-{category_name}/clip_audios/", video_id)

        # Get corresponding audio file
        audio_path = os.path.join(root_audio_path, f"{video_id}.wav")
        if not os.path.exists(audio_path):
            logging.warning(f"No audio file found for video ID: {video_id}")
            continue

        # Detect scenes
        scene_list, fps = create_clip_videos(input_video_path, output_video_dir, min_scene_len_seconds=min_length, max_scene_len_seconds=max_length)

        # Split the video into clips
        if scene_list:
            split_video_ffmpeg(input_video_path, scene_list, output_dir=output_video_dir)

            # Process each scene
            for i, (start, end) in enumerate(scene_list):
                clip_filename = f"{scene_number:05}.mp4"
                audio_clip_filename = f"{scene_number:05}.wav"
                old_filename = f"{video_id}-Scene-{i+1:03}.mp4"
                old_path = os.path.join(output_video_dir, old_filename)
                new_video_path = os.path.join(output_video_dir, clip_filename)

                # Rename video clip
                if os.path.exists(old_path):
                    shutil.move(old_path, new_video_path)
                    print(f"Renamed {old_path} to {new_video_path}")

                # Generate corresponding audio clip
                audio_output_path = os.path.join(output_audio_dir, audio_clip_filename)
                split_audio_clip(audio_path, audio_output_path, start.get_seconds(), end.get_seconds())

                # Append to JSON annotation
                scene_data.append({
                    "video_path": f"{video_id}/{clip_filename}",
                    "audio_path": f"{video_id}/{audio_clip_filename}",
                    "video_id": video_id,
                    "clip_id": f"{scene_number:05}",
                    "fps": fps,
                    "start_time": start.get_timecode(),
                    "end_time": end.get_timecode(),
                    "start_frame": start.get_frames(),
                    "end_frame": end.get_frames(),
                    "duration_seconds": round(end.get_seconds() - start.get_seconds(), 2),
                    "caption": ""  
                })
                scene_number += 1  # Increment global scene number
        else:
            logging.info(f"No scenes found for video ID: {video_id}")
            continue

    # Save global JSON and CSV
    save_json(annotation_path, f"{category_name}_annotations", scene_data)
    save_csv(annotation_path, f"{category_name}_annotations", scene_data)