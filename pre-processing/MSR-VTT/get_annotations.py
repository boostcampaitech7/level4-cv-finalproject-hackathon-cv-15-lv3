import os
import json
import logging
import re
import shutil
import csv
import argparse
import cv2

logging.basicConfig(filename="process.log", level=logging.INFO)

def sorted_num(file_list):
    """Sort filenames numerically based on numbers in the filename."""
    def extract_number(filename):
        match = re.search(r'\d+', filename)  
        return int(match.group()) if match else float('inf')
    return sorted(file_list, key=extract_number)

def save_json(annotation_path, file_name, data):
    """Save data to a JSON file."""
    os.makedirs(annotation_path, exist_ok=True)
    json_path = os.path.join(annotation_path, f"{file_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def save_csv(annotation_path, file_name, data):
    """Save data to a CSV file."""
    os.makedirs(annotation_path, exist_ok=True)
    csv_path = os.path.join(annotation_path, f"{file_name}.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def copy_file(src_path, dest_path):
    """Copy a file from src_path to dest_path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and audios into a structured dataset.")
    parser.add_argument("--num_videos", type=int, default=-1, help="Number of videos to process. Default is -1 (process all).")

    args = parser.parse_args()

    root_video_path = "./data/MSR-VTT/videos"
    root_audio_path = "./data/MSR-VTT/audios"
    output_video_dir = "./data/MSR-VTT/raw_videos"
    output_audio_dir = "./data/MSR-VTT/raw_audios"
    annotation_path = "./data/MSR-VTT/annotations"

    video_files = sorted_num(os.listdir(root_video_path))

    if args.num_videos != -1:
        if args.num_videos < -1:
            raise ValueError("--num_videos must be -1 (for all videos) or a positive integer.")
        video_files = video_files[:args.num_videos]

    dataset = []

    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        input_video_path = os.path.join(root_video_path, video_file)
        input_audio_path = os.path.join(root_audio_path, f"{video_id}.wav")

        # Video output path
        video_output_path = os.path.join(output_video_dir, video_id, video_file)
        copy_file(input_video_path, video_output_path)

        # Audio output path
        audio_output_path = os.path.join(output_audio_dir, video_id, f"{video_id}.wav")
        if os.path.exists(input_audio_path):
            copy_file(input_audio_path, audio_output_path)
        else:
            logging.warning(f"No audio file found for video ID: {video_id}")

        # Collect metadata
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0
        cap.release()

        dataset.append({
            "video_path": f"{video_id}/{video_file}",
            "audio_path": f"{video_id}/{video_id}.wav" if os.path.exists(input_audio_path) else "",
            "video_id": video_id,
            "clip_id": video_id,  # Use video_id as clip_id
            "fps": fps if fps else 0,
            "start_time": 0,
            "end_time": round(duration, 2),
            "start_frame": 0,
            "end_frame": int(frame_count) if frame_count else 0,
            "duration_seconds": round(duration, 2),  # Add duration
            "caption": ""  # Add empty caption
        })

    # Save dataset annotations
    save_json(annotation_path, "MSRVTT_annotations", dataset)
    save_csv(annotation_path, "MSRVTT_annotations", dataset)

    print(f"Processing complete. Dataset annotations saved to {annotation_path}.")