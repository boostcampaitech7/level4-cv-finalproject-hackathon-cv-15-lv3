import json
import re

def extract_numbers(video_path):
    """Extracts numerical components from the video_path for proper sorting."""
    match = re.search(r'video_(\d+)/(\d+)\.mp4', video_path)
    if match:
        video_num = int(match.group(1))  # Extract video_X (X)
        clip_num = int(match.group(2))   # Extract clip number (e.g., 00001.mp4 → 1)
        return (video_num, clip_num)
    return (float('inf'), float('inf'))  # Assign high value if no match (places them at the end)

def sort_json_by_video_path(input_json_path, output_json_path):
    """Sorts JSON data by video_path numerically and saves the sorted output."""

    # Load JSON data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by extracted numerical values from video_path
    sorted_data = sorted(data, key=lambda x: extract_numbers(x.get("video_path", "")))

    # Save sorted JSON data
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Sorting complete! Sorted data saved to: {output_json_path}")

# Example usage
input_json = "/data/ephemeral/home/chan/level4-cv-finalproject-hackathon-cv-15-lv3/embedding/pre-processing/gt_db.json"  # Adjust path if necessary
output_json = "sorted_gt_db_v2.json"

sort_json_by_video_path(input_json, output_json)