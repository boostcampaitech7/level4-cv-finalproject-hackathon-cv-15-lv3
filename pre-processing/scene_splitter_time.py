import os
import cv2
import json
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def save_scene(video_path, start_time, end_time, output_folder, scene_number):
    """Saves a video segment (scene) to the specified folder."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = os.path.join(output_folder, f"scene_{scene_number:03d}.mp4")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def process_video(input_video, output_folder, min_scene_length=5.0, max_scene_length=15.0):
    """Processes a video, saves scenes longer than min_scene_length to output_folder, and logs scene details to JSON."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_manager = VideoManager([input_video])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    # scene_manager.add_detector(HashDetector())

    video_manager.set_downscale_factor()  # Downscale for faster processing
    video_manager.start()

    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    print(f"Detected {len(scene_list)} scenes.")

    scene_number = 1
    scene_data = []

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time

        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i + 1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),
        ))

        if duration >= min_scene_length and duration <= max_scene_length:
            print(f"Saving scene {scene_number}: {scene[0].get_timecode()} to {scene[1].get_timecode()} ({duration:.2f}s)")
            save_scene(input_video, start_time, end_time, output_folder, scene_number)

            scene_data.append({
                "scene_number": scene_number,
                "start_time": scene[0].get_timecode(),
                "end_time": scene[1].get_timecode(),
                "start_frame": scene[0].get_frames(),
                "end_frame": scene[1].get_frames(),
                "duration_seconds": duration
            })
            scene_number += 1

    # Save scene data to JSON
    json_path = os.path.join(output_folder, "scene_data.json")
    with open(json_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)

    video_manager.release()

if __name__ == "__main__":
    # root_path = "./data/Youtube-8M-Movieclips/videos"
    # video_files = os.listdir(root_path)
    # print(len(video_files))
    # assert False

    input_video_path = "./data/Youtube-8M-Movieclips/videos/7QZB_cbjdM8.mp4"  # Replace with your video path
    output_directory = "./data/Youtube-8M-Movieclips/outputs"   # Replace with your desired output folder
    process_video(input_video_path, output_directory)
