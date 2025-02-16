import subprocess
import os
from scenedetect import detect, AdaptiveDetector
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def split_video(video_path, output_dir):
    """Splits a video into scenes and saves each scene as a separate video file."""
    try:
        # Extract base name from the video path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nStarting video split: {video_path}")
        
        # Detect scenes in the video
        scene_list = detect(video_path, AdaptiveDetector())
        if not scene_list:
            print(f"No scenes detected: {video_path}")
            return False
        
        # Create directory to save clips
        clips_dir = os.path.join(output_dir, f"{base_name}")
        os.makedirs(clips_dir, exist_ok=True)
        
        # Split video into scenes
        for i, (start_time, end_time) in enumerate(scene_list):
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()
            output_file = os.path.join(clips_dir, f"clip_{i+1:03d}.mp4")
            
            # Construct ffmpeg command for video ㅁre-encoding
            command = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_sec),
                '-to', str(end_sec),
                '-c:v', 'mpeg4',  # Use mpeg4 codec for video
                '-c:a', 'aac',    # Use aac codec for audio
                '-strict', 'experimental',
                '-b:a', '192k',
                '-y',
                output_file
            ]
            
            # Execute the command 
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error splitting scene {i+1}: {result.stderr}")
                return False
            else:
                print(f"Scene {i+1} saved as {output_file}")
        
        print(f"Video split complete: {len(scene_list)} clips created")
        return True
    
    except Exception as e:
        print(f"Failed to split video - file: {video_path}")
        print(f"Error message: {str(e)}")
        return False

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        '-y',  # Overwrite output file if it exists
        audio_path
    ]
    subprocess.run(command, capture_output=True, text=True)

def detect_audio_scenes(audio_path, min_silence_len=1000, silence_thresh=-40):
    """Detects scenes based on audio changes."""
    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [(start / 1000, end / 1000) for start, end in nonsilent_ranges]

def split_video_by_audio(video_path, output_dir):
    """Splits a video into scenes based on audio changes."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nStarting video split: {video_path}")
        
        # Extract audio from video
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        extract_audio(video_path, audio_path)
        
        # Detect scenes based on audio
        scene_list = detect_audio_scenes(audio_path)
        if not scene_list:
            print(f"No audio-based scenes detected: {video_path}")
            return False
        
        # Create directory to save clips
        clips_dir = os.path.join(output_dir, f"{base_name}_clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        # Split video into scenes
        for i, (start_sec, end_sec) in enumerate(scene_list):
            output_file = os.path.join(clips_dir, f"clip_{i+1:03d}.mp4")
            
            command = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_sec),
                '-to', str(end_sec),
                '-c:v', 'mpeg4',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-b:a', '192k',
                '-y',
                output_file
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error splitting scene {i+1}: {result.stderr}")
                return False
            else:
                print(f"Scene {i+1} saved as {output_file}")
        
        print(f"Video split complete: {len(scene_list)} clips created")
        return True
    
    except Exception as e:
        print(f"Failed to split video - file: {video_path}")
        print(f"Error message: {str(e)}")
        return False

def process_videos(videos_path, output_dir):
    """Processes a list of videos, splitting each into scenes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success_count = 0
    if split_video_by_audio(videos_path, output_dir):
        success_count += 1
    
    print(f"Video processing complete! Success: {success_count}")

def main():
    videos_path = "../videos/Home Alone (1990) - Kevin Escapes Scene (5⧸5) ｜ Movieclips.mp4"
    output_dir = "../videos/clips"
    process_videos(videos_path, output_dir)

if __name__ == "__main__":
    main() 