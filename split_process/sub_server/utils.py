from moviepy.video.io.VideoFileClip import VideoFileClip
import os


def split_video(video_path, output_dir, clip_duration=5):
    clips = []
    with VideoFileClip(video_path) as video:
        duration = video.duration
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for start in range(0, int(duration), clip_duration):
            end = min(start + clip_duration, duration)
            clip_filename = f"{base_name}_{start:05d}_{int(end):05d}.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            video.subclipped(start, end).write_videofile(clip_path, codec='libx264', audio_codec='aac', threads=1, preset='ultrafast', logger=None)
            clips.append({
                "video_path": clip_path,
                "start_time": f"{start:.2f}",
                "end_time": f"{end:.2f}"
            })
    return clips


def generate_caption(video_path):
    pass