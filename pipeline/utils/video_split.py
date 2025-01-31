from abc import ABC, abstractmethod
import cv2
from moviepy import VideoFileClip
from scenedetect import detect, ContentDetector
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os

@contextmanager
def suppress_output():
    """모든 출력을 억제하는 컨텍스트 매니저"""
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class VideoSegmenter(ABC):
    """비디오 세그멘테이션을 위한 기본 클래스"""
    
    @abstractmethod
    def get_segments(self, video_path):
        """비디오 파일을 세그먼트로 나누는 메서드
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            list of tuple: (start_time, end_time) 형태의 세그먼트 리스트
        """
        pass

class FixedDurationSegmenter(VideoSegmenter):
    """고정 길이로 비디오를 나누는 세그멘터"""
    
    def __init__(self, segment_duration=5):
        self.segment_duration = segment_duration
    
    def get_segments(self, video_path):
        with suppress_output():
            with VideoFileClip(video_path) as video:
                duration = video.duration
                segments = []
                start_time = 0
                
                while start_time < duration:
                    end_time = min(start_time + self.segment_duration, duration)
                    if end_time - start_time >= 1:  # 최소 1초 이상인 세그먼트만 포함
                        segments.append((start_time, end_time))
                    start_time = end_time
                
        return segments

class SceneDetectionSegmenter(VideoSegmenter):
    """Scene detection을 사용하여 비디오를 나누는 세그멘터"""
    
    def __init__(self, threshold=30.0, min_scene_len=1):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def get_segments(self, video_path):
        # Scene detection 실행
        scenes = detect(video_path, ContentDetector(
            threshold=self.threshold,
            min_scene_len=self.min_scene_len
        ))
        
        # (start_time, end_time) 형태로 변환
        segments = [(scene[0].get_seconds(), scene[1].get_seconds()) 
                   for scene in scenes]
        
        return segments

class ShotBoundarySegmenter(VideoSegmenter):
    """Shot boundary detection을 사용하여 비디오를 나누는 세그멘터"""
    
    def __init__(self, threshold=30):
        self.threshold = threshold
    
    def get_segments(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        segments = []
        
        prev_frame = None
        start_time = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # 프레임 간 차이 계산
                diff = cv2.absdiff(frame, prev_frame)
                score = diff.mean()
                
                # Shot boundary 감지
                if score > self.threshold:
                    end_time = frame_count / fps
                    if end_time - start_time >= 1:  # 최소 1초 이상
                        segments.append((start_time, end_time))
                    start_time = end_time
            
            prev_frame = frame.copy()
            frame_count += 1
        
        # 마지막 세그먼트 추가
        end_time = frame_count / fps
        if end_time - start_time >= 1:
            segments.append((start_time, end_time))
        
        cap.release()
        return segments

# Factory 패턴을 사용하여 세그멘터 생성
def create_segmenter(method="fixed", **kwargs):
    """세그멘터 생성 함수
    
    Args:
        method (str): 세그멘테이션 방법 ("fixed", "scene", "shot")
        **kwargs: 각 세그멘터의 파라미터
    
    Returns:
        VideoSegmenter: 세그멘터 인스턴스
    """
    segmenters = {
        "fixed": FixedDurationSegmenter,
        "scene": SceneDetectionSegmenter,
        "shot": ShotBoundarySegmenter
    }
    
    if method not in segmenters:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return segmenters[method](**kwargs)