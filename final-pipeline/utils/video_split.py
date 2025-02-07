from abc import ABC, abstractmethod
import cv2
from moviepy import VideoFileClip
from scenedetect import detect, ContentDetector, AdaptiveDetector
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

# class SceneDetectionSegmenter(VideoSegmenter):
#     """Scene detection을 사용하여 비디오를 나누는 세그멘터"""
    
#     def __init__(self, adaptive_threshold=3.0, min_scene_len=30):
#         """
#         Args:
#             adaptive_threshold (float): 장면 변화 감지 민감도 (낮을수록 더 민감)
#                                      - 2.0: 더 민감한 감지
#                                      - 3.0: 기본값, 일반적인 용도
#                                      - 4.0: 덜 민감한 감지
#             min_scene_len (int): 최소 장면 길이 (프레임 단위)
#                                - 15: 기본값 (~0.5초 @ 30fps)
#                                - 10: 짧은 장면 허용
#                                - 30: 긴 장면 보장
#         """
#         self.adaptive_threshold = adaptive_threshold
#         self.min_scene_len = min_scene_len
    
#     def get_segments(self, video_path):
#         try:
#             # Scene detection 실행
#             scenes = detect(video_path, AdaptiveDetector(
#                 adaptive_threshold=self.adaptive_threshold,
#                 min_scene_len=self.min_scene_len
#             ))
            
#             # (start_time, end_time) 형태로 변환
#             segments = [(scene[0].get_seconds(), scene[1].get_seconds()) 
#                        for scene in scenes]
            
#             # 세그먼트가 없는 경우 처리
#             if not segments:
#                 print(f"⚠️ No scenes detected in {video_path}, using entire video")
#                 with VideoFileClip(video_path) as video:
#                     segments = [(0, video.duration)]
            
#             return segments
            
#         except Exception as e:
#             print(f"🚨 Error during scene detection for {video_path}: {str(e)}")
#             # 오류 발생 시 전체 비디오를 하나의 세그먼트로
#             with VideoFileClip(video_path) as video:
#                 return [(0, video.duration)]

class SceneDetectionSegmenter(VideoSegmenter):
    """Scene detection을 사용하여 비디오를 나누는 세그멘터"""
    
    def __init__(self, threshold=27.0, min_scene_len=30):
        """
        Args:
            threshold (float): 장면 변화 감지를 위한 임계값
                             - 낮을수록 더 민감하게 감지
                             - 27.0: 기본값
                             - 20.0: 더 민감한 감지
                             - 35.0: 덜 민감한 감지
            min_scene_len (int): 최소 장면 길이 (프레임 단위)
                               - 15: 기본값 (~0.5초 @ 30fps)
                               - 10: 짧은 장면 허용
                               - 30: 긴 장면 보장
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def get_segments(self, video_path):
        try:
            # Scene detection 실행
            scenes = detect(video_path, ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len
            ))
            
            # (start_time, end_time) 형태로 변환
            segments = [(scene[0].get_seconds(), scene[1].get_seconds()) 
                       for scene in scenes]
            
            # 세그먼트가 없는 경우 처리
            if not segments:
                print(f"⚠️ No scenes detected in {video_path}, using entire video")
                with VideoFileClip(video_path) as video:
                    segments = [(0, video.duration)]
            
            return segments
            
        except Exception as e:
            print(f"🚨 Error during scene detection for {video_path}: {str(e)}")
            # 오류 발생 시 전체 비디오를 하나의 세그먼트로
            with VideoFileClip(video_path) as video:
                return [(0, video.duration)]

class ShotBoundarySegmenter(VideoSegmenter):
    """Shot boundary detection을 사용하여 비디오를 나누는 세그멘터"""
    
    def __init__(self, threshold=30, min_segment_length=1):
        self.threshold = threshold
        self.min_segment_length = min_segment_length
    
    def get_segments(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segments = []
        
        if not cap.isOpened():
            print(f"🚨 Error: Could not open video {video_path}")
            return segments
        
        try:
            prev_frame = None
            start_time = 0
            frame_count = 0
            
            while frame_count < total_frames:  # total_frames로 체크
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 그레이스케일로 변환하여 계산 효율성 향상
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 프레임 간 차이 계산 (그레이스케일)
                    diff = cv2.absdiff(frame_gray, prev_frame)
                    score = diff.mean()
                    
                    # Shot boundary 감지
                    if score > self.threshold:
                        end_time = frame_count / fps
                        if end_time - start_time >= self.min_segment_length:
                            segments.append((start_time, end_time))
                        start_time = end_time
                
                prev_frame = frame_gray  # 그레이스케일 이미지 저장
                frame_count += 1
            
            # 마지막 세그먼트 추가
            end_time = frame_count / fps
            if end_time - start_time >= self.min_segment_length:
                segments.append((start_time, end_time))
            
        except Exception as e:
            print(f"🚨 Error processing video {video_path}: {str(e)}")
        
        finally:
            cap.release()
        
        # 빈 세그먼트 리스트인 경우 전체 비디오를 하나의 세그먼트로
        if not segments:
            print(f"⚠️ No segments detected for {video_path}, using entire video")
            segments = [(0, total_frames / fps)]
        
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