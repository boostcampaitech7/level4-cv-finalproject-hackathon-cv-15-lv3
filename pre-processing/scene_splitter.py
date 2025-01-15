from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import os

def split_video(video_path, output_dir):
    try:
        # 파일명에서 확장자 제거
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n비디오 분할 시작: {video_path}")
        
        # 장면 감지
        scene_list = detect(video_path, AdaptiveDetector())
        
        if not scene_list:
            print(f"장면을 감지할 수 없습니다: {video_path}")
            return False
            
        # 클립 저장할 디렉토리 생성
        clips_dir = os.path.join(output_dir, f"{base_name}_clips")
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)
            
        # 비디오 분할
        split_video_ffmpeg(
            video_path, 
            scene_list, 
            output_file_template=os.path.join(clips_dir, f"clip_$SCENE_NUMBER.mp4")
        )
        
        print(f"비디오 분할 완료: {len(scene_list)} 개의 클립 생성됨")
        return True
        
    except Exception as e:
        print(f"비디오 분할 실패 - 파일: {video_path}")
        print(f"에러 메시지: {str(e)}")
        return False

def main():
    # 비디오 파일이 있는 디렉토리와 출력 디렉토리 설정
    videos_dir = "./videos"
    output_dir = "./clips"
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # videos 디렉토리에서 모든 mp4 파일 찾기
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("처리할 비디오 파일이 없습니다.")
        return
        
    print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # 각 비디오 파일에 대해 분할 실행
    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(videos_dir, video_file)
        print(f"\n[{i}/{len(video_files)}] 처리 중...")
        if split_video(video_path, output_dir):
            success_count += 1
    
    print(f"\n모든 비디오 처리 완료!")
    print(f"성공: {success_count}/{len(video_files)}")

if __name__ == "__main__":
    main() 
