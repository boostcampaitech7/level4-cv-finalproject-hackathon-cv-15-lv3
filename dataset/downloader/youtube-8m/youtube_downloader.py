from pytube import YouTube
import os

def download_youtube_video(url, output_path='.'):
    try:
        # YouTube 객체 생성
        yt = YouTube(url)
        
        # 비디오 스트림 선택 (가장 높은 해상도)
        video_stream = yt.streams.get_highest_resolution()
        
        # 비디오 다운로드
        print(f"Downloading: {yt.title}")
        video_stream.download(output_path)
        print("Download completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def download_videos_from_file(file_path, output_path='.'):
    # 파일에서 URL 읽기
    with open(file_path, 'r') as file:
        urls = file.readlines()
    
    # 각 URL에 대해 다운로드 수행
    for url in urls:
        url = url.strip()  # 공백 제거
        if url:  # URL이 비어있지 않은 경우
            download_youtube_video(url, output_path)

# 사용 예
if __name__ == "__main__":
    url_file_path = 'Movieclips_urls.txt'  # URL 파일 경로
    output_directory = 'downloaded_videos'  # 다운로드할 폴더 경로

    # 다운로드할 폴더 생성
    os.makedirs(output_directory, exist_ok=True)

    download_videos_from_file(url_file_path, output_directory)