import yt_dlp
import os

def download_video(url, output_path):
    try:
        # yt-dlp 옵션 설정
        ydl_opts = {
            'format': 'best',  # 최고 품질
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # 출력 템플릿
            'quiet': False,  # 진행 상황 표시
            'no_warnings': False,
            'ignoreerrors': True,  # 에러 발생시 계속 진행
        }
        
        print(f"\n다운로드 시작: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
        
    except Exception as e:
        print(f"다운로드 실패 - URL: {url}")
        print(f"에러 메시지: {str(e)}")
        return False
    
def main():
    # 입력 파일과 출력 디렉토리 경로 설정
    urls_file = "dataset/movie_urls.txt"
    output_dir = "./videos"
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # URLs 파일 읽기
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {urls_file} 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(urls)}개의 URL을 찾았습니다.")
    
    # 각 URL에 대해 다운로드 실행
    success_count = 0
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] 처리 중...")
        if download_video(url, output_dir):
            success_count += 1
    
    print(f"\n다운로드 완료!")
    print(f"성공: {success_count}/{len(urls)}")

if __name__ == "__main__":
    main()
