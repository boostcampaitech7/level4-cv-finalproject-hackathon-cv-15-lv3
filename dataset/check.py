import json
import glob
import os

def load_movie_theater_ids(filename="id_movie.txt"):
    """Movie theater ID 목록 로드"""
    with open(filename, 'r') as f:
        content = f.read()
        start = content.find('[')
        end = content.find(']')
        if start != -1 and end != -1:
            ids = content[start+1:end].replace('"', '').split(',')
            return set(ids)
    return set()

def analyze_json_files():
    total_files = 0
    total_videos = 0
    all_video_ids = set()  # 중복 ID 체크를 위한 set
    original_ids = load_movie_theater_ids()  # 원본 ID 목록 로드
    
    print("\n=== JSON Files Analysis ===")
    print("\nFiles with multiple videos:")
    print("-" * 50)
    
    # 모든 JSON 파일 처리
    for json_file in glob.glob("movie_theater_*.json"):
        total_files += 1
        with open(json_file, 'r') as f:
            data = json.load(f)
            videos_in_file = len(data['records'])
            total_videos += videos_in_file
            
            # 현재 파일의 video ID들 수집
            current_file_ids = []
            for record in data['records']:
                video_id = record['context']['feature'][0]['value']['bytes_list']['value']
                current_file_ids.append(video_id)
                all_video_ids.add(video_id)
            
            # 2개 이상의 비디오가 있는 파일 출력
            if videos_in_file > 1:
                print(f"File: {json_file}")
                print(f"Number of videos: {videos_in_file}")
                print("Video IDs:", current_file_ids)
                print("-" * 50)
    
    # 전체 통계 출력
    print("\n=== Summary ===")
    print(f"Total JSON files found: {total_files}")
    print(f"Total videos across all files: {total_videos}")
    print(f"Unique video IDs: {len(all_video_ids)}")
    
    if total_videos != len(all_video_ids):
        print(f"\nNote: Found {total_videos - len(all_video_ids)} duplicate video IDs")
    
    # 파일 크기 정보
    total_size = sum(os.path.getsize(f) for f in glob.glob("movie_theater_*.json"))
    print(f"\nTotal size of all JSON files: {total_size / (1024*1024):.2f} MB")
    
    # ID 비교 분석
    print("\n=== ID Analysis ===")
    print(f"Original movie theater IDs: {len(original_ids)}")
    print(f"Found video IDs: {len(all_video_ids)}")
    
    # 원본 목록에 없는 ID 찾기
    unexpected_ids = all_video_ids - original_ids
    if unexpected_ids:
        print(f"\nFound {len(unexpected_ids)} IDs not in original list:")
        for id in unexpected_ids:
            print(f"- {id}")
    
    # 찾지 못한 ID 찾기
    missing_ids = original_ids - all_video_ids
    if missing_ids:
        print(f"\nMissing {len(missing_ids)} IDs from original list:")
        for id in missing_ids:
            print(f"- {id}")

if __name__ == "__main__":
    analyze_json_files()