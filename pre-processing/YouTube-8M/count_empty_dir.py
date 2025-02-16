import os

def count_empty_folders(directory):
    empty_folder_count = 0
    for root, dirs, files in os.walk(directory):
        if not dirs and not files:  # 폴더 안에 아무것도 없으면 빈 폴더로 간주
            empty_folder_count += 1
    return empty_folder_count

# 사용자가 지정할 폴더 경로 입력
directory_path = "/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/pre-processing/YouTube-8M/data/YouTube-8M-clips"  # 원하는 경로로 변경
empty_folders = count_empty_folders(directory_path)
print(f"빈 폴더 개수: {empty_folders}")
