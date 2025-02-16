import json
import re

# JSON 파일 경로
json_file_path = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/output copy.json'

# JSON 파일 읽기
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 비디오 ID를 저장할 집합
video_ids = set()

# 정규 표현식 패턴 (video_숫자 추출)
pattern = re.compile(r'video_(\d+)')

# 각 항목에서 video_id 추출
for item in data:
    match = pattern.search(item['video_path'])
    if match:
        video_ids.add(match.group(0))  # 'video_숫자' 형태로 추가

# 결과 출력
print("발견된 비디오 목록:")
for video_id in sorted(video_ids):
    print(video_id)
