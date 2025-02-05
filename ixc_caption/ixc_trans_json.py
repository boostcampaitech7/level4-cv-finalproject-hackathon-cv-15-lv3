import json
import re

# 파일 경로 설정
output_copy_path = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/output_copy.json'
matched_videos_path = '/data/ephemeral/home/data/split_exp/matched_videos.json'

# JSON 파일 읽기
with open(output_copy_path, 'r') as file:
    output_data = json.load(file)

with open(matched_videos_path, 'r') as file:
    matched_videos = json.load(file)

# matched_videos를 딕셔너리로 변환 (빠른 검색을 위해)
matched_dict = {video['video_name']: video for video in matched_videos}

# 정규 표현식 패턴 (video_숫자 형태와 추가 번호까지 추출)
pattern = re.compile(r'(video_\d+)')

# output_data 업데이트
for item in output_data:
    match = pattern.search(item['video_path'])
    if match:
        video_base_name = match.group(0) + '.mp4'  # 예: video_32.mp4

        # 일치하는 비디오가 있는 경우 정보 업데이트
        if video_base_name in matched_dict:
            matched_video = matched_dict[video_base_name]
            item['video_id'] = matched_video['video_id']
            item['video_title'] = matched_video['title']
            item['video_url'] = matched_video['url']

# 업데이트된 JSON 파일 저장
updated_output_path = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/updated_output_copy.json'
with open(updated_output_path, 'w') as file:
    json.dump(output_data, file, indent=4)

print(f'업데이트된 JSON 파일이 저장되었습니다: {updated_output_path}')
