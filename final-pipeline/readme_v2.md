# 티빙 해커톤 코드 실행 가이드 (CV-15)

## 프로젝트 폴더 구조

```bash
final-pipeline
├── clips/   
│   ├── text2video/                     # Text to Video 클립 생성 위치
│   └── video2text/                     # Video to Text 클립 생성 위치  
│
├── database/   
├── output/
├── split_process/
├── text_to_video/
├── utils/
├── video_to_text/
│
├── videos/
│   ├── input_video/                   # 가산점 평가용 외부 데이터(비디오)파일 위치
│   └── mapping/
│       ├── YouTube_8M_annotation.json # Video_id 매핑 파일
│	└── YouTube_8M_video/              # 권장 데이터(비디오) 파일 위치
│
├── readme.md 
├── run.py                             # V2T, T2V 실행 스크립트
│
├── text2video_input.yaml              # Text to Video 평가 input 입력 파일 
└── video2text_input.yaml              # Video to Text 평가 input 입력 파일

```

## Video to Text 실행 방법


### 1. 기초 평가 (YouTube-8M) - 입력 방법


평가를 위한 ***(video_id, timestamp_start, timestamp_end)*** 정보를 [video2text_input.yaml](./video2text_input.yaml) 파일에 입력합니다.

- YouTube-8M ***비디오 제목 및 url*** 과 ***video_id*** 매핑 테이블은 [YouTube_8M_annotation.json](./mapping/YouTube_8M_annotation.json) 파일에서 확인할 수 있습니다.

- 평가할 비디오의 ***시작(timestamp_start)*** 및 ***끝(timestamp_end)*** 구간을 명시해주세요.
```yaml
# example : video2text_input.yaml

# YouTube-8M 권장 데이터 셋 예시
# - video_id: videos/YouTube_8M/YouTube_8M_video/{video_name}.mp4
#   timestamps:
#     - {start_time: 0.0, end_time: 5.0}

videos:
  - video_id: ./videos/YouTube_8M_video/video_257.mp4 # video_id
    timestamps:  # 처리할 시간 구간들
      - {start_time: 55.0, end_time: 60.0}
```
---
### 2. 가산점 평가 (외부 비디오) - 입력 방법
외부 비디오를 평가하려면 [videos/input_video/](./videos/input_video/) 폴더에 가산점 평가용 비디오를 추가해주세요.
```bash
final-pipeline
│
├── videos/
│   ├── input_video/                   # 가산점 평가용 외부 데이터(비디오)파일 위치
│   │   ├── new_vieo_1.mp4             # 가산점 평가용 비디오
│   │   ├── new_vieo_2.mp4
```

평가를 위한 ***(video_id, timestamp_start, timestamp_end)*** 정보를 [video2text_input.yaml](./video2text_input.yaml) 파일에 입력해주세요.
```yaml
# example : video2text_input.yaml

# 외부 Input Video 예시
# - video_id: videos/input_video/{Video File Name}.mp4
#   timestamps:
#     - {start_time: 0.0, end_time: 5.0}

videos:
  - video_id: ./videos/input_video/new_video_1.mp4 # new video path
    timestamps:  # 처리할 시간 구간들
      - {start_time: 0.0, end_time: 5.0}
```
----
### 3. 실행 방법
터미널에서 [final-pipeline](./final-pipeline) 폴더로 이동합니다.
```bash
# final-pipline/
python run.py video2text
```
- 결과는 터미널에 출력됩니다.
- 클립 파일을 확인하려면 [clips/video2text/](./clips/video2text/) 폴더를 확인하세요.

---

## Text to Video 실행 방법

### 1. 기초 평가 (YouTube-8M) - 입력 방법

[text2video_input.yaml](./text2video_input.yaml) 파일에 다음 정보들을 입력해주세요.

```yaml
# example : text2video_input.yaml
process_new: false # 새로운 외부 비디오 입력 여부
new_videos_dir : ./videos/iput_video # 새로운 외부 비디오 root directory
top_k : 1    # 검색 영상 갯수                             
queries:     # 입력 Query
    - "남자들이 헤드셋 끼고 컴퓨터 하는 장면"
    - "두 사람이 눈 오는 날 걷는 장면"
    - 복싱 경기하는 장면"
``` 

### 2. 가산점 평가 (외부 비디오 + YouTube-8M) - 입력 방법
***process_new*** 를 ***true*** 로 변경해주세요.

***new_videos_dir*** 경로를 정확하게 입력해주세요.

```yaml
# example : text2video_input.yaml 
process_new: true # 새로운 외부 비디오 입력 여부
new_videos_dir : ./videos/iput_video # 새로운 외부 비디오 root directory
... # (top_k, queries 입력방법 기초 평가와 동일)
```

---

### 3. 실행 방법
터미널에서 [final-pipeline](./final-pipeline) 폴더로 이동합니다.
```bash
# final-pipline/
python run.py text2video
```
- 검색 결과는 터미널에 출력됩니다.
- 클립 파일은 [clips/text2video/](./clips/text2video/) 폴더에서 확인할 수 있습니다.

