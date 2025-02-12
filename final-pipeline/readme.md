# 티빙 해커톤 코드 실행 가이드 (CV-15)

평가는 서버 4번 [/data/ephemeral/home/final-pipeline](/data/ephemeral/home/final-pipeline) 에서 진행해주세요.

```bash
cd /data/ephemeral/home/final-pipeline
```

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
│   ├── input_video/                    # 가산점 평가용 외부 데이터(비디오)파일 위치
│   └── YouTube_8M/
│       ├── YouTube_8M_annotation.json  # Video_id, 제목 매핑 파일
│	└── YouTube_8M_video/           # 권장 데이터(비디오) 파일 위치
│
├── readme.md 
├── run.py                              # V2T, T2V 실행 스크립트
│
├── text2video_input.yaml               # Text to Video 평가 input 입력 파일 
└── video2text_input.yaml               # Video to Text 평가 input 입력 파일

```

## Video to Text 실행 방법

### 1. 기초 평가 (YouTube-8M) - 입력 방법

평가를 위한 ***(video_id, timestamp_start, timestamp_end)*** 정보를 [video2text_input.yaml](./video2text_input.yaml) 파일에 입력합니다.

- YouTube-8M ***비디오 제목 및 url*** 과 ***video_id*** 매핑 테이블은 [YouTube_8M_annotation.json](./videos/YouTube_8M/YouTube_8M_annotation.json) 파일에서 확인할 수 있습니다.

```bash
# YouTube-8M 정보 매핑 예시
# title 이나 url 정보로 video_id를 검색해주세요. (ex. Ctrl+F, cmd+F)

    {
        "video_id": "./videos/YouTube_8M_video/video_1.mp4",
        "title": "Legally Blonde 2 (3/11) Movie CLIP - The Testing Facility (2003) HD",
        "url_id": "uXsQ8IIi6YI",
        "url": "http://www.youtube.com/watch?v=uXsQ8IIi6YI"
    },
```

- 평가할 비디오의 ***시작(timestamp_start)*** 및 ***끝(timestamp_end)*** 구간을 명시해주세요.

```yaml
# example : video2text_input.yaml

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
│   │   ├── new_video_1.mp4             # 가산점 평가용 비디오
│   │   ├── new_video_2.mp4
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

---

### 3. 실행 방법

터미널에서 [final-pipeline](./final-pipeline) 폴더로 이동합니다.

```bash
# final-pipline/
python run.py video2text
```

- 구간으로 잘린 입력 비디오는 [clips/video2text/](./clips/video2text/) 폴더에서 확인할 수 있습니다.
- 결과는 터미널에 출력되고 [captioning_result.txt](./clips/video2text/captioning_result.txt)에 저장됩니다.

  ```bash
  # example 입력 : (비디오 파일명, 시작 시간, 종료 시간)

  final-pipeline
  │
  ├── clips/
  │   ├── video2text/
  │   │   ├── (비디오 파일명)_(시작 시간)_(종료 시간).mp4   # 클립 비디오  
  │   │   ├── captioning_result.txt   # cationing 결과
  ```
- Captioning 생성 시간은 구간 당 약 10~15초 정도 소요 됩니다. 터미널 출력이 없어도 기다려주세요.

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
    - "복싱 경기하는 장면"
```

### 2. 가산점 평가 (외부 비디오 + YouTube-8M) - 입력 방법

***process_new*** 를 ***true*** 로 변경해주세요.

***new_videos_dir*** 경로는 Video to Text 처럼 맞춰져 있으므로
[videos/input_video/](./videos/input_video/) 폴더에 가산점 평가용 비디오를 추가해주세요.

```yaml
# example : text2video_input.yaml 
process_new: true # 새로운 외부 비디오 입력 여부
new_videos_dir : ./videos/iput_video # 새로운 외부 비디오 root directory
... # (top_k, queries 입력방법 기초 평가와 동일)
```

```bash
final-pipeline
│
├── videos/
│   ├── input_video/                   # 가산점 평가용 외부 데이터(비디오)파일 위치
│   │   ├── new_video_1.mp4             # 가산점 평가용 비디오
│   │   ├── new_video_2.mp4
```

외부 비디오는 모두 폴더에 한 번에 넣고 실행을 해주시길 바랍니다.

**(참고)**
가산점 평가 시 전처리 진행 과정이 터미널에 늦게 출력되므로 기다려 주시길 바랍니다.
전처리(DB 생성)와 검색은 동시에 동작하고 소요 시간은 따로 출력 됩니다.

**(주의사항!)**
만약, 이미 한번 코드가 실행된 상태에서 외부 비디오를 추가로 넣고 싶다면
생성된 다음 파일 [output/text2video/temp_combined_db.json](./output/text2video/temp_combined_db.json)을 삭제한 뒤 외부비디오를 [videos/input_video/](./videos/input_video/)에 넣고 코드를 다시 실행시켜주세요

해당 경우 처음부터 전처리를 진행하므로 시간이 10분 가량 소요됩니다.

### 3. 실행 방법

터미널에서 [final-pipeline](./final-pipeline) 폴더로 이동합니다.

```bash
# final-pipline/
python run.py text2video
```

- retrieval 결과 클립 비디오는 [clips/text2video/](./clips/text2video/) 폴더에서 확인할 수 있습니다.
- 결과는 터미널에 출력되고 클립 생성 폴더에 [retrieval_result.txt](./clips/text2video/retrieval_result.txt)로도 저장됩니다.

  ```bash
  # example 입력 : (비디오 파일명, 시작 시간, 종료 시간)

  final-pipeline
  │
  ├── clips/
  │   ├── text2video/
  │   │   ├── (쿼리)_(rank)_(비디오 파일명)_(시작 시간)_(종료 시간).mp4   # retrieval 결과 클립 비디오  
  │   │   ├── retrieval_result.txt   # retrieval 결과
  ```
  **(참고)** 번역기가 동작을 안할 시 [utils/translator.py](./utils/translator.py) 코드에서 DeepLTranslator API key를 사전에 제공한 예비 API key로 변경해주세요.

  ```python
  # 예비 API key : aaa69e50-8536-4f58-a127-f94834afa71b:fx

  class DeepLTranslator:
    """DeepL API를 사용한 한국어 ↔ 영어 번역기 클래스"""

    def __init__(self):
        self.api_key = "{예비 API key}"
        self.url = "https://api-free.deepl.com/v2/translate"
  ```
