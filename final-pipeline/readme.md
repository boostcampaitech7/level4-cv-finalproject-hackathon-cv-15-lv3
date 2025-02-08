터미널상에서 final-pipeline 위치로 이동 부탁드립니다.

# 티빙 해커톤 코드 실행 가이드 (CV-15)

## Project Structure

```
final-pipeline
├── clips/   
│   ├── text2video/                     # Text to Video 클립 생성 위치
│   └── video2text/                     # Video to Text 클립 생성 위치  
├── database/   
├── output/
├── split_process/
├── text_to_video/
├── utils/
├── video_to_text/
├── videos/
│   ├── input_video/                    # 가산점 평가용 외부 데이터(비디오)파일 위치
│   └── YouTube_8M/
│       ├── YouTube_8M_annotation.json  # Video_id 매핑 파일
│	└── YouTube_8M_video/           # 권장 데이터(비디오) 파일 위치
├── readme.md 
├── run.py                              # V2T, T2V 실행 스크립트
├── text2video_input.yaml               # Text to Video 평가 input 입력 파일 
└── vidoe2text_input.yaml               # Video to Text 평가 input 입력 파일


```

## Video to Text

#### - Video to Text 평가를 위한 (video_id, timestamp_start, timestamp_end) 입력 방법

1. video2text_input.yaml파일에 들어갑니다.
2. 권장 데이터를 평가하는 경우(YouTube-8M) Video_id를 예시에 맞춰서 적으시면 됩니다.
3. 평가를 원하시는 구간 시작과 끝을 적어주시면 됩니다.
4. YouTube-8M 비디오의 제목과 Video_id 매핑 테이블은 위 폴더 구조에 나와있듯이
   videos/YouTube_8M/ 폴더에 YouTube_8M_annotation.json으로 있으므로 확인해 주시면 됩니다.
5. 외부 비디오를 평가하는 경우 videos/input_video/ 폴더에 넣어주시면 됩니다.
6. 외부 비디오는 video_id 대신 비디오 이름(파일명)을 예시에 맞춰서 적으시면 됩니다.
7. 마찬가지로 평가를 원하시는 구간 시작과 끝을 적어주시면 됩니다.

#### - Video to Text 결과 확인 방법

1. 위에서 입력을 넣으셨다면 터미널상에서 final-pipeline 위치로 이동 부탁드립니다.
2. 다음 명령어를 입력해 주시면 됩니다.
   ```
   python run.py video2text
   ```
3. 스크립트 실행 결과가 터미널에 출력됩니다.
4. 입력 비디오 구간을 클립으로 보고 싶으시면 위 폴더 구조에 나와있듯이
   clips/video2text/ 폴더에 생성이 되므로 확인해 주시면 될 것 같습니다.

## Text to Video (Frame)

#### - Text to Video 평가를 위한 쿼리 입력 방법

1. text2video_input.yaml파일에 들어갑니다.
2. 권장 데이터만 평가하는 기초 평가의 경우 process_new를 false로 해주시면 됩니다.
3. query 부분에 평가를 원하시는 쿼리를 예시에 맞춰서 넣어주시면 됩니다.
4. top_k 부분에 원하시는 쿼리에 대한 응답 개수를 적어주시면 됩니다.
5. 가산점 평가를 진행하시는 경우 process_new를 true로 해주시면 됩니다.
6. 가산점 평가를 위한 외부 비디오들은 videos/input_video/ 폴더에 한 번에 전부 넣어주시면 됩니다.
   (Video to Text에서 진행한 외부 비디오와 같은 경우 그대로 사용하시고 아닌경우 비워주신 뒤 추가하시면 됩니다.)
7. 마찬가지로 평가를 원하시는 쿼리를 예시에 맞춰서 넣어주시면 됩니다.

#### - Text to Video 결과 확인 방법

1. 위에서 입력을 넣으셨다면 터미널상에서 final-pipeline 위치로 이동 부탁드립니다.
2. 다음 명령어를 입력해 주시면 됩니다.
   ```
   pyhton run.py text2video
   ```
3. 스크립트 실행 결과(비디오 이름, 구간)가 터미널에 출력됩니다.
4. 검색 결과의 비디오 클립은 위 폴더 구조에 나와있듯이
   clips/text2video/ 폴더에 생성이 되므로 확인해 주시면 될 것 같습니다.
5. 가산점 평가 시에도 위에서 입력 방법만 바꾸신 후에 2번과 같은 명령어를 사용해 주시면 됩니다.

### 감사합니다.
