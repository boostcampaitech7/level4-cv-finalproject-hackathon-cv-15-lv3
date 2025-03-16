# Data Pre-processing


## 1. YouTube-8M

### 1-1. Get category infomation 

다음 레포지토리의 작업을 참고하여 작성되었습니다:

[gsssrao/yotube-8m-videos-frames](https://github.com/gsssrao/youtube-8m-videos-frames)

다음 코드를 실행하면 **YouTube-8M** 데이터에서 원하는 *category* 의 *video_id* 와 *url* 정보가 담긴 txt 파일을 얻을 수 있습니다. 

```bash
bash get_category_ids.sh (number-of-videos) (category-name)
```

**(number-of-videos)** : *category* 에서 얻을 *video_id* 의 갯수

모든 video를 얻고 싶으면 0 입력

**(category-name)** : 얻고 싶은 category 이름 

category 이름은 [youtube8mcategories.txt](./youtube8mcategories.txt) 에서 찾을 수 있습니다.

**Example usage**

```bash
bash get_category_ids.sh 10 Movieclips
```

### 1-2. Download raw videos & raw audios

urls.txt 정보를 활용하여 Youtube에서 해당 url에 해당하는 raw video와 raw audio를 다운로드합니다. 
코드를 실행하기 전에 *yt-dlp* 를 설치해주세요.

```bash
pip install yt-dlp
```

**Get category infomation** 단계에서 원하는 category의 video_id 와 url 텍스트 파일을 얻었다면, 다음 코드를 통해 video와 audio를 다운로드할 수 있습니다.

```bash
python download_videos_audios.py --category_name (category-name)
```

**Example Usage**

```bash
python download_videos_audios.py --category_name Movieclips
```

만약 우리의 코드가 아닌 다른 방법으로 url 텍스트 파일을 얻었다면 [download_vieos_audios.py](./download_videos_audios.py) 코드의 
*Todo* 부분에서 *input_file* 경로를 변경해주세요. 

### 1-3. Split videos and audios

PySceneDetect 라이브리러리를 활용해서 video 와 audio를 clip 단위로 분할합니다.
필요한 인자는 다음과 같습니다

- 필수 인자
    - category_name : ex. Movieclips

- 선택 인자
    - num_vieos : 몇 개의 비디오를 분할할지 갯수 default : 전체 비디오 분할
    - min_length : clip video의 최소 길이 : 3 (sec)
    - max_length : clip video의 최대 길이 : 15 (sec)

**Example Usage**

```bash
# if you want to use default-setting
python preprocessing.py --category_name Movieclips

# if you want to customize setting
python preprocessing.py --category_name Movieclips --num_videos 5 --min_length 2 --max_length 5
```




