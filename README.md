<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=CV-15&fontSize=40&animation=fadeIn&fontAlignY=38&desc=박지완&descAlignY=51&descAlign=62"/>
</p>

# Data

### Filter의 ID(mid)값 찾기 
```
cd dataset
grep "{Filter Name}" id.txt
```


***Example Usage***
```
grep "Movie theater" id.txt
```


### ID에 해당하는 list만들기
```
curl -o id_list/{file name}.txt https://storage.googleapis.com/data.yt8m.org/2/j/v/{id}.js
```

***Example Usage***
```
curl -o id_list/movie_ids.txt https://storage.googleapis.com/data.yt8m.org/2/j/v/0kcc7.js
```


### Filter의 데이터 다운로드
```
download_and_process.py (재훈님 도와줘요)
```

***Example Usage***
```
download_and_process.py (재훈님 도와줘요)
```

# My Project
This project focuses on using the YouTube 8M dataset to develop models for video-to-text and text-to-video applications.

The primary goal is to enable efficient translation of video content into meaningful textual descriptions and to identify matching video scenes based on text input.

## Features
- **Video-to-Text**: Generate detailed captions or descriptions from video data.
- **Text-to-Video**: Identify and retrieve video scenes that match the given text input.

## Dataset
This project leverages the [YouTube 8M dataset](https://research.google.com/youtube8m/) for training and evaluation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
아직 어떤 license를 사용해야할 지 모르겠음
