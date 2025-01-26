# 입력/출력 디렉토리 설정
INPUT_DIR="videos"
OUTPUT_DIR="audios"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 각 비디오 파일에 대해
for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    output="$OUTPUT_DIR/${filename}.wav"
    
    echo "Processing $filename..."
    
    # 오디오 스트림이 있는지 확인
    if ffmpeg -i "$video" 2>&1 | grep -q "Stream.*Audio:"; then
        # 오디오 추출 시도
        ffmpeg -i "$video" -map 0:a:0 -vn -acodec pcm_s16le -ar 16000 -ac 1 "$output" -y 2>/dev/null || {
            echo "Failed to convert $filename"
        }
    else
        echo "No audio stream found in $filename"
    fi
done

# 결과 확인
echo "Total videos: $(find "$INPUT_DIR" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" -o -name "*.webm" \) | wc -l)"
echo "Total audios converted: $(find "$OUTPUT_DIR" -type f -name "*.wav" | wc -l)"