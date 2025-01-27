from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from moviepy import VideoFileClip
from utils.video_captioning import VideoCaptioningPipeline, find_video_file
from utils.embedding import FaissSearch, get_cached_model
from utils.translate import DeepLTranslator, Translator, ParallelTranslator, DeepGoogleTranslator
from sentence_transformers import SentenceTransformer
import json

app = Flask(__name__)

# Directory for storing search result video clips
STATIC_VIDEO_DIR = "static/search_results/t2v"
os.makedirs(STATIC_VIDEO_DIR, exist_ok=True)  # Ensure the directory exists
STATIC_VIDEO_DIR_v2t = "static/search_results/v2t"
os.makedirs(STATIC_VIDEO_DIR_v2t, exist_ok=True)  # Ensure the directory exists

# FAISS and DeepL API settings
VIDEOS_DIR = "./videos"
KEEP_CLIPS = False
SEGMENT_DURATION = 5
DEEPL_API_KEY = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"
JSON_PATH = "output/text2video/t2v_embedding.json"
SOURCE_JSON_PATH = "output/text2video/t2v_captions.json"
# VideoCaptioning_flag = False # True일 때 captioning 진행 (영상 한개 4분)
translator_mode = 'google' # deepl, translate, batch-deepl, batch-translate, google
max_workers = 4 # 번역 배치 크기

# if not VideoCaptioning_flag:
#     # ✅ 모델을 앱 시작 시 한 번만 로드
#     faiss_search = FaissSearch(json_path=JSON_PATH)

def save_search_result_clip(video_path, start_time, end_time, clip_name):
    """Saves a video clip from the original video"""
    try:
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        output_path = os.path.join(STATIC_VIDEO_DIR, f"{clip_name}.mp4")
        clip.write_videofile(output_path, codec='libx264', audio=False)
        clip.close()
        print(f"✅ Search result clip saved: {output_path}")
        return f"/{output_path}"  # Return web-friendly URL
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files dynamically"""
    return send_from_directory(STATIC_VIDEO_DIR, filename)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global faiss_search
    """Process search request"""
    data = request.json
    mode = data.get("mode")
    VideoCaptioning_flag = data.get("video_captioning_flag", False)  # ✅ 체크박스 값 적용

    if mode not in ["V2T", "T2V"]:
        return jsonify({"error": "Invalid mode"}), 400

    if VideoCaptioning_flag:
        pipeline = VideoCaptioningPipeline(keep_clips=KEEP_CLIPS, segment_duration=SEGMENT_DURATION)
        results = pipeline.process_directory(VIDEOS_DIR)
        
        if results:
            pipeline.save_results(results)

        # ✅ 모델을 앱 시작 시 한 번만 로드
        faiss_search = FaissSearch(json_path=JSON_PATH)

    if mode == "T2V":
        query_text = data.get("query_text")
        top_k = int(data.get("top_k", 2))
        print(top_k)
        if not query_text:
            return jsonify({"error": "No text provided"}), 400

        # ✅ 번역기 선택 (translator_mode 기반)
        if translator_mode == "deepl":
            translator = DeepLTranslator(api_key=DEEPL_API_KEY)
        elif translator_mode == "translate":
            translator = Translator()
        elif translator_mode == "batch-deepl":
            translator = ParallelTranslator(DeepLTranslator(api_key=DEEPL_API_KEY), max_workers=max_workers)
        elif translator_mode == "batch-translate":
            translator = ParallelTranslator(Translator(), max_workers=max_workers)
        elif translator_mode == "google":
            translator = DeepGoogleTranslator()
        else:
            raise ValueError(f"🚨 지원되지 않는 translator_mode: {translator_mode}")

        # ✅ FAISS 검색 객체 생성 및 임베딩 저장
        faiss_search = FaissSearch(json_path=JSON_PATH)
        faiss_search.generate_and_save_embeddings(SOURCE_JSON_PATH)

        # ✅ FAISS 검색 수행
        similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k)
        total_clips = len(similar_captions)  # 전체 클립 개수 계산

        # <h2 style='text-align: center; margin-bottom: 20px;'>🔍 검색 결과</h2><br><br>
        results_html = """
        <div style='display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;'>
        """
        output = []

        for i, (caption, similarity, video_info) in enumerate(similar_captions):
            clip_name = f"search_result_{i+1}_{video_info['clip_id']}"
            saved_path = save_search_result_clip(
                video_info["video_path"],
                video_info["start_time"],
                video_info["end_time"],
                clip_name
            )
            # 가운데 정렬
            #result_text = f"""
            # <div style='border: 2px solid #4CAF50; padding: 15px; max-width: 450px; background-color: #f9f9f9; border-radius: 10px; 
            #             display: flex; flex-direction: column; justify-content: space-between; min-height: 400px;'>
            #     <div style='flex-grow: 1;'>
            #         <h3 style='color: #4CAF50; text-align: center;'>🎯 검색 결과 {i+1} / 전체 {total_clips} 클립</h3>
            #         <p><strong>📌 클립 ID:</strong> {video_info["clip_id"]}</p>
            #         <p><strong>📊 유사도:</strong> {similarity:.4f}</p>
            #         <p><strong>🎬 비디오:</strong> {os.path.basename(video_info["video_path"])}</p>
            #         <p><strong>⏰ 구간:</strong> {video_info["start_time"]}초 ~ {video_info["end_time"]}초</p>
            #         <p><strong>📝 캡션:</strong> {caption}</p>
            #     </div>
            #     {f"<video width='100%' controls style='border-radius: 10px; margin-top: 10px;'><source src='{saved_path}' type='video/mp4'>Your browser does not support the video tag.</video>" if saved_path else "<p style='color: red;'>⚠️ 비디오 저장 오류</p>"}
            # </div>
            # """
            
            # 왼쪽 정렬
            result_text = f"""
            <div style="border: 2px solid #4CAF50; padding: 15px; max-width: 450px; background-color: #f9f9f9; border-radius: 10px; display: flex; flex-direction: column; justify-content: space-between; align-items: flex-start; min-height: 400px;">
    
                <div style="flex-grow: 1; text-align: left; width: 100%;">
                    <h3 style="color: #4CAF50; text-align: left;">🎯 검색 결과 {i+1} / 전체 {total_clips} 클립</h3>
                    <p><strong>📌 클립 ID:</strong> {video_info["clip_id"]}</p>
                    <p><strong>📊 유사도:</strong> {similarity:.4f}</p>
                    <p><strong>🎬 비디오:</strong> {os.path.basename(video_info["video_path"])}</p>
                    <p><strong>⏰ 구간:</strong> {video_info["start_time"]}초 ~ {video_info["end_time"]}초</p>
                    <p><strong>📝 캡션:</strong> {caption}</p>
                </div>

                {f"<video width='100%' controls style='border-radius: 10px; margin-top: 10px;'><source src='{saved_path}' type='video/mp4'>Your browser does not support the video tag.</video>" if saved_path else "<p style='color: red;'>⚠️ 비디오 저장 오류</p>"}
            </div>
            """

            results_html += result_text
            output.append({"similarity": float(similarity), "caption": caption, "clip_path": saved_path})

        results_html += "</div>"
        return jsonify({"message": "T2V processing complete", "results": output, "html": results_html})
    elif mode == "V2T":
        video_segments = data.get("video_segments", [])

        if not video_segments or not isinstance(video_segments, list):
            return jsonify({"error": "Invalid video segment data"}), 400

        # 파이프라인 초기화
        pipeline = VideoCaptioningPipeline(
            keep_clips=True,
            mode="video2text"
        )

        output = []
        video_process_input = []  # pipeline.process_videos()를 위한 리스트

        for segment in video_segments:
            video_id = segment.get("video_id")
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if not video_id or not start_time or not end_time:
                return jsonify({"error": "Missing video segment data"}), 400

            # 원본 비디오 경로
            original_video_path = f"/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/web/videos/{video_id}.mp4"

            # pipeline 처리용 데이터 추가
            video_process_input.append((original_video_path, start_time, end_time))


        # 📌 비디오 처리 실행 (JSON 생성 전에 실행해야 함)
        flag_v2t = True
        results = pipeline.process_videos(video_process_input, flag_v2t)
        pipeline.save_results(results)  # ./output/video2text/v2t_captions.json 생성됨


        caption_ko_map = {}

        try:
            with open("./output/video2text/v2t_captions.json", "r", encoding="utf-8") as f:
                v2t_captions = json.load(f)

            # ✅ clip_id 기준으로 caption_ko 매핑
            caption_ko_map = {
                item['clip_id']: item.get("caption_ko", "한국어 자막 없음") for item in v2t_captions
            }

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ JSON 파일 로드 오류: {e}")
            return jsonify({"error": "Failed to load caption JSON"}), 500

        # print("Caption KO Map:", caption_ko_map)  # 디버깅용 출력

        # ✅ output 리스트 구성 (clip_id는 JSON에서 가져옴)
        output = []
        for item in v2t_captions:
            output.append({
                "clip_id": item["clip_id"],  # pipeline에서 자동 생성된 clip_id 사용
                "video_id": item.get("video_id", "unknown_video"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "caption_ko": caption_ko_map.get(item["clip_id"], "한국어 자막 없음"),
                "clip_path": f"static/search_results/v2t/{video_id}_{item['clip_id']}.mp4"
            })

        for item in output:
            print(item['clip_path'])  # 모든 클립 출력

        # <h2 style='text-align: center; margin-bottom: 20px;'>🎥 V2T 검색 결과</h2><br><br>
        # ✅ HTML 출력 생성
        results_html = """
        <div style='display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;'>
        """

        for item in output:
            result_text = f"""
            <div style="border: 2px solid #4CAF50; padding: 15px; max-width: 450px; background-color: #f9f9f9; border-radius: 10px; display: flex; flex-direction: column; justify-content: space-between; align-items: flex-start; min-height: 400px;">
                <div style="flex-grow: 1; text-align: left; width: 100%;">
                    <h3 style="color: #4CAF50; text-align: left;">🎬 클립 정보</h3>
                    <p><strong>📌 클립 ID:</strong> {item["clip_id"]}</p>
                    <p><strong>⏰ 구간:</strong> {item["start_time"]}초 ~ {item["end_time"]}초</p>
                    <p><strong>🇰🇷 한국어 캡션:</strong> {item["caption_ko"]}</p>
                </div>
                {f"<video width='100%' controls style='border-radius: 10px; margin-top: 10px;'><source src='{item['clip_path']}' type='video/mp4'>Your browser does not support the video tag.</video>" if item['clip_path'] else "<p style='color: red;'>⚠️ 비디오 저장 오류</p>"}
            </div>
            """
            results_html += result_text

        results_html += "</div>"

        return jsonify({"message": "V2T processing complete", "results": output, "html": results_html})


    return jsonify({"message": "V2T processing complete"})

if __name__ == '__main__':
    app.run(debug=True)
