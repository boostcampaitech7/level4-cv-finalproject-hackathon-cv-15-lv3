from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from moviepy import VideoFileClip
from video_captioning import VideoCaptioningPipeline, find_video_file
from embedding import FaissSearch, DeepLTranslator

app = Flask(__name__)

# Directory for storing search result video clips
STATIC_VIDEO_DIR = "static/search_results"
os.makedirs(STATIC_VIDEO_DIR, exist_ok=True)  # Ensure the directory exists

# FAISS and DeepL API settings
VIDEOS_DIR = "/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/videos"
KEEP_CLIPS = False
SEGMENT_DURATION = 5
DEEPL_API_KEY = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"
JSON_PATH = "output/embedding.json"
SOURCE_JSON_PATH = "output/captions.json"

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
    """Process search request"""
    data = request.json
    query_text = data.get("query_text")
    mode = data.get("mode")

    if not query_text:
        return jsonify({"error": "No text provided"}), 400

    if mode not in ["V2T", "T2V"]:
        return jsonify({"error": "Invalid mode"}), 400

    # pipeline = VideoCaptioningPipeline(keep_clips=KEEP_CLIPS, segment_duration=SEGMENT_DURATION)
    # results = pipeline.process_directory(VIDEOS_DIR)
    
    # if results:
    #     pipeline.save_results(results)

    if mode == "T2V":
        translator = DeepLTranslator(api_key=DEEPL_API_KEY)
        faiss_search = FaissSearch(json_path=JSON_PATH)
        faiss_search.generate_and_save_embeddings(SOURCE_JSON_PATH)

        similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=2)
        total_clips = len(similar_captions)  # 전체 클립 개수

        results_html = """
        <h2 style='text-align: center; margin-bottom: 20px;'>🔍 검색 결과</h2><br><br>
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

            result_text = f"""
            <div style='border: 2px solid #4CAF50; padding: 15px; max-width: 450px; background-color: #f9f9f9; border-radius: 10px; 
                        display: flex; flex-direction: column; justify-content: space-between; min-height: 400px;'>
                <div style='flex-grow: 1;'>
                    <h3 style='color: #4CAF50; text-align: center;'>🎯 검색 결과 {i+1} / 전체 {total_clips} 클립</h3>
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

    return jsonify({"message": "V2T processing complete"})

if __name__ == '__main__':
    app.run(debug=True)
