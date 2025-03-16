import argparse
import yaml
import json
import os
import sys
import time
import subprocess
from utils.translator import DeepGoogleTranslator, DeepLTranslator
from video_to_text.video_captioning import TarsierVideoCaptioningPipeline
from text_to_video.embedding import FaissSearch
from split_process.main_server.main_server_run import main as split_process_main
from split_process.main_server.config import Config as SplitConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def video_to_text_process():
    """비디오를 텍스트로 변환하는 파이프라인"""
    print("\n🚀 비디오-텍스트 변환 파이프라인 시작...")
    process_start_time = time.time()
    
    # YAML 설정 파일 로드
    try:
        with open('video2text_input.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {str(e)}")
        return

    # 기본 설정값 (코드로 관리)
    KEEP_CLIPS = True  # 클립 저장을 위해 True로 변경
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "/data/ephemeral/home/Tarsier-7b"
    clips_dir = os.path.join(current_dir, "clips/video2text/")  # 클립 저장 경로
    
    # clips 디렉토리 생성
    os.makedirs(clips_dir, exist_ok=True)

    # 파이프라인 초기화
    pipeline = TarsierVideoCaptioningPipeline(
        model_path=model_path,
        keep_clips=KEEP_CLIPS,
        mode="video2text",
        video_metadata={},
        clips_dir=clips_dir  # 클립 저장 경로 지정
    )
    
    # 비디오 처리
    video_list = []
    for video_data in config.get('videos', []):
        video_path = video_data['video_id']
        
        if not os.path.exists(video_path):
            print(f"⚠️ 비디오를 찾을 수 없음: {video_path}")
            continue
        
        video_list.extend([
            (video_path, ts['start_time'], ts['end_time'])
            for ts in video_data['timestamps']
        ])
    
    if not video_list:
        print("❌ 처리할 비디오가 없습니다.")
        return
    
    # 비디오 처리 및 캡션 생성
    result_txt_path = os.path.join(clips_dir, "captioning_result.txt")
    with open(result_txt_path, 'w', encoding='utf-8') as f:
        # 비디오 처리 및 캡션 생성
        print(f"\n🎥 비디오 처리 중... (총 {len(video_list)}개 클립)")
        f.write(f"\n🎥 비디오 처리 중... (총 {len(video_list)}개 클립)\n")
        results = []
        for idx, (video_path, start_time, end_time) in enumerate(video_list, 1):
            process_msg = f"\n처리 중: {idx}/{len(video_list)} - {os.path.basename(video_path)} ({start_time}초 ~ {end_time}초)"
            print(process_msg)
            f.write(process_msg + "\n")
            
            result = pipeline.process_video(video_path, start_time, end_time)
            if result:
                results.append(result)
                print(f"✅ 완료")
                f.write("✅ 완료\n")
        
        # 결과 출력
        print("\n📝 생성된 캡션:")
        print("=" * 80)
        f.write("\n📝 생성된 캡션:\n")
        f.write("=" * 80 + "\n")
        
        for i, ((original_path, start_time, end_time), result) in enumerate(zip(video_list, results), 1):
            if 'YouTube_8M/YouTube_8M_video' in original_path:
                video_name = os.path.basename(original_path)
                mapping_path = './videos/YouTube_8M/YouTube_8M_annotation/Movieclips_annotation.json'
                
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as map_f:
                        mapping_data = json.load(map_f)
                        video_info = next(
                            (item for item in mapping_data if item['video_name'] == video_name),
                            None
                        )
                        if video_info:
                            video_title = video_info['title']
                            clip_info = f"\n🎬 클립 {i}: {video_title} (ID: {video_name})"
                        else:
                            clip_info = f"\n🎬 클립 {i}: {video_name}"
                except Exception as e:
                    clip_info = f"\n🎬 클립 {i}: {video_name}"
            else:
                video_name = os.path.basename(original_path)
                clip_info = f"\n🎬 클립 {i}: {video_name}"
            
            print(clip_info)
            f.write(clip_info + "\n")
            
            result_info = f"⏰ 구간: {result['start_time']}초 ~ {result['end_time']}초\n결과: {result['caption_ko']}"
            print(result_info)
            f.write(result_info + "\n")
            
            separator = "-" * 80
            print(separator)
            f.write(separator + "\n")
        
        # 결과 출력 후 시간 계산
        total_time = time.time() - process_start_time
        minutes, seconds = divmod(total_time, 60)
        
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            time_msg = f"\n✨ 전체 처리 완료 (총 {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초)"
        else:
            time_msg = f"\n✨ 전체 처리 완료 (총 {int(minutes)}분 {seconds:.1f}초)"
        
        print(time_msg)
        f.write(time_msg + "\n")
        
        summary = f"📊 처리된 세그먼트: {len(results)}/{len(video_list)}\n💾 클립 저장 위치: {clips_dir}"
        print(summary)
        f.write(summary + "\n")
    
    print(f"📝 결과가 저장됨: {result_txt_path}")

def save_search_clip(video_path, output_path, start_time, end_time):
    """검색 결과 비디오 클립을 저장하는 함수"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-c", "copy",
        output_path,
        "-y"
    ]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def text_to_video_search():
    """텍스트로 비디오 검색하는 파이프라인"""
    print("\n🚀 텍스트-비디오 검색 파이프라인 시작...")
    start_time = time.time()
    
    # YAML 설정 파일 로드
    try:
        with open('text2video_input.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {str(e)}")
        return

    queries = config.get('queries', [])
    process_new = config.get('process_new', False)
    new_videos_dir = config.get('new_videos_dir', '')
    top_k = config.get('top_k', 1)

    # DB 경로 설정
    main_db_path = "database/caption_embedding_tf_357_final4.json"
    new_db_path = "output/text2video/new_videos_captions.json"
    temp_db_path = "output/text2video/temp_combined_db.json"

    # 새로운 비디오가 있는 경우 처리
    if process_new and new_videos_dir and os.path.exists(new_videos_dir):
        if not os.path.exists(temp_db_path):  # temp_combined_db가 없는 경우만 새로 생성
            print(f"\n🎥 새로운 비디오 처리 중... ({new_videos_dir})")
            
            # 설정 업데이트 및 분산 처리
            SplitConfig.VIDEOS_DIR = new_videos_dir
            SplitConfig.SPLIT_VIDEOS_DIR = os.path.join(new_videos_dir, "split")
            
            print("📦 외부 비디오 전처리 시작...")
            process_start_time = time.time()
            split_process_main()
            
            # JSON 결과 취합
            print("\n📊 외부 비디오 전처리 결과 취합 중...")
            json_results = []
            json_dir = "/data/ephemeral/home/json"
            
            for json_file in os.listdir(json_dir):
                if json_file.startswith("video_files_") and json_file.endswith(".json"):
                    with open(os.path.join(json_dir, json_file), 'r') as f:
                        json_results.extend(json.load(f))
            
            # 새 결과를 DB에 저장
            with open(new_db_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=4, ensure_ascii=False)
            
            # temp_combined_db 생성
            print("🔄 통합 DB 생성 중...")
            with open(main_db_path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
            combined_data.extend(json_results)
            
            with open(temp_db_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=4, ensure_ascii=False)
            
            external_data_preprocessing_time = time.time() - process_start_time
            print(f"⏱️ 외부 데이터 전처리 완료 시간 ({external_data_preprocessing_time:.1f}초)")
    
    # 클립 저장 디렉토리 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_clips_dir = os.path.join(current_dir, "clips/text2video/")
    os.makedirs(search_clips_dir, exist_ok=True)

    # 검색 결과 저장할 txt 파일 생성
    result_txt_path = os.path.join(search_clips_dir, "retrieval_result.txt")
    with open(result_txt_path, 'w', encoding='utf-8') as f:
        # FAISS 검색
        search_time = time.time()
        translator = DeepLTranslator()
        
        # DB 선택
        if process_new and os.path.exists(temp_db_path):
            search_db_path = temp_db_path
            print("🔍 통합 DB에서 검색 준비 중...")
        else:
            search_db_path = main_db_path
            print("🔍 기본 DB에서 검색 준비 중...")
        
        faiss_search = FaissSearch(json_path=search_db_path)
        all_results = {}  # 모든 쿼리의 결과를 저장할 딕셔너리
        
        print(f"\n총 {len(queries)}개의 쿼리 처리 시작...")
        
        for query_idx, query_text in enumerate(queries, 1):
            print(f"\n📝 쿼리 {query_idx}/{len(queries)}: '{query_text}'")
            
            similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=top_k)
            all_results[query_text] = similar_captions
            
            external_video_dir = "./videos/input_video"
            youtube_videos_dir = "./videos/YouTube_8M/YouTube_8M_video"

            # 각 쿼리의 결과 출력 및 클립 저장
            print(f"\n🎯 '{query_text}'의 검색 결과:")
            f.write(f"\n검색 결과:\n")
            for i, (similarity, video_info) in enumerate(similar_captions, 1):
                video_path = video_info['video_path']
                video_start_time = float(video_info['start_time'])
                video_end_time = float(video_info['end_time'])
                
                # video_id 유무에 따라 비디오 경로 결정
                if 'video_id' in video_info and video_info['video_id']:
                    # YouTube 비디오인 경우 경로 수정
                    video_folder = video_path.split('/')[0]
                    full_video_path = os.path.join(youtube_videos_dir, f"{video_folder}.mp4")
                else:
                    # 외부 입력 비디오인 경우
                    full_video_path = os.path.join(external_video_dir, video_path)
                
                if not os.path.exists(full_video_path):
                    print(f"  ⚠️ 원본 비디오를 찾을 수 없음: {video_path}")
                    continue
                    
                # 클립 파일명 생성
                query_slug = "_".join(query_text.split())[:30]
                base_video_name = os.path.splitext(os.path.basename(video_path))[0]
                if 'video_id' in video_info and video_info['video_id']:
                    # YouTube 비디오인 경우 폴더명을 사용
                    base_video_name = video_path.split('/')[0]
                clip_filename = f"{query_slug}_rank{i}_{base_video_name}_{video_start_time}_{video_end_time}.mp4"
                clip_path = os.path.join(search_clips_dir, clip_filename)
                
                try:
                    # ffmpeg로 비디오 클립 추출
                    save_search_clip(full_video_path, clip_path, video_start_time, video_end_time)
                    print(f"  💾 클립 저장: {clip_filename}")
                except Exception as e:
                    print(f"  ⚠️ 클립 저장 실패: {str(e)}")
                
                # 클립 저장 결과를 txt에 기록
                result_text = f"""
결과
📊 유사도: {similarity:.4f}
🎬 비디오: {os.path.basename(video_path)}
⏰ 구간: {video_start_time}초 ~ {video_end_time}초
📝 제목: {video_info['title']}
🔍 검색어: {query_text}
💾 저장된 클립: {clip_filename if os.path.exists(clip_path) else '저장 실패'}
----------------------------------------
"""
                f.write(result_text)
                print(result_text)
                            
                f.write(f"\n검색어 '{query_text}' 처리 완료\n")
                f.write("-" * 50 + "\n")
                        
                # 검색 완료 정보 저장
                summary = f"""
\n검색 요약:
• 총 검색어 수: {len(queries)}개
• 검색 소요 시간: {time.time() - search_time:.1f}초
• 클립 저장 위치: {search_clips_dir}
• 전체 처리 시간: {time.time() - start_time:.1f}초
"""
                f.write(summary)
    
    print(f"\n⏱️ 전체 검색 완료 ({time.time() - search_time:.1f}초)")
    print(f"💾 클립 저장 위치: {search_clips_dir}")
    
    total_time = time.time() - start_time
    print(f"\n✨ 전체 처리 완료 (총 {total_time:.1f}초)")

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('mode', choices=['text2video', 'video2text'],
                      help='Choose pipeline mode: text2video or video2text')
    
    args = parser.parse_args()
    
    if args.mode == 'text2video':
        text_to_video_search()
    else:
        video_to_text_process()

if __name__ == "__main__":
    main()