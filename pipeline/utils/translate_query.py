import json
import os
from tqdm import tqdm
from translator import DeepGoogleTranslator  # ✅ 번역기 클래스 임포트

def translate_queries(json_path, output_json, save_every_n=30):
    """
    JSON 파일에서 "query" 필드를 한국어로 번역하고 기존 영어 문장은 "query_en"으로 저장하는 함수.
    
    Args:
        json_path (str): 입력 JSON 파일 경로
        output_json (str): 번역된 데이터를 저장할 JSON 경로
        save_every_n (int): N개 처리할 때마다 JSON을 저장하는 간격 (기본값: 30)
    """

    # JSON 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translator = DeepGoogleTranslator()  # ✅ 번역기 객체 생성
    failed_queries = []  # 번역 실패한 문장 리스트
    save_counter = 0  # 중간 저장 카운터

    try:
        # ✅ tqdm 추가하여 진행률 표시
        for entry in tqdm(data, total=len(data), desc="Translating Queries"):
            if 'query' not in entry or not entry['query'].strip():
                continue  # ✅ "query" 필드가 없거나 빈 값이면 스킵

            # ✅ 이미 번역된 경우 스킵
            if 'query_en' in entry and entry['query_en'].strip():
                continue  

            original_query = entry['query']

            # ✅ 영어 → 한국어 번역 수행
            translated_query = translator.translate_en_to_ko(original_query)

            # ✅ 번역 실패한 경우, 로그 저장 후 스킵
            if not translated_query:
                failed_queries.append(original_query)
                continue  

            # ✅ 기존 영어 문장은 "query_en" 필드에 저장
            entry['query_en'] = original_query
            entry['query'] = translated_query  # ✅ 번역된 한국어 문장 저장

            # ✅ 주기적으로 JSON 저장
            save_counter += 1
            if save_counter % save_every_n == 0:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        # ✅ 전체 데이터 처리 완료 후 저장
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"❌ Error encountered: {e}")
        print(f"Saving progress before exiting...")

        # ✅ 오류 발생 시 JSON 저장 후 안전 종료
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    finally:
        # ✅ 번역 실패한 문장 로그 저장
        if failed_queries:
            failed_log_path = "log_failed_queries.txt"
            with open(failed_log_path, 'w', encoding='utf-8') as log_file:
                for query in failed_queries:
                    log_file.write(query + "\n")
            print(f"❗ Failed queries saved at {failed_log_path}")

    print(f"✅ Translation complete! JSON saved at {output_json}")

# ✅ 사용 예시
translate_queries(
    json_path='/data/ephemeral/home/chan/level4-cv-finalproject-hackathon-cv-15-lv3/embedding/pre-processing/gt_videos.json',
    output_json='gt_db.json'
)