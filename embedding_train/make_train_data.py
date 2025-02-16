import json
import gzip
import deepl
import os

# ✅ JSON 파일 로드 함수
def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ 일반 JSON 파일 저장 함수
def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ JSON dataset saved to {file_path}")

# ✅ gzip 압축된 JSONL 파일 저장 함수
def save_gzip_jsonl(data, file_path):
    """Save sentence pairs as a gzip-compressed JSONL file."""
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        for sample in data:
            json.dump(sample, f)
            f.write("\n")
    print(f"✅ Gzip-compressed dataset saved to {file_path}")

# ✅ DeepL API를 이용한 번역 함수
def translate_text(text, translator):
    """Translate Korean text to English using DeepL API."""
    if not text:
        return ""
    translation = translator.translate_text(text, source_lang="KO", target_lang="EN-US")
    print(translation.text)
    return translation.text

# ✅ 데이터셋 변환 함수
def create_sentence_pairs(input_file, json_output, gzip_output, auth_key):
    """Create a sentence pairs dataset and save as both JSON and gzip JSONL."""
    # JSON 파일 로드
    data = load_json(input_file)

    # DeepL 번역기 객체 생성
    translator = deepl.Translator(auth_key)

    # 문장 쌍 데이터 생성
    sentence_pairs = []
    for item in data:
        translated_query = translate_text(item["query"], translator)
        sentence_pairs.append([item["caption"], translated_query])  # [영어 caption, 번역된 영어 query]

    # ✅ JSON과 gzip JSONL 형식으로 저장
    save_json(sentence_pairs, json_output)
    save_gzip_jsonl(sentence_pairs, gzip_output)

# ✅ 실행 코드
if __name__ == "__main__":
    # DeepL API 키 설정 (환경 변수에서 가져오기)
    auth_key = os.getenv("DEEPL_API_KEY")  # 환경 변수에서 가져오기
    if not auth_key:
        auth_key = "aaa69e50-8536-4f58-a127-f94834afa71b:fx"  # 직접 입력 (보안 취약)

    # 파일 경로 설정
    input_file = "/data/ephemeral/home/gt_v6_update.json"
    json_output_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data/sentence_pairs.json"      # 일반 JSON
    gzip_output_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data/sentence_pairs.json.gz"  # 압축된 JSONL

    # 문장 쌍 데이터셋 생성 및 저장
    create_sentence_pairs(input_file, json_output_file, gzip_output_file, auth_key)
