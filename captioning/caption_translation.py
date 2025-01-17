import deepl
import json

def translate_captions(input_file, output_file, auth_key):
    # DeepL Translator Client 생성
    translator = deepl.Translator(auth_key)

    # JSON 파일 읽기
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 영어 캡션 → 한국어 번역
    for item in data:
        translation = translator.translate_text(item["caption"], source_lang="EN", target_lang="KO")
        item["caption_kr"] = translation.text  # 번역 결과 저장

    # JSON 파일 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 함수 호출 예시
auth_key = "70c47127-a731-47c8-a901-69f325feacf7:fx"  # DeepL API 키
translate_captions("updated_Movieclips_annotations.json", "captions_translated_deepl2.json", auth_key)
