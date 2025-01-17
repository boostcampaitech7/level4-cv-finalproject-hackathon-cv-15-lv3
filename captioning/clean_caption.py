import json

# JSON 파일 로드
with open("updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 캡션 수정 함수
def clean_caption(caption):
    sentences = caption.split(".")  # 문장 단위로 분리
    sentences = [s.strip() for s in sentences if s.strip()]  # 공백 제거 및 비어있는 문장 필터
    if sentences and not sentences[-1].endswith((".", "?", "!")):  # 마지막 문장이 완결되지 않았는지 확인
        sentences.pop()  # 마지막 문장 제거
    return ". ".join(sentences) + "." if sentences else ""  # 문장 합치기

# 모든 캡션 수정
for item in data:
    item["caption"] = clean_caption(item["caption"])

# 수정된 JSON 저장
with open("captions_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Caption cleaning completed.")
