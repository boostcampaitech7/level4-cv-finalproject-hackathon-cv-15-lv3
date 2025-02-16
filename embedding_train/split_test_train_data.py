import json
import gzip
import random

# ✅ JSON 파일 로드 함수
def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ JSON 파일 저장 함수
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

# ✅ 데이터셋 분할 함수
def split_dataset(input_file, train_output, test_output, split_ratio=0.9):
    """Split dataset into train and test sets and save them separately."""
    # JSON 데이터 로드
    data = load_json(input_file)
    
    # 데이터 섞기
    random.shuffle(data)

    # 분할 인덱스 설정
    split_idx = int(len(data) * split_ratio)

    # Train/Test 데이터 분리
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # ✅ Train: JSONL Gzip 파일로 저장
    save_gzip_jsonl(train_data, train_output)

    # ✅ Test: 일반 JSON 파일로 저장
    save_json(test_data, test_output)

# ✅ 실행 코드
if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data/sentence_pairs.json"
    train_output_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data/train_sentence_pairs.json.gz"  # Train용 gzip
    test_output_file = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data/test_sentence_pairs.json"       # Test용 JSON

    # 데이터셋 분할 및 저장
    split_dataset(input_file, train_output_file, test_output_file)
