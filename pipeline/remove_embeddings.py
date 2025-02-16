import json
import os

def remove_embeddings(input_json_path):
    """JSON 파일에서 embedding 필드를 제거하는 함수"""
    print(f"\n🔄 임베딩 제거 시작: {input_json_path}")
    
    # 원본 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # embedding 필드 제거
    for item in data:
        if 'embedding' in item:
            del item['embedding']
    
    # 결과 저장
    output_path = input_json_path.replace('.json', '_no_embeddings.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 임베딩이 제거된 파일 저장됨: {output_path}")
    print(f"📊 처리된 항목 수: {len(data)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        remove_embeddings(json_path)
    else:
        print("❌ JSON 파일 경로를 입력해주세요.")