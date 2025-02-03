import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def time_overlap(gt_start, gt_end, entry_start, entry_end):
    """겹치는 시간 계산"""
    overlap_start = max(gt_start, entry_start)
    overlap_end = min(gt_end, entry_end)
    return max(0, overlap_end - overlap_start)  # 겹치는 시간이 없으면 0


def load_criteria_from_excel(excel_path, db_data):
    """엑셀 파일에서 기준 데이터를 로드하여 DB와 매칭"""
    print("엑셀 파일에서 기준 데이터를 로드 중...")

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"엑셀 파일을 읽는 중 오류 발생: {e}")
        exit(1)

    criteria_data = []
    unmatched_entries = 0

    print(f"총 {len(df)}개의 기준 데이터 확인 완료. DB에서 매칭을 시작합니다...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="기준 데이터 매칭 중"):
        video_id = row['VideoURL']
        gt_start = row['StartTime']
        gt_end = row['EndTime']

        # DB에서 video_id와 url이 일치하는 항목 찾기
        matched_entries = [
            entry for entry in db_data
            if entry["video_id"] == video_id and entry["caption"] is not None
        ]

        if not matched_entries:
            unmatched_entries += 1
            continue  # 일치하는 항목이 없는 경우 스킵

        # start_time & end_time이 겹치는 정도가 가장 큰 항목 선택
        best_match = max(
            matched_entries,
            key=lambda e: time_overlap(gt_start, gt_end, float(e["start_time"]), float(e["end_time"]))
        )

        # 최종 기준 데이터에 추가
        criteria_data.append(best_match)

    print(f"기준 데이터 추출 완료: {len(criteria_data)}개 항목 선택 (매칭 실패 {unmatched_entries}개 제외)")
    return criteria_data


def cosine_similarity_gpu(emb1, emb2):
    """GPU 기반 코사인 유사도 계산"""
    emb1 = torch.tensor(emb1, device="cuda") if not torch.is_tensor(emb1) else emb1
    emb2 = torch.tensor(emb2, device="cuda") if not torch.is_tensor(emb2) else emb2

    emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)

    return torch.mm(emb1_norm, emb2_norm.T).cpu()


def select_similar_data_based_on_criteria_gpu(criteria_data, db_data, embedding_key="embedding", top_k=6):
    """기준에 따라 유사도가 높은 데이터 선택 (GPU 활용)"""
    print("기준 데이터를 바탕으로 유사도가 높은 데이터 찾기 시작 (GPU 사용)...")

    selected_data = []

    # DB 임베딩을 미리 텐서로 변환하여 GPU로 옮김
    db_embeddings = torch.tensor(
        [entry[embedding_key] for entry in db_data if embedding_key in entry and entry["caption"] is not None],
        device="cuda"
    )
    db_entries = [entry for entry in db_data if embedding_key in entry and entry["caption"] is not None]

    for criterion in tqdm(criteria_data, desc="유사도 계산 중"):
        if embedding_key not in criterion:
            print(f"경고: {criterion['video_id']}의 임베딩 정보가 없습니다. 스킵합니다.")
            continue

        criterion_embedding = torch.tensor(criterion[embedding_key], device="cuda").unsqueeze(0)

        # GPU에서 유사도 계산
        similarities = cosine_similarity_gpu(criterion_embedding, db_embeddings)
        similarities = similarities.squeeze(0).tolist()

        # 유사도가 높은 항목 선택
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

        for idx in top_k_indices:
            entry = db_entries[idx]
            selected_entry = {
                "video_path": entry.get("video_path", ""),
                "video_id": entry.get("video_id", ""),
                "video_title": entry.get("video_title", ""),
                "video_url": entry.get("video_url", ""),
                "start_time": entry.get("start_time", ""),
                "end_time": entry.get("end_time", ""),
                "caption": entry.get("caption", "")
            }
            selected_data.append(selected_entry)

    print(f"유사도 기반 샘플링 완료: {len(selected_data)}개 항목 선택됨")
    return selected_data


def save_gt_db_as_json_and_csv(selected_data, json_output_path, csv_output_path):
    """GT-DB를 JSON 및 CSV 형식으로 저장"""
    print(f"{len(selected_data)}개의 데이터 저장 중...")

    # JSON 저장
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(selected_data, json_file, ensure_ascii=False, indent=4)

    # CSV 저장
    with open(csv_output_path, "w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["video_path", "video_id", "video_title", "video_url", "start_time", "end_time", "caption"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in selected_data:
            writer.writerow({
                "video_path": entry.get("video_path", ""),
                "video_id": entry.get("video_id", ""),
                "video_title": entry.get("video_title", ""),
                "video_url": entry.get("video_url", ""),
                "start_time": entry.get("start_time", ""),
                "end_time": entry.get("end_time", ""),
                "caption": entry.get("caption", "")
            })

    print(f"JSON 파일 저장 완료: {json_output_path}")
    print(f"CSV 파일 저장 완료: {csv_output_path}")


# 파일 경로 설정
excel_path = "/data/ephemeral/home/data/json_DB_v1/evaluation_dataset_v2.xlsx"
json_output_path = "./json/gt_db_sampled_v1_stella.json"
csv_output_path = "./csv/gt_db_sampled_v1_stella.csv"

db_data_path = "/data/ephemeral/home/data/json_DB_v1/caption_embedding_stella.json"
print("DB 데이터 로드 중...")
try:
    with open(db_data_path, "r", encoding="utf-8") as db_file:
        db_data = json.load(db_file)
    print(f"DB 데이터 로드 완료: {len(db_data)}개 항목")
except Exception as e:
    print(f"DB 데이터를 로드하는 중 오류 발생: {e}")
    exit(1)

# 엑셀에서 기준 데이터 추출
criteria_data = load_criteria_from_excel(excel_path, db_data)

# 데이터 선택 및 저장
selected_data = select_similar_data_based_on_criteria_gpu(criteria_data, db_data)
save_gt_db_as_json_and_csv(selected_data, json_output_path, csv_output_path)