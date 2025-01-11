import hashlib
import itertools
import json
import os
from six.moves import urllib
import sys
import tensorflow as tf
from datetime import datetime

# 기존 다운로드 관련 함수들
def LetterRange(start, end):
    return list(map(chr, range(ord(start), ord(end) + 1)))

VOCAB = LetterRange('a', 'z') + LetterRange('A', 'Z') + LetterRange('0', '9')
file_ids = [''.join(i) for i in itertools.product(VOCAB, repeat=2)]
file_index = {f: i for (i, f) in enumerate(file_ids)}

def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as fin:
        for chunk in iter(lambda: fin.read(128 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def download_file(source_url, destination_path):
    def _progress(count, block_size, total_size):
        # 진행률을 stderr로 출력 (로그 파일에 저장되지 않음)
        sys.stderr.write('\r>> Downloading %s %.1f%%' % (
            source_url, float(count * block_size) / float(total_size) * 100.0))
        sys.stderr.flush()
    
    urllib.request.urlretrieve(source_url, destination_path, _progress)
    statinfo = os.stat(destination_path)
    # 다운로드 완료 메시지는 stdout으로 출력 (로그 파일에 저장됨)
    print(f'\nSuccessfully downloaded {destination_path}, {statinfo.st_size} bytes.')
    return destination_path

# 새로운 처리 관련 함수들
def load_movie_theater_ids(filename="id_movie.txt"):
    """Movie theater ID 목록 로드"""
    with open(filename, 'r') as f:
        content = f.read()
        # JSONP 형식에서 ID 목록 추출
        start = content.find('[')
        end = content.find(']')
        if start != -1 and end != -1:
            ids = content[start+1:end].replace('"', '').split(',')
            return set(ids)
    return set()

def process_tfrecord(tfrecord_path, movie_theater_ids):
    """TFRecord 파일에서 movie theater 관련 데이터만 추출"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    movie_theater_records = []
    total_records = 0
    
    print(f"\nProcessing: {tfrecord_path}")
    print(f"File size: {os.path.getsize(tfrecord_path) / (1024*1024):.2f} MB")
    print("Searching for movie theater videos...")
    
    for record in dataset:
        total_records += 1
        sequence_example = tf.train.SequenceExample()
        sequence_example.ParseFromString(record.numpy())
        
        video_id = sequence_example.context.feature['id'].bytes_list.value[0].decode('utf-8')
        
        if video_id in movie_theater_ids:
            print(f"Found movie theater video: {video_id}")
            record_data = {
                "context": {
                    "feature": [
                        {
                            "key": "id",
                            "value": {
                                "bytes_list": {
                                    "value": video_id
                                }
                            }
                        },
                        {
                            "key": "labels",
                            "value": {
                                "int64_list": {
                                    "value": list(sequence_example.context.feature['labels'].int64_list.value)
                                }
                            }
                        }
                    ]
                },
                "feature_lists": {
                    "feature_list": {}
                }
            }
            
            # rgb와 audio 특징 처리
            for feature_name in ['rgb', 'audio']:
                if feature_name in sequence_example.feature_lists.feature_list:
                    feature_list_data = {
                        "key": feature_name,
                        "value": {
                            "feature": []
                        }
                    }
                    
                    for feature in sequence_example.feature_lists.feature_list[feature_name].feature:
                        feature_list_data["value"]["feature"].append({
                            "bytes_list": {
                                "value": list(map(int, feature.bytes_list.value[0]))
                            }
                        })
                    
                    record_data["feature_lists"]["feature_list"][feature_name] = feature_list_data
            
            movie_theater_records.append(record_data)
    
    print(f"\nProcessed {total_records} total records")
    print(f"Found {len(movie_theater_records)} movie theater videos")
    return movie_theater_records

def save_records(records, tfrecord_name):
    """추출된 레코드를 JSON으로 저장"""
    if records:
        output_filename = f"movie_theater_{tfrecord_name}.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": tfrecord_name,
            "records": records
        }
        
        print(f"\nSaving movie theater records to {output_filename}...")
        with open(output_filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully saved {len(records)} records")
        print(f"JSON file size: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")
        
        # 처리 완료된 tfrecord 파일 삭제
        print(f"Removing processed tfrecord file: {tfrecord_name}")
        os.remove(tfrecord_name)
        print("File removed successfully")
    else:
        print(f"\nNo movie theater records found in {tfrecord_name}")
        print(f"Removing tfrecord file: {tfrecord_name}")
        os.remove(tfrecord_name)
        print("File removed successfully")

def main():
    if 'partition' not in os.environ:
        print('Must provide environment variable "partition". e.g. 2/frame/train', file=sys.stderr)
        exit(1)
    if 'mirror' not in os.environ:
        print('Must provide environment variable "mirror". e.g. "us"', file=sys.stderr)
        exit(1)

    # Movie theater ID 목록 로드
    movie_theater_ids = load_movie_theater_ids()
    print(f"Loaded {len(movie_theater_ids)} movie theater IDs")

    partition = os.environ['partition']
    mirror = os.environ['mirror']
    partition_parts = partition.split('/')

    assert mirror in {'us', 'eu', 'asia'}
    assert len(partition_parts) == 3
    assert partition_parts[1] in {'video_level', 'frame_level', 'video', 'frame'}
    assert partition_parts[2] in {'train', 'test', 'validate'}

    plan_url = f'http://data.yt8m.org/{partition_parts[0]}/download_plans/{partition_parts[1]}_{partition_parts[2]}.json'
    plan_filename = f'{partition.replace("/", "_")}_download_plan.json'

    if not os.path.exists(plan_filename):
        print('Starting fresh download...')
        download_file(plan_url, plan_filename)

    download_plan = json.loads(open(plan_filename).read())
    
    # 딕셔너리 키를 리스트로 변환하여 순회
    files_to_process = list(download_plan['files'].keys())
    
    for f in files_to_process:
        fname, ext = f.split('.')
        out_f = f'{str(fname[:-2])}{file_index[str(fname[-2:])]:04d}.{ext}'

        if os.path.exists(out_f) and md5sum(out_f) == download_plan['files'][f]:
            print(f'Skipping already downloaded file {out_f}')
            continue

        print(f'Downloading: {out_f}')
        download_url = f'http://{mirror}.data.yt8m.org/{partition}/{f}'
        
        try:
            # 파일 다운로드
            download_file(download_url, out_f)
            
            if md5sum(out_f) == download_plan['files'][f]:
                print(f'Successfully downloaded {out_f}')
                
                # 다운로드된 파일 즉시 처리
                try:
                    print(f"\nStarting to process {out_f}...")  # 디버깅용 출력 추가
                    records = process_tfrecord(out_f, movie_theater_ids)
                    save_records(records, out_f)
                    
                    # 다운로드 계획에서 처리된 파일 제거
                    del download_plan['files'][f]
                    with open(plan_filename, 'w') as plan_file:
                        json.dump(download_plan, plan_file)
                        
                except Exception as e:
                    print(f"Error processing {out_f}: {str(e)}")
            else:
                print(f'Error downloading {f}. MD5 does not match!')
                if os.path.exists(out_f):
                    os.remove(out_f)
                    
        except Exception as e:
            print(f"Error during download/process of {out_f}: {str(e)}")
            if os.path.exists(out_f):
                os.remove(out_f)

    print('All done. No more files to download.')

if __name__ == '__main__':
    main()