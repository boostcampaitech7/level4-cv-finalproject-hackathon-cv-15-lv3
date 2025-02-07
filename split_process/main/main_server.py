import os
import math
import subprocess
from typing import List, Dict
import glob


script_folder='/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/split_process/sub' #Sub 서버에 보낼 스크립트가 들어있는 폴더
remote_path="/data/ephemeral/home"  # Sub 서버에서 처리될 Main 폴더
videos_dir = "/data/ephemeral/home/sample" #전송할 비디오 파일이 들어있는 폴더
ssh_key_path = "/data/ephemeral/home/CH_1.pem"  # CH_1.pem의 전체 경로로 수정해주세요
    

sub_name = "sub_server.py" 
remote_video_path=os.path.join(remote_path, "split_process_videos") #폴더명을 변경하려면, Sub 스크립트에서도 변경해주어야합니다.
remote_json_path=os.path.join(remote_path, "split_process_json")
remote_script_path=os.path.join(remote_path, "split_process_script")
sub_script_file = os.path.join(remote_script_path, sub_name)
file_list = glob.glob(f"{script_folder}/*")



class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username
        
def get_video_files(videos_dir: str) -> List[str]:
    """videos 디렉토리에서 모든 mp4 파일 목록을 가져옵니다."""
    return [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

def distribute_files(files: List[str], num_servers: int) -> Dict[int, List[str]]:
    """파일들을 서버 수에 맞게 균등하게 분배합니다."""
    files_per_server = math.ceil(len(files) / num_servers)
    distribution = {}
    
    for i in range(num_servers):
        start_idx = i * files_per_server
        end_idx = min((i + 1) * files_per_server, len(files))
        distribution[i] = files[start_idx:end_idx]
    
    return distribution

def scp_transfer(source_path: str, server: ServerInfo, ssh_key_path: str) -> bool:
    """SCP를 사용하여 파일을 전송합니다."""
    try:
        cmd = [
            'scp',
            '-i', ssh_key_path,  # SSH 키 파일 경로
            '-P', str(server.port),
            source_path,
            f'{server.username}@{server.ip}:{remote_video_path}'
        ]
        subprocess.run(cmd, check=True)
        
        cmd = [
            'scp',
            '-i', ssh_key_path,  # SSH 키 파일 경로
            '-P', str(server.port),
            ssh_key_path,
            f'{server.username}@{server.ip}:{ssh_key_path}'
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"전송 실패: {source_path} -> {server.ip}")
        print(f"에러: {str(e)}")
        return False

def run_scene_splitter(server: ServerInfo, ssh_key_path: str) -> bool:
    """원격 서버에서 스크립트를 전송 후 실행합니다."""
    try:
        script_cmd = [
            'scp',
            '-i', ssh_key_path,
            '-P', str(server.port)
        ] + file_list +  [f'{server.username}@{server.ip}:{remote_script_path}']
        subprocess.run(script_cmd, check=True)
        # 필요한 패키지 설치 및 스크립트 실행
        cmd = [
            'ssh',
            '-i', ssh_key_path,
            '-p', str(server.port),
            f'{server.username}@{server.ip}',
            f'/opt/conda/bin/python {sub_script_file}'  # 스크립트 실행
        ]
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"scene_splitter 실행 실패 - 서버: {server.ip}")
        print(f"에러: {str(e)}")
        return False

def main():
    # SSH 키 파일의 전체 경로 지정
    # 먼저 키 파일 권한 수정
    try:
        os.chmod(ssh_key_path, 0o600)
    except Exception as e:
        print(f"키 파일 권한 수정 실패: {str(e)}")
        return
    
    # 서버 정보 설정
    servers = [
        ServerInfo(
            ip="10.28.224.47",
            port=30767,
            username="root",
        ),
        ServerInfo(
            ip="10.28.224.149",
            port=31596,
            username="root",
        ),
        ServerInfo(
            ip="10.28.224.153",
            port=31699,
            username="root",
        )
    ]
    
    # 원격 서버에 디렉토리 생성 명령 추가
    def create_remote_directory(server: ServerInfo) -> bool:
        try:
            cmd = [
                'ssh',
                '-i', ssh_key_path,
                '-p', str(server.port),
                f'{server.username}@{server.ip}',
                f'mkdir -p {remote_video_path}'
            ]
            subprocess.run(cmd, check=True)
            cmd = [
                'ssh',
                '-i', ssh_key_path,
                '-p', str(server.port),
                f'{server.username}@{server.ip}',
                f'mkdir -p {remote_json_path}'
            ]
            subprocess.run(cmd, check=True)
            cmd = [
                'ssh',
                '-i', ssh_key_path,
                '-p', str(server.port),
                f'{server.username}@{server.ip}',
                f'mkdir -p {remote_script_path}'
            ]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"원격 디렉토리 생성 실패: {server.ip}")
            print(f"에러: {str(e)}")
            return False
    
    # 각 서버에 디렉토리 생성
    for server in servers:
        if not create_remote_directory(server):
            print(f"서버 {server.ip}에 디렉토리 생성 실패")
            return
    
    video_files = get_video_files(videos_dir)
    
    if not video_files:
        print("처리할 비디오 파일이 없습니다.")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # 파일 분배 (현재 서버 포함 4개로 분배)
    distribution = distribute_files(video_files, 4)
    
    # 현재 서버용 파일들은 이동하지 않음
    local_files = distribution[0]
    print(f"\n현재 서버에서 처리할 파일들: {len(local_files)}개")
    for file in local_files:
        print(f"- {file}")
    
    # 나머지 서버들에 파일 전송
    for server_idx, server in enumerate(servers, 1):
        files_to_transfer = distribution[server_idx]
        print(f"\n서버 {server.ip}:{server.port}로 전송할 파일들: {len(files_to_transfer)}개")
        
        success_count = 0
        for file in files_to_transfer:
            source_path = os.path.join(videos_dir, file)
            print(f"전송 중: {file} -> {server.ip}:{server.port}")
            
            if scp_transfer(source_path, server, ssh_key_path):
                success_count += 1
                
        print(f"서버 {server.ip} 전송 완료: {success_count}/{len(files_to_transfer)} 성공")
        
        # 파일 전송이 완료되면 scene_splitter 실행
        print(f"\n서버 {server.ip}에서 scene_splitter 실행 중...")
        if run_scene_splitter(server, ssh_key_path):
            print(f"서버 {server.ip}의 scene_splitter 실행 완료")
        else:
            print(f"서버 {server.ip}의 scene_splitter 실행 실패")

if __name__ == "__main__":
    main() 