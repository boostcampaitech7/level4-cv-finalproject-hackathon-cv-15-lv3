
import os
import math
import subprocess
from typing import List, Dict
from sub_server_process import process
import socket

server_ip = socket.gethostbyname(socket.gethostname())

# 고유한 파일명 생성

ssh_key_path='/data/ephemeral/home/CH_1.pem' 
video_dir = "/data/ephemeral/home/split_process_videos"
remote_path= "/data/ephemeral/home/json" #메인서버에 생성할 josn 폴더
output_file = f"/data/ephemeral/home/split_process_json/video_files_{server_ip}.txt" #메인서버에 생성할 josn 파일이름

process() #sub_server_process 실행

class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username
server=ServerInfo(
            ip="10.28.224.178",
            port=32289,
            username="root",
        )

try:
    os.chmod(ssh_key_path, 0o600)
except Exception as e:
    print(f"키 파일 권한 수정 실패: {str(e)}")


cmd = [
    'ssh','-o', 'StrictHostKeyChecking=no',
    '-i', ssh_key_path,
    '-p', str(server.port),
    f'{server.username}@{server.ip}',
    f'mkdir -p {remote_path}'
]
subprocess.run(cmd, check=True)

cmd = [
    'scp',
    '-i', ssh_key_path,  # SSH 키 파일 경로
    '-P', str(server.port),
    output_file,
    f'{server.username}@{server.ip}:{remote_path}'
]
subprocess.run(cmd, check=True)

