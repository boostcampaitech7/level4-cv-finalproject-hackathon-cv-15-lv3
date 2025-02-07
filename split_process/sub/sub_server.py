
import os
import math
import subprocess
from typing import List, Dict
from sub_server_process import process, output_file

process()

class ServerInfo:
    def __init__(self, ip: str, port: int, username: str, remote_path: str):
        self.ip = ip
        self.port = port
        self.username = username
        self.remote_path = remote_path

ssh_key_path='/data/ephemeral/home/CH_1.pem'
try:
    os.chmod(ssh_key_path, 0o600)
except Exception as e:
    print(f"키 파일 권한 수정 실패: {str(e)}")

server=ServerInfo(
            ip="10.28.224.178",
            port=32289,
            username="root",
            remote_path="/data/ephemeral/home/json"  # 경로 수정
        )

cmd = [
    'ssh','-o', 'StrictHostKeyChecking=no',
    '-i', ssh_key_path,
    '-p', str(server.port),
    f'{server.username}@{server.ip}',
    f'mkdir -p {server.remote_path}'
]
subprocess.run(cmd, check=True)

cmd = [
    'scp',
    '-i', ssh_key_path,  # SSH 키 파일 경로
    '-P', str(server.port),
    output_file,
    f'{server.username}@{server.ip}:{server.remote_path}'
]

subprocess.run(cmd, check=True)

