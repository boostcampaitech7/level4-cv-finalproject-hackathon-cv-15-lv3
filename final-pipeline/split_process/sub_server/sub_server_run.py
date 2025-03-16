
import os
import math
import subprocess
from typing import List, Dict
from sub_server_process import process
from config import Config
# 고유한 파일명 생성
process() #sub_server_process 실행

class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username
server=ServerInfo(
            ip="your_ip_here",
            port="your_port_here",
            username="your_username_here",
        )

try:
    os.chmod(Config.ssh_key_path, 0o600)
except Exception as e:
    print(f"키 파일 권한 수정 실패: {str(e)}")

#output 폴더 생성
cmd = [
    'ssh','-o', 'StrictHostKeyChecking=no',
    '-i', Config.ssh_key_path,
    '-p', str(server.port),
    f'{server.username}@{server.ip}',
    f'mkdir -p {Config.remote_path}'
]
subprocess.run(cmd, check=True)


#output 파일 전송
cmd = [
    'scp',
    '-i', Config.ssh_key_path,  # SSH 키 파일 경로
    '-P', str(server.port),
    Config.output_file,
    f'{server.username}@{server.ip}:{Config.remote_path}'
]
subprocess.run(cmd, check=True)

