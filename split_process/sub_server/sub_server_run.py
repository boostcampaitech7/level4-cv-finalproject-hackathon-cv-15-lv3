import os
import subprocess
from typing import NoReturn
from sub_server_process import process
from config import Config
from utils import set_ssh_key_permissions, create_remote_directory, transfer_file_to_server

class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username

server = ServerInfo(ip="10.28.224.178", port=32289, username="root")

def main() -> NoReturn:
    """프로그램의 주요 실행 흐름을 정의합니다."""
    process()  # sub_server_process 실행
    set_ssh_key_permissions()
    create_remote_directory(server)
    transfer_file_to_server(server)

if __name__ == "__main__":
    main()
