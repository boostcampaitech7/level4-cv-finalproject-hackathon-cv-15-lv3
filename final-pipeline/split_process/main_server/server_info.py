class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username

SERVERS = [
    ServerInfo("10.28.224.149", 31655, "root"),
    ServerInfo("10.28.224.47", 30767, "root"),
    ServerInfo("10.28.224.118", 32740, "root"),
    ServerInfo("10.28.224.178", 32289, "root")
]
