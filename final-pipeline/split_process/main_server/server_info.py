class ServerInfo:
    def __init__(self, ip: str, port: int, username: str):
        self.ip = ip
        self.port = port
        self.username = username

SERVERS = [
    ServerInfo("your_ip_here", "your_port_here", "your_username_here"),
    ServerInfo("your_ip_here", "your_port_here", "your_username_here"),
    ServerInfo("your_ip_here", "your_port_here", "your_username_here"),
    ServerInfo("your_ip_here", "your_port_here", "your_username_here")
]
