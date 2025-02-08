import socket

server_ip = socket.gethostbyname(socket.gethostname())

class Config:
    ssh_key_path='/data/ephemeral/home/CH_1.pem' 
    video_dir = "/data/ephemeral/home/split_process/split_process_videos" #비디오가 저장될 폴더(메인 서버에서 만들어준 폴더 이름)
    remote_path= "/data/ephemeral/home/json" #메인서버에 생성할 josn 폴더
    output_file = f"/data/ephemeral/home/split_process/split_process_json/video_files_{server_ip}.json" #메인서버에 생성할 json 파일이름
