import gdown
import os

def download_files():
    # Video url 
    video_son = "1vVwjW1dE1dE1drIdd4ZSILfbCGPD4weoNiu"
    video_path = "/content/data/video.avi"

    # Model url
    player_goalkeeper_url = "https://drive.google.com/uc?id=1jkzZJN0y91Hs9hkaJXYrkkOegcuBhE2D"
    player_path = "/content/models/player_model.pt"
    ball_url = "https://drive.google.com/uc?id=1L1QoAQ-IwyI_yCM3BmpX52nxM4ASsrSq"
    ball_path = "/content/models/ball_model.pt"
    
    gdown.download(f"https://drive.google.com/uc?id={video_son}", video_path, quiet=False)
    print("Video đã được tải về tại:", video_path)
  
    gdown.download(player_goalkeeper_url, player_path, quiet=False)
    print("Mô hình phát hiện cầu thủ đã tải tại:", player_path)

    gdown.download(ball_url, ball_path, quiet=False)
    print("Mô hình phát hiện bóng đã tải tại:", ball_path)


'''
file_id_full = "1Kt_budq2pRdU3etjqi99mlKqHItRuH6t"
file_id_30s = "1-0Y8HF4bt3oT6yE5B5O1PEgEmPR0Pid_"
file_id_10s = "1-CjZGOzpcrvHyZN9Uzs8wz3eGC6IhCf1"
file_id_2s = "1-HJI7MjuyUX5fZzgKSj3kq_nw181zUzC"
video2_30s = "14tDZKljc5shcuR4oHRLsAC_51CFz89Tw"
video_son = "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
video_path = "/content/video.avi"
'''
