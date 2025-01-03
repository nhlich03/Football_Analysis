import cv2
import os
import numpy as np

from team_classifier import TeamClassifier
from utils.bbox_utils import convert_xywh_to_xyxy, get_distance
from utils.video_utils import read_video


class ProcessVideo:
    def __init__(self):
        self.colors = {
            0: (0, 0, 255),  # Ball - đỏ
            2: (0, 255, 0),  # Player - xanh lá
            3: (255, 0, 0),  # Referee - xanh dương
            4: (255, 255, 0),  # Team1 Vàng
            5: (255, 0, 255)  # Team2 Hồng
        }
        self.possession_count = {'Team1': 0, 'Team2': 0}
        self.team_classifier = None
        self.threshold = 100
        self.output_frames = []

    def process_video(self, video_path, output_video_path, detect_results_dir):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        video_frames = read_video(video_path)
        for frame_num, frame in enumerate(video_frames):
            frame_dir = os.path.join(detect_results_dir, f"frame_{frame_num}")
            labels_file_path = os.path.join(frame_dir, "player", "labels", "image0.txt")

            if not os.path.exists(labels_file_path):
                continue

            processed_frame = self.process_frame(frame, labels_file_path)
            self.output_frames.append(processed_frame)
            out.write(processed_frame)

    def process_frame(self, frame, labels_file_path):
        ball_box = None
        team1_players = []
        team2_players = []
        bounding_boxes_info = []
        frame_height, frame_width, _ = frame.shape

        with open(labels_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                cls, x, y, w, h, conf = map(float, parts)
                bounding_box = [x, y, w, h]
                object_type = cls

                # If player, then split to team1 and team2 
                if cls == 2:
                    x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bounding_box, frame.shape)
                    player_crop = frame[y_min:y_max, x_min:x_max]
                    player_img_array = np.array(player_crop)
                    team_id = self.team_classifier.get_player_team(player_img_array)

                    if team_id == 1:
                        team1_players.append(bounding_box)
                        object_type = 4
                    elif team_id == 2:
                        team2_players.append(bounding_box)
                        object_type = 5

                elif cls == 0:
                    ball_box = bounding_box

                # Get color for bounding box
                color = self.get_color(object_type)

                # Draw bounding box to frame
                self.draw_bbox(frame, bounding_box, conf, color)

                # Draw possession
                self.draw_possession_info(frame, self.possession_count)

                # If ball exist, assign ball to player and draw ball connection
                if ball_box is not None:
                    self.assign_ball_player(frame, ball_box, team1_players, team2_players, self.threshold)

        return frame

    def draw_bbox(self, frame, bounding_box, conf, color):
        x1, x2, y1, y2 = convert_xywh_to_xyxy(bounding_box, frame.shape)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)

        # Show confidence index
        conf_text = f'{conf:.2f}'
        cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)

    def draw_possession_info(self, frame, possession_count):
        """
        Draw possession information on the frame.

        :param frame: Current frame of the video.
        :param possession_count: Dictionary containing the number of ball possessions for each team.
        """
        total = sum(possession_count.values())
        team1_percent = possession_count['Team1'] / total * 100 if total > 0 else 0
        team2_percent = possession_count['Team2'] / total * 100 if total > 0 else 0

        # Get colors for Team1 and Team2
        team1_color = self.get_color(4)
        team2_color = self.get_color(5)

        # Draw Team1 percentage
        cv2.putText(frame, f"Team1: {team1_percent:.1f}%",
                    (10, frame.shape[0] - 40),  # Adjust y-coordinate for two lines
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team1_color, 2)

        # Draw Team2 percentage
        cv2.putText(frame, f"Team2: {team2_percent:.1f}%",
                    (10, frame.shape[0] - 10),  # Lower y-coordinate for Team2
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team2_color, 2)

    def assign_ball_player(self, frame, ball_box, team1_players, team2_players, threshold):
        min_distance = float('inf')
        team_id = -1
        assigned_player_box = None

        for player_box in team1_players:
            distance = get_distance(player_box, ball_box, frame.shape)
            if distance < min_distance:
                min_distance = distance
                team_id = 4
                assigned_player_box = player_box

        for player_box in team2_players:
            distance = get_distance(player_box, ball_box, frame.shape)
            if distance < min_distance:
                min_distance = distance
                team_id = 5
                assigned_player_box = player_box

        if min_distance <= threshold:
            color = self.get_color(team_id)
            self.draw_ball_connection(frame, ball_box, assigned_player_box, color)

            if team_id == 4:
                self.possession_count['Team1'] += 1
            elif team_id == 5:
                self.possession_count['Team2'] += 1

    def draw_ball_connection(self, frame, ball_box, player_box, color):
        """
        Draw a line connecting the ball to the player.

        :param frame: Current frame of the video.
        :param ball_box: Bounding box of the ball [x, y, w, h].
        :param player_box: Bounding box of the player [x, y, w, h].
        :param color: Color of the connecting line.
        """
        bx, by, bw, bh = ball_box
        px, py, pw, ph = player_box

        # Calculate the center point of the ball and player from the given bounding box
        ball_center = (int(bx * frame.shape[1]), int(by * frame.shape[0]))
        player_center = (int(px * frame.shape[1]), int(py * frame.shape[0]))

        cv2.line(frame, player_center, ball_center, color, 2)

    def is_ball_with_player(self, ball_box, player_box, threshold=0.01):
        """
        Check if the ball is close to any player.

        :param ball_box: Bounding box of the ball [x, y, w, h].
        :param player_box: Bounding box of the player [x, y, w, h].
        :param threshold: Distance threshold to consider the ball as held.
        :return: Bounding box of the player holding the ball or None if no player is near the ball.
        """
        if ball_box is None:
            return None

        bx, by, _, _ = ball_box
        px, py, _, _ = player_box

        # Calculate distance between centers of the ball and player
        distance = ((bx - px)**2 + (by - py)**2)**0.5

        if distance < threshold:
            return player_box
        return None

    def get_color(self, object_type):
        color = self.colors.get(object_type, (255, 255, 255))
        return color