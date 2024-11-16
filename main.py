from utils import (read_video, save_video, measure_dist,convert_px_to_m,convert_m_to_px,draw_player_stats)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from minicourt import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd
import constants
def main():
    #READING THE VIDEO
    input_video_path="input_videos/input_video.mp4"
    video_frames=read_video(input_video_path)
    #PLAYER AND BALL DETECTION

    player_tracker=PlayerTracker(model_path='yolov8x')
    ball_tracker=BallTracker(model_path='models/best.pt')
    player_detections=player_tracker.detect_frames(video_frames,read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections=ball_tracker.detect_frames(video_frames,read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections=ball_tracker.interpolate_ball_pos(ball_detections)

    #COURT LINE DETECTION
    court_model_path="models/keypoints_model.pth"
    court_line_detector=CourtLineDetector(court_model_path)
    court_keypoints=court_line_detector.predict(video_frames[1]) #Camera is steady so no need to send multiple frames

    #PLAYERS IDENTIFICATION USING EUCLIDEAN DIST
    player_detections=player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    #DISPLAY OUTPUT
    mini_court=MiniCourt(video_frames[0])

    #BALL SHOT FRAMES
    ball_shot_frames=ball_tracker.get_ball_hit_frames(ball_detections)
    # print(ball_shot_frames)

    #CONVERT POS TO MINICOURT POS
    player_mini_court_detections,ball_mini_court_detections= mini_court.convert_bbox_to_minicourt_coords(player_detections,ball_detections, court_keypoints)

    player_stats_data=[
        {   
            'frame_no':0,
            'player_1_number_of_shots':0,
            'player_1_total_shot_speed':0,
            'player_1_last_shot_speed':0,
            'player_1_total_player_speed':0,
            'player_1_last_player_speed':0,

            'player_2_number_of_shots':0,
            'player_2_total_shot_speed':0,
            'player_2_last_shot_speed':0,
            'player_2_total_player_speed':0,
            'player_2_last_player_speed':0,
        }
    ]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame=ball_shot_frames[ball_shot_ind]
        end_frame=ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_sec=(end_frame-start_frame)/24

        distance_covered_by_ball_in_px=measure_dist(ball_mini_court_detections[start_frame][1],ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_in_m=convert_px_to_m(distance_covered_by_ball_in_px, constants.DOUBLE_LINE_WIDTH, mini_court.width_of_mini_court())
        speed_of_ball=(distance_covered_by_ball_in_m*3600)/(ball_shot_time_in_sec*1000)   
        #PLAYER WITH BALL 
        player_positions=player_mini_court_detections[start_frame]
        player_shot=min(player_positions.keys(),key=lambda player_id: measure_dist(player_positions[player_id],ball_mini_court_detections[start_frame][1]))

        opponent_id=1 if player_shot==2 else 2

        distance_covered_by_opponent_in_px=measure_dist(player_mini_court_detections[start_frame][opponent_id],player_mini_court_detections[end_frame][opponent_id])
        distance_covered_by_opponent_in_m=convert_px_to_m(distance_covered_by_opponent_in_px, constants.DOUBLE_LINE_WIDTH, mini_court.width_of_mini_court())
        speed_of_opponent=(distance_covered_by_opponent_in_m*3600)/(ball_shot_time_in_sec*1000)   

        current_player_stats=deepcopy(player_stats_data[-1])
        current_player_stats['frame_num']=start_frame
        current_player_stats[f'player_{player_shot}_number_of_shots']+=1
        current_player_stats[f'player_{player_shot}_total_shot_speed']+=speed_of_ball
        current_player_stats[f'player_{player_shot}_last_shot_speed']=speed_of_ball

        current_player_stats[f'player_{opponent_id}_total_player_speed']+=speed_of_opponent
        current_player_stats[f'player_{opponent_id}_last_player_speed']=speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df=pd.DataFrame(player_stats_data)
    frames_df=pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df=pd.merge(frames_df,player_stats_data_df,on='frame_num',how='left')
    player_stats_data_df=player_stats_data_df.ffill()

    player_stats_data_df['player_1_avg_shot_speed']=player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_avg_shot_speed']=player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_avg_player_speed']=player_stats_data_df['player_1_total_player_speed']/(player_stats_data_df['player_2_number_of_shots'])
    player_stats_data_df['player_2_avg_player_speed']=player_stats_data_df['player_2_total_player_speed']/(player_stats_data_df['player_1_number_of_shots'])

    #BBOXES
    output_video_frames=player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames=ball_tracker.draw_bboxes(output_video_frames,ball_detections)

    output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    output_video_frames=mini_court.draw_mini_court(output_video_frames)
    output_video_frames=mini_court.draw_points_on_minicourt(output_video_frames,player_mini_court_detections)
    output_video_frames=mini_court.draw_points_on_minicourt(output_video_frames,ball_mini_court_detections,color=(200,200,50))
     
    output_video_frames=draw_player_stats(output_video_frames,player_stats_data_df)
    
    #FRAME NUMBER
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame,f"Frame:{i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    save_video(output_video_frames,"output_videos/output_video.avi")

if __name__ == "__main__":
    main()