from utils import save_video, read_video
from trackers import PlayerTracker,BallTracker
from court_line_detection import CourtLineDetection

def main():
    #Read video
    input_video_path=('input_videos/input_video.mp4')
    video_frames=read_video(input_video_path)

    #Detect players
    player_tracker=PlayerTracker('models/yolov8x')
    player_detections=player_tracker.detect_frames(video_frames,read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')

    #Detect ball
    ball_tracker=BallTracker('models/best.pt')
    ball_detections=ball_tracker.detect_frames(video_frames,read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')

    #Detect Court Lines
    court_line_detector=CourtLineDetection('models/keypoints_model.pth')
    court_keypoints=court_line_detector.predict(video_frames[0])

    #Choose players
    player_detections=player_tracker.filter_players(court_keypoints,player_detections)

    #Draw Output
    output_video_frames=player_tracker.draw_detection(video_frames,player_detections)
    output_video_frames=ball_tracker.draw_detection(output_video_frames,ball_detections)
    output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__=="__main__":
    main()