from ultralytics import YOLO
import cv2
import pickle
from utils import get_bbox_width,get_center_of_bbox,measure_distance

class PlayerTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)

    
    def filter_players(self,court_keypoints,player_detections):
        player_detections_first_frame=player_detections[0]
        chosen_players=self.choose_players(court_keypoints,player_detections_first_frame)
        filtered_players_detections=[]

        for player_dict in player_detections:
            filtered_players_dict={track_id: bbox for track_id,bbox in player_dict.items() if track_id in chosen_players}
            filtered_players_detections.append(filtered_players_dict)
        return filtered_players_detections
    

    def choose_players(self,court_keypoints,player_detections):
        distances=[]
        for track_id, bbox in player_detections.items():
            player_center=get_center_of_bbox(bbox)
            min_distance=float('inf')

            for i in range(0,len(court_keypoints),2):
                court_keypoint=(court_keypoints[i],court_keypoints[i+1])
                distance=measure_distance(court_keypoint, player_center)
                if distance<min_distance:
                    min_distance=distance
            
            distances.append((track_id,min_distance))
        
        distances.sort(key=lambda x:x[1])
        chosen_players=[distances[0][0],distances[1][0]]
        return chosen_players



    def detect_frames(self,frames,read_from_stub=False, stub_path=None):
        player_detections=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb') as f:
                player_detections=pickle.load(f)
            return player_detections

        for frame in frames:
            result=self.detect_frame(frame)
            player_detections.append(result)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(player_detections,f)

        return player_detections


    def detect_frame(self,frame):
        results=self.model.track(frame, persist=True)[0]
        id_name_dict=results.names
        player_dict={}
        for box in results.boxes:
            track_id=int(box.id.tolist()[0])
            result=box.xyxy.tolist()[0]
            cls_id=box.cls.tolist()[0]
            cls_name=id_name_dict[cls_id]
            if cls_name=="person":
                player_dict[track_id]=result
        return player_dict
    
    def draw_detection(self,video_frames,player_detections):
        output_video_frames=[]
        for frame,player_dict in zip(video_frames,player_detections):
            for track_id,bbox in player_dict.items():
                x1,y1,x2,y2=bbox
                x_center, _ = get_center_of_bbox(bbox)
                width = get_bbox_width(bbox)
                cv2.ellipse(frame, center=(int(x_center),int(y2)), axes=(int(width),int(0.35*width)), angle=0.0, startAngle=-45, endAngle=235, color = (0,255,0), thickness=2, lineType=cv2.LINE_4)
                
                rectangle_width=40
                rectangle_height=20
                x1_rect=x_center-rectangle_width//2
                x2_rect=x_center+rectangle_width//2
                y1_rect=(y2-rectangle_height//2)+15
                y2_rect=(y2+rectangle_height//2)+15

                cv2.rectangle(
                    frame,
                    (int(x1_rect),int(y1_rect)),
                    (int(x2_rect),int(y2_rect)),
                    (0,255,0),
                    cv2.FILLED
                )

                x1_text=x1_rect+12

                cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(x1_text),int(y1_rect+15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2
                )
            output_video_frames.append(frame)

        return output_video_frames