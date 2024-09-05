from ultralytics import YOLO
import cv2
import pickle
from utils import get_bbox_width,get_center_of_bbox
import numpy as np
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)


    def interpolate_ball_positions(self,ball_positions):
        ball_positions=[x.get(1,[]) for x in ball_positions]
        df_ball_positions=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        
        #Interpolate missing values
        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions=df_ball_positions.bfill()

        ball_positions=[{1:x}for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    
    def detect_frames(self,frames,read_from_stub=False, stub_path=None):
        ball_detections=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb') as f:
                ball_detections=pickle.load(f)
            return ball_detections

        for frame in frames:
            result=self.detect_frame(frame)
            ball_detections.append(result)
        ball_detections=self.interpolate_ball_positions(ball_detections)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(ball_detections,f)

        return ball_detections


    def detect_frame(self,frame):
        results=self.model.track(frame, conf=0.15)[0]
        ball_dict={}
        for box in results.boxes:
            result=box.xyxy.tolist()[0]
            ball_dict[1]=result
        return ball_dict
    
    def draw_detection(self,video_frames,ball_detections):
        output_video_frames=[]
        for frame,ball_dict in zip(video_frames,ball_detections):
            for track_id,bbox in ball_dict.items():
                y=int(bbox[1])
                x,_=get_center_of_bbox(bbox)

                triangle_points=np.array([
                    [x,y],
                    [x-10,y-20],
                    [x+10,y-20]
                ])

                cv2.drawContours(frame, [triangle_points], 0, (0,255,255), cv2.FILLED)
                cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
                
            output_video_frames.append(frame)

        return output_video_frames