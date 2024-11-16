from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
class BallTracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)

    def interpolate_ball_pos(self,ball_pos):
        ball_pos=[x.get(1,[]) for x in ball_pos]
        df_ball_pos=pd.DataFrame(ball_pos,columns=['x1','y1','x2','y2']) #LISTS TO DF

        df_ball_pos=df_ball_pos.interpolate()
        df_ball_pos=df_ball_pos.bfill() #TO HANDLE NULL DETECTION IN FIRST FRAME

        ball_pos=[{1:x} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos

    def get_ball_hit_frames(self,ball_pos):
        ball_pos=[x.get(1,[]) for x in ball_pos]
        df_ball_pos=pd.DataFrame(ball_pos,columns=['x1','y1','x2','y2']) #LISTS TO DF

        df_ball_pos['ball_hit']=0
        df_ball_pos['mid_y']=(df_ball_pos['y1']+df_ball_pos['y2'])/2
        df_ball_pos['mid_y_rolling_mean']=df_ball_pos['mid_y'].rolling(window=5,min_periods=1,center=False).mean()
        df_ball_pos['delta_y']=df_ball_pos['mid_y_rolling_mean'].diff()


        min_change_frames_for_hit=25
        for i in range (1,len(df_ball_pos)-int(min_change_frames_for_hit*1.2)):
            neg_pos_change=df_ball_pos['delta_y'].iloc[i]>0 and df_ball_pos['delta_y'].iloc[i+1]<0
            pos_pos_change=df_ball_pos['delta_y'].iloc[i]<0 and df_ball_pos['delta_y'].iloc[i+1]>0

            if neg_pos_change or pos_pos_change:
                change_cnt=0
                for change_frame in range(i+1,i+int(min_change_frames_for_hit*1.2)+1):
                    neg_pos_change_fol_frame=df_ball_pos['delta_y'].iloc[i]>0 and df_ball_pos['delta_y'].iloc[change_frame]<0
                    pos_pos_change_fol_frame=df_ball_pos['delta_y'].iloc[i]<0 and df_ball_pos['delta_y'].iloc[change_frame]>0
                
                    if neg_pos_change and neg_pos_change_fol_frame:
                        change_cnt+=1
                    elif pos_pos_change and pos_pos_change_fol_frame:
                        change_cnt+=1
                if change_cnt>min_change_frames_for_hit-1:
                    df_ball_pos['ball_hit'].iloc[i]=1

        ball_hit_frames=df_ball_pos[df_ball_pos['ball_hit']==1].index.to_list()
        return ball_hit_frames

    def detect_frames(self,frames,read_from_stub=False,stub_path=None):
        ball_detections=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb') as f:
                ball_detections=pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict=self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(ball_detections,f)
        return ball_detections

    def detect_frame(self,frame):
        results=self.model.predict(frame,conf=0.15)[0]

        ball_dict={}
        for box in results.boxes:
            result=box.xyxy.tolist()[0] 
            ball_dict[1]=result
        
        return ball_dict

    def draw_bboxes(self,video_frames,ball_detections):
        output_video_frames=[]
        for frame, ball_dict in zip(video_frames,ball_detections):
            #DISPLAY BOUNDING BOXES IN VIDEO FRAMES
            for track_id,bbox in ball_dict.items():
                x1,y1,x2,y2=bbox
                cv2.putText(frame,f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1]  -10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,0),2) #x,y coord and rgb color and 2 for solid border
            output_video_frames.append(frame)
        return output_video_frames