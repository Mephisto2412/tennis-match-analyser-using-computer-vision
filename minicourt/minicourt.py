import cv2
import sys
sys.path.append('../')
import constants
import numpy as np
from utils import(
    convert_m_to_px,
    convert_px_to_m,
    get_foot_pos,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_dist,
    get_centre_of_bbox,
    measure_dist
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width=250
        self.drawing_rectangle_height=500
        self.buffer=50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()
    
    def set_canvas_background_box_position(self,frame):
        frame=frame.copy()
        
        self.end_x=frame.shape[1]-self.buffer
        self.end_y=self.buffer+self.drawing_rectangle_height
        self.start_x=self.end_x-self.drawing_rectangle_width
        self.start_y=self.end_y-self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x=self.start_x+self.padding_court
        self.court_start_y=self.start_y+self.padding_court
        self.court_end_x=self.end_x-self.padding_court
        self.court_end_y=self.end_y-self.padding_court
        self.court_drawing_width=self.court_end_x-self.court_start_x

    def convert_m_px(self,meters):
        return convert_m_to_px(meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_key_points=[0]*28
        drawing_key_points[0],drawing_key_points[1]=int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2],drawing_key_points[3]=int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4]=int(self.court_start_x)
        drawing_key_points[5]=self.court_start_y+ self.convert_m_px(constants.HALF_COURT_LINE_HEIGHT*2)
        drawing_key_points[6]=drawing_key_points[0]+self.court_drawing_width
        drawing_key_points[7]=drawing_key_points[5]
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_m_px(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_m_px(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_m_px(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_m_px(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_m_px(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_m_px(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_m_px(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_m_px(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    
    def draw_background_rectangle(self,frame):
        shapes=np.zeros_like(frame,np.uint8) #dummy image with all zeroes
        cv2.rectangle(shapes,(self.start_x,self.start_y),(self.end_x,self.end_y),(255,255,255),cv2.FILLED)
        out=frame.copy()
        alpha=0.5
        mask=shapes.astype(bool)
        out[mask]=cv2.addWeighted(frame,alpha,shapes,1-alpha,0)[mask]
        return out

    def draw_mini_court(self,frames):
        out_frames=[]
        for frame in frames:
            frame=self.draw_background_rectangle(frame)
            frame=self.draw_court(frame)
            out_frames.append(frame)
        return out_frames

    def draw_court(self,frame):
        for i in range (0,len(self.drawing_key_points),2):
            x=int(self.drawing_key_points[i])
            y=int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (255,0,0), -1)


        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        
        return frame
    
    def get_mini_court_coord(self,object_pos,closest_keypoint, closest_keypoint_index,player_height_in_pixels,player_height_in_meters):
        dist_from_keypoint_x_pixels, dist_from_keypoint_y_pixels= measure_xy_dist(object_pos,closest_keypoint)

        dist_from_keypoint_x_meters=convert_px_to_m(dist_from_keypoint_x_pixels,player_height_in_meters,player_height_in_pixels)
        dist_from_keypoint_y_meters= convert_px_to_m(dist_from_keypoint_y_pixels,player_height_in_meters,player_height_in_pixels)

        #CONVERT METERS DISTANCE TO MINICOURT COORDS:
        mini_court_x_dist_pixels=self.convert_m_px(dist_from_keypoint_x_meters)
        mini_court_y_dist_pixels=self.convert_m_px(dist_from_keypoint_y_meters)

        closest_minicourt_keypoint=(
            self.drawing_key_points[closest_keypoint_index*2],
            self.drawing_key_points[closest_keypoint_index*2+1]
        )

        minicourt_player_pos=(closest_minicourt_keypoint[0]+mini_court_x_dist_pixels,
                              closest_minicourt_keypoint[1]+mini_court_y_dist_pixels
                              )
        return minicourt_player_pos


    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def convert_bbox_to_minicourt_coords(self,player_boxes,ball_boxes,og_court_keypoints):
        player_heights={
            1:constants.PLAYER_1_HEIGHT,
            2:constants.PLAYER_2_HEIGHT
        }

        output_player_boxes=[]
        output_ball_boxes=[]
        for frame_num,player_box in enumerate(player_boxes):
            ball_box=ball_boxes[frame_num][1]
            ball_pos=get_centre_of_bbox(ball_box)
            closest_player_id_to_ball=min(player_box.keys(), key=lambda x: measure_dist(ball_pos,get_centre_of_bbox(player_box[x])))

            output_player_bbox_dict={}
            for player_id,bbox in player_box.items():
                foot_pos=get_foot_pos(bbox)

                closest_keypt_index=get_closest_keypoint_index(foot_pos,og_court_keypoints,[0,2,12,13])
                closest_keypt=(og_court_keypoints[closest_keypt_index*2],og_court_keypoints[closest_keypt_index*2+1])

                frame_min_index=max(0,frame_num-20)
                frame_max_index=min(len(player_boxes),frame_num+20)

                bbox_height_in_pixels= [get_height_of_bbox(player_boxes[i][player_id])for i in range(frame_min_index,frame_max_index)]

                max_player_height_in_pixels=max(bbox_height_in_pixels)

                mini_court_player_pos=self.get_mini_court_coord(foot_pos,closest_keypt,closest_keypt_index,max_player_height_in_pixels,player_heights[player_id])

                output_player_bbox_dict[player_id]= mini_court_player_pos

                if closest_player_id_to_ball==player_id:
                    closest_keypt_index=get_closest_keypoint_index(ball_pos,og_court_keypoints,[0,2,12,13])
                    closest_keypt=(og_court_keypoints[closest_keypt_index*2],og_court_keypoints[closest_keypt_index*2+1])
                    
                    mini_court_player_pos=self.get_mini_court_coord(ball_pos,closest_keypt,closest_keypt_index,max_player_height_in_pixels,player_heights[player_id])

                    output_ball_boxes.append({1: mini_court_player_pos})
            output_player_boxes.append(output_player_bbox_dict)
        return output_player_boxes,output_ball_boxes

    def draw_points_on_minicourt(self,frames,pos,color=(0,255,0)):
        for frame_num,frame in enumerate(frames):
            for position in pos[frame_num].items():
                x=position[1][0]
                y=position[1][1]
                x=int(x)
                y=int(y)
                # print(x,y)
                cv2.circle(frame,(x,y),5,color,-1)
        return frames