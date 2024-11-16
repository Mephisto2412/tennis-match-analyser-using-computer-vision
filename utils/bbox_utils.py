def get_centre_of_bbox(bbox):
    x1,y1,x2,y2=bbox
    x_centre=int((x1+x2)/2)
    y_centre=int((y1+y2)/2)
    return (x_centre,y_centre)

def measure_dist(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_pos(bbox):
    x1,y1,x2,y2=bbox
    x_centre=int((x1+x2)/2)
    return (x_centre,y2)

def get_closest_keypoint_index(point,keypoints,keypoint_indexes):
    closest_dist=float('inf')
    key_point_index=keypoint_indexes[0]
    for keypoint_index in keypoint_indexes:
        keypoint=keypoints[keypoint_index*2], keypoints[keypoint_index*2+1],
        dist=abs(point[1]-keypoint[1])

        if dist<closest_dist:
            closest_dist=dist
            key_point_index=keypoint_index
    return key_point_index

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_dist(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

# def get_centre_of_bbox(bbox):
#     x1,y1,x2,y2=bbox
#     x_centre=int((x1+x2)/2)
#     y_centre=int((y1+y2)/2)
#     return (x_centre,y_centre)