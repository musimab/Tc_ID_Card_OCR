from copy import deepcopy
import cv2
import numpy as np

class NearestBox:
    
    def __init__(self, distance_thresh, draw_line = False) -> None:
        self.draw_line = draw_line
        self.DISTANCE_THRESH = distance_thresh
        self.target_box_cordinates = None

    def getRightAndLeftCentersforAllBoxes(self, box_coordinates):
        """
        it takes box coordinates that are generated from craft and matched box indexes
        box_coordinates : character regions that consist of (20,4) numpy arrays
        return param: right and left vertice centers of all box coordinates
        """

        right_centers_box_full = np.zeros((len(box_coordinates),2))
        left_centers_box_full  = np.zeros((len(box_coordinates),2))

        # left vertice centers for all boxes (cx, cy )          = ( x , y + h/2 ) 
        # right vertice centers for boxes (cx, cy )          = ( x + w , y + h/2 ) 
        for i, box in enumerate(box_coordinates):
            right_centers_box_full[i] = (box[0]+ box[1], round(box[2]+box[3]/2))
            left_centers_box_full[i]  = (box[0], round(box[2]+ box[3]/2))
        
        return  right_centers_box_full, left_centers_box_full


    def setTargetBoxCordinates(self, target_box_cord):
        self.target_box_cordinates = target_box_cord

    def getTargetBoxCordinates(self):
        return self.target_box_cordinates


    def getRightAndLeftCentersforTargetBoxes(self, box_coordinates, box_indexes):
        """
        it takes box coordinates that are generated from craft and matched box indexes
        box_coordinates : character regions that consist of (20,4) numpy arrays
        box indexes : target 4 box indexes in tuple type
        return param: right and left centers of target and all box coordinates
        """
        # 4 target regions both right and left vertice centers
        right_centers = np.zeros((4,2), dtype=np.int32)
        left_centers = np.zeros((4,2), dtype=np.int32)
        
        box1 = box_coordinates[box_indexes[0]]
        box2 = box_coordinates[box_indexes[1]]
        box3 = box_coordinates[box_indexes[2]]
        box4 = box_coordinates[box_indexes[3]]
        # right vertice center for box (cx, cy )          = ( x + w , y + h/2 ) 
        right_centers[0] = (box1[0]+ box1[1], round(box1[2]+box1[3]/2))
        right_centers[1] = (box2[0]+ box2[1], round(box2[2]+box2[3]/2))
        right_centers[2] = (box3[0]+ box3[1], round(box3[2]+box3[3]/2))
        right_centers[3] = (box4[0]+ box4[1], round(box4[2]+box4[3]/2))
        # left vertice center for box (cx, cy )          = ( x , y + h/2 ) 
        left_centers[0] =  (box1[0], round(box1[2]+box1[3]/2))
        left_centers[1] =  (box2[0], round(box2[2]+box2[3]/2))
        left_centers[2] =  (box3[0], round(box3[2]+box3[3]/2))
        left_centers[3] =  (box4[0], round(box4[2]+box4[3]/2))

        return right_centers, left_centers

    def searchNearestBoundingBoxes(self, box_coordinates, box_indexes, img):
        """
        Retrieves the coordinates of the boxes in the ID card image 
        and the indices of the corresponding regions matched with the mask image. 
        If there are any boxes along a certain Euclidian distance to 
        the right or left of the target boxes, it detects them and updates and 
        returns the coordinates of these boxes.
        """
        right_centers_box_full, left_centers_box_full = self.getRightAndLeftCentersforAllBoxes(box_coordinates)
        right_centers, left_centers = self.getRightAndLeftCentersforTargetBoxes(box_coordinates, box_indexes)
        
        box1 = box_coordinates[box_indexes[0]]
        box2 = box_coordinates[box_indexes[1]]
        box3 = box_coordinates[box_indexes[2]]
        box4 = box_coordinates[box_indexes[3]]

        target_box_cordinates = {}
        target_box_cordinates["box1"] = box1
        target_box_cordinates["box2"] = box2
        target_box_cordinates["box3"] = box3
        target_box_cordinates["box4"] = box4

        self.setTargetBoxCordinates(target_box_cordinates)

        right_centers_distance1 = np.zeros((len(right_centers_box_full), 1))
        right_centers_distance2 = np.zeros((len(right_centers_box_full), 1))
        right_centers_distance3 = np.zeros((len(right_centers_box_full), 1))
        right_centers_distance4 = np.zeros((len(right_centers_box_full), 1))

        left_centers_distance1 = np.zeros((len( left_centers_box_full), 1) )
        left_centers_distance2 = np.zeros((len( left_centers_box_full), 1) )
        left_centers_distance3 = np.zeros((len( left_centers_box_full), 1) )
        left_centers_distance4 = np.zeros((len( left_centers_box_full), 1) )


        for i , left_box_centers in enumerate(left_centers_box_full):
            
            right_centers_distance1[i] = np.linalg.norm(right_centers[0] - left_box_centers) #box1 right center - other boxes left center
            right_centers_distance2[i] = np.linalg.norm(right_centers[1] - left_box_centers)
            right_centers_distance3[i] = np.linalg.norm(right_centers[2] - left_box_centers)
            right_centers_distance4[i] = np.linalg.norm(right_centers[3] - left_box_centers)
        
        for i , right_box_centers in enumerate(right_centers_box_full):
            
            left_centers_distance1[i] = np.linalg.norm(left_centers[0] - right_box_centers) # box1 left center - other boexes right center
            left_centers_distance2[i] = np.linalg.norm(left_centers[1] - right_box_centers)
            left_centers_distance3[i] = np.linalg.norm(left_centers[2] - right_box_centers)
            left_centers_distance4[i] = np.linalg.norm(left_centers[3] - right_box_centers)

        box1_r_neighbours = np.where(np.all(right_centers_distance1>0, axis=1 ) & np.all(right_centers_distance1 < [self.DISTANCE_THRESH], axis=1))
        box2_r_neighbours = np.where(np.all(right_centers_distance2>0, axis=1 ) & np.all(right_centers_distance2 < [self.DISTANCE_THRESH], axis=1))
        box3_r_neighbours = np.where(np.all(right_centers_distance3>0, axis=1 ) & np.all(right_centers_distance3 < [self.DISTANCE_THRESH], axis=1))
        box4_r_neighbours = np.where(np.all(right_centers_distance4>0, axis=1 ) & np.all(right_centers_distance4 < [self.DISTANCE_THRESH], axis=1))

        box1_l_neighbours = np.where(np.all(left_centers_distance1>0, axis=1 ) & np.all(left_centers_distance1 < [self.DISTANCE_THRESH], axis=1))
        box2_l_neighbours = np.where(np.all(left_centers_distance2>0, axis=1 ) & np.all(left_centers_distance2 < [self.DISTANCE_THRESH], axis=1))
        box3_l_neighbours = np.where(np.all(left_centers_distance3>0, axis=1 ) & np.all(left_centers_distance3 < [self.DISTANCE_THRESH], axis=1))
        box4_l_neighbours = np.where(np.all(left_centers_distance4>0, axis=1 ) & np.all(left_centers_distance4 < [self.DISTANCE_THRESH], axis=1))


        box_right_neighbours = {}
        box_left_neighbours = {}

        if(box1_r_neighbours[0].size):
            box_right_neighbours["box1"] = (box1, box_coordinates[np.squeeze(box1_r_neighbours)])
        if(box2_r_neighbours[0].size):
            box_right_neighbours["box2"] = (box2, box_coordinates[np.squeeze(box2_r_neighbours)])
        if(box3_r_neighbours[0].size):
            box_right_neighbours["box3"] = (box3, box_coordinates[np.squeeze(box3_r_neighbours)])
        if(box4_r_neighbours[0].size):
            box_right_neighbours["box4"] = (box4, box_coordinates[np.squeeze(box4_r_neighbours)])

        if(box1_l_neighbours[0].size):
            box_left_neighbours["box1"] = (box1, box_coordinates[np.squeeze(box1_l_neighbours)])
        if(box2_l_neighbours[0].size):
            box_left_neighbours["box2"] = (box2, box_coordinates[np.squeeze(box2_l_neighbours)])
        if(box3_l_neighbours[0].size):
            box_left_neighbours["box3"] = (box3, box_coordinates[np.squeeze(box3_l_neighbours)])
        if(box4_l_neighbours[0].size):
            box_left_neighbours["box4"] = (box4, box_coordinates[np.squeeze(box4_l_neighbours)])
                       
        self.updateBoxCoordinates( box_right_neighbours, box_left_neighbours)

        return np.asarray([target_box_cordinates["box1"], target_box_cordinates["box2"], target_box_cordinates["box3"], target_box_cordinates["box4"]])

    def updateBoxCoordinates(self,  box_right_neighbours, box_left_neighbours):
        
        new_box = self.getTargetBoxCordinates()
        
        #print("old box cordinates:", new_box)
        
        for box_name, box_indx in box_right_neighbours.items():

            new_box[box_name] = self.getExtendedBoxCoordinates(box_indx[0], box_indx[1])
        
        for box_name, box_indx in box_left_neighbours.items():

            new_box[box_name] = self.getExtendedBoxCoordinates(box_indx[0], box_indx[1])
        
        #print("new box cordinates:", new_box)
    
    def getExtendedBoxCoordinates(self, box, box_r_or_l):
        """
        box : original box cordinate (x, w, y, h)
        box_r_or_l : nearest right or left box cordinate (xn, wn, yn, hn)

        return type extended box cordinate
        """
        print("box_r_or_l.size:", box_r_or_l.size)
        new_box = np.zeros_like(box)

        if(box_r_or_l.size <= 4):
            
            new_box[0] = box[0] if(box[0] < box_r_or_l[0]) else box_r_or_l[0]
            new_box[1] = box_r_or_l[1] + box[1]  + self.DISTANCE_THRESH
            new_box[2] = box[2]
            new_box[3] = box[3] if(box[3] > box_r_or_l[3]) else box_r_or_l[3]
        
        else:
            for bx_rl in box_r_or_l:
                new_box[0] = box[0] if(box[0] < bx_rl[0]) else bx_rl[0]
                new_box[1] = bx_rl[1] + box[1]  + self.DISTANCE_THRESH
                new_box[2] = box[2]
                new_box[3] = box[3] if(box[3] > bx_rl[3]) else bx_rl[3]
        
        return new_box
    
    def drawlineBetweenBox(self,BoxNum, right_centers, left_centers_box_full, box2_r_neighbours, img):
    
        start_point = int(right_centers[BoxNum][0]),int(right_centers[BoxNum][1])
        box2_neighbour_indexes = np.squeeze(box2_r_neighbours)
        end_point = int(left_centers_box_full[box2_neighbour_indexes][0]), int(left_centers_box_full[box2_neighbour_indexes][1])

        return cv2.line(img, end_point, start_point,  (0,255,0), 3)

