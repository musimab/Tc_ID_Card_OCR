import cv2
import numpy as np

class NearestBox:
    
    def __init__(self, distance_thresh, draw_line = False) -> None:
        self.draw_line = draw_line
        self.DISTANCE_THRESH = distance_thresh
    
    def getRightAndLeftBoxCenters(self, box_coordinates, box_indexes):
    
        right_centers = np.zeros((4,2), dtype=np.int32)
        left_centers = np.zeros((4,2), dtype=np.int32)
        
        right_centers_box_full = np.zeros((len(box_coordinates),2))
        left_centers_box_full  = np.zeros((len(box_coordinates),2))

        box1 = box_coordinates[box_indexes[0]]
        box2 = box_coordinates[box_indexes[1]]
        box3 = box_coordinates[box_indexes[2]]
        box4 = box_coordinates[box_indexes[3]]
    
        right_centers[0] = (box1[0]+ box1[1], round(box1[2]+box1[3]/2))
        right_centers[1] = (box2[0]+ box2[1], round(box2[2]+box2[3]/2))
        right_centers[2] = (box3[0]+ box3[1], round(box3[2]+box3[3]/2))
        right_centers[3] = (box4[0]+ box4[1], round(box4[2]+box4[3]/2))

        left_centers[0] =  (box1[0], round(box1[2]+box1[3]/2))
        left_centers[1] =  (box2[0], round(box2[2]+box2[3]/2))
        left_centers[2] =  (box3[0], round(box3[2]+box3[3]/2))
        left_centers[3] =  (box4[0], round(box4[2]+box4[3]/2))

        for i, box in enumerate(box_coordinates):
            right_centers_box_full[i] = (box[0]+ box[1], round(box[2]+box[3]/2))
            left_centers_box_full[i]  = (box[0], round(box[2]+ box[3]/2))
        
        return right_centers, left_centers, right_centers_box_full, left_centers_box_full

    def searchNearestBoundingBoxes(self, box_coordinates, box_indexes, img):
        """
        Retrieves the coordinates of the boxes in the ID card image 
        and the indices of the corresponding regions matched with the mask image. 
        If there are any boxes along a certain Euclidian distance to 
        the right or left of the target boxes, it detects them and updates and 
        returns the coordinates of these boxes.
        """
        right_centers, left_centers, right_centers_box_full, left_centers_box_full = self.getRightAndLeftBoxCenters(box_coordinates, box_indexes)
        
        box1 = box_coordinates[box_indexes[0]]
        box2 = box_coordinates[box_indexes[1]]
        box3 = box_coordinates[box_indexes[2]]
        box4 = box_coordinates[box_indexes[3]]

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

        if(box1_r_neighbours[0].size):
            box_index = 0
            box1_r_indexes = np.squeeze(box1_r_neighbours)
            box1_r = box_coordinates[box1_r_indexes]
            new_box1 = self.getExtendedBoxCoordinates(box1, box1_r)

            print("box1:", box1)
            print("right box1:", box1_r)
            print("new box1:", new_box1)
            
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box1_r_neighbours, img)
            box1 = new_box1
        
        if(box2_r_neighbours[0].size):
            box_index = 1
            box2_r_indexes = np.squeeze(box2_r_neighbours)
            box2_r = box_coordinates[box2_r_indexes]
            new_box2 = self.getExtendedBoxCoordinates(box2, box2_r)
        
            print("box2:", box2)
            print("right box2:",box2_r)
            print("new box2:", new_box2)
            
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box2_r_neighbours, img)
            box2 = new_box2
        
        if(box3_r_neighbours[0].size):
            box_index = 2
            box3_r_indexes = np.squeeze(box3_r_neighbours)
            box3_r = box_coordinates[box3_r_indexes]
            new_box3 = self.getExtendedBoxCoordinates(box3, box3_r)

            print("box3:", box3)
            print("right box3:",box3_r)
            print("new box3:", new_box3)
            
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box3_r_neighbours, img)
            box3 = new_box3

        if(box4_r_neighbours[0].size):
            box_index = 3
            box4_r_indexes = np.squeeze(box4_r_neighbours)
            box4_r = box_coordinates[box4_r_indexes]
            new_box4 = self.getExtendedBoxCoordinates(box4, box4_r)

            print("box4:", box4)
            print("right box4:",box4_r)
            print("new box4:", new_box4)
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box4_r_neighbours, img)
            box4 = new_box4

        box1_l_neighbours = np.where(np.all(left_centers_distance1>0, axis=1 ) & np.all(left_centers_distance1 < [self.DISTANCE_THRESH], axis=1))
        box2_l_neighbours = np.where(np.all(left_centers_distance2>0, axis=1 ) & np.all(left_centers_distance2 < [self.DISTANCE_THRESH], axis=1))
        box3_l_neighbours = np.where(np.all(left_centers_distance3>0, axis=1 ) & np.all(left_centers_distance3 < [self.DISTANCE_THRESH], axis=1))
        box4_l_neighbours = np.where(np.all(left_centers_distance4>0, axis=1 ) & np.all(left_centers_distance4 < [self.DISTANCE_THRESH], axis=1))

        if(box1_l_neighbours[0].size):
            
            box_index = 0
            box1_l_indexes = np.squeeze(box1_l_neighbours)
            box1_l = box_coordinates[box1_l_indexes]
            new_box1 = self.getExtendedBoxCoordinates(box1, box1_l)
            
            print("box1:", box1)
            print("left box1:", box1_l)
            print("new box1:", new_box1)
            
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box1_l_neighbours, img)
            box1 = new_box1

        if(box2_l_neighbours[0].size):
            
            box_index = 1
            box2_l_indexes = np.squeeze(box2_l_neighbours)
            box2_l = box_coordinates[box2_l_indexes]
            new_box2 = self.getExtendedBoxCoordinates(box2, box2_l)
            print("box2:", box2)
            print("left box2:", box1_l)
            print("new box2:", new_box2)
            
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box2_l_neighbours, img)
            box2 = new_box2

        if(box3_l_neighbours[0].size):
            
            box_index = 2
            box3_l_indexes = np.squeeze(box3_l_neighbours)
            box3_l = box_coordinates[box3_l_indexes]
            new_box3 = self.getExtendedBoxCoordinates(box3, box3_l)
            print("box3:", box3)
            print("left box3:", box3_l)
            print("new box3:", new_box3)
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box3_l_neighbours, img)
            box3 = new_box3

        if(box4_l_neighbours[0].size):
            
            box_index = 3
            box4_l_indexes = np.squeeze(box4_l_neighbours)
            box4_l = box_coordinates[box4_l_indexes]
            new_box1 = self.getExtendedBoxCoordinates(box4, box4_l)
            print("box4:", box4)
            print("left box4:", box4_l)
            print("new box4:", new_box4)
            if(self.draw_line):
                img = self.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box4_l_neighbours, img)
            box4 = new_box4

        return box1, box2, box3, box4


    def getExtendedBoxCoordinates(self, box1, box1_r):
        
        new_box = np.zeros_like(box1)
        new_box[0] = box1[0] if(box1[0] < box1_r[0]) else box1_r[0]
        new_box[1] = box1_r[1] + box1[1]  + self.DISTANCE_THRESH
        new_box[2] = box1[2]
        new_box[3] = box1[3] if(box1[3] > box1_r[3]) else box1_r[3]
        
        return new_box
    
    def drawlineBetweenBox(self,BoxNum, right_centers, left_centers_box_full, box2_r_neighbours, img):
    
        start_point = int(right_centers[BoxNum][0]),int(right_centers[BoxNum][1])
        box2_neighbour_indexes = np.squeeze(box2_r_neighbours)
        end_point = int(left_centers_box_full[box2_neighbour_indexes][0]), int(left_centers_box_full[box2_neighbour_indexes][1])

        return cv2.line(img, end_point,start_point,  (0,255,0), 3)

