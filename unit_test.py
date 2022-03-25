
import unittest
import numpy as np
from find_nearest_box import NearestBox
import cv2

class Test_Nearest_Box(unittest.TestCase):
    
    def setUp(self):
        self.Nearest_Box = NearestBox(distance_thresh=3,draw_line=False)

        self.box_coordinates2 = np.random.randint(1,100, size=(20,4))
                             
        self.box_coordinates = np.array([[4,10,8,6],
                                        [6,10,12,6],  #box1
                                        [17,5,13,6],  #box1_r_l
                                        [10,12,8,6],
                                        [5,6,10,6],
                                        [8,15,20,6]])
        self.box_indexes = (0,1,3,5)
        self.img = cv2.imread("identityCardRecognition/deneme_testr/id1.jpeg")
    
    
    #def tearDown(self):
    #    print("testler")


    def test_getRightAndLeftCentersforAllBoxes(self):

        right_cent2, left_cent2 = self.Nearest_Box.getRightAndLeftCentersforTargetBoxes(self.box_coordinates, self.box_indexes)
        
        self.assertTupleEqual(tuple(right_cent2[0]), (14,11))
        self.assertTupleEqual(tuple(right_cent2[1]), (16,15))
        self.assertTupleEqual(tuple(right_cent2[2]), (22,11))
        self.assertTupleEqual(tuple(right_cent2[3]), (23,23))

        self.assertTupleEqual(tuple(left_cent2[0]), (4,11))
        self.assertTupleEqual(tuple(left_cent2[1]), (6,15))
        self.assertTupleEqual(tuple(left_cent2[2]), (10,11))
        self.assertTupleEqual(tuple(left_cent2[3]), (8,23))
            

    def test_output_size(self):
        
        right_cent, left_cent = self.Nearest_Box.getRightAndLeftCentersforAllBoxes(self.box_coordinates2)
        
        right_cent2, left_cent2 = self.Nearest_Box.getRightAndLeftCentersforTargetBoxes(self.box_coordinates2, self.box_indexes)
        
        self.assertEqual(right_cent.shape, (20,2) )
        self.assertEqual(left_cent.shape,  (20,2) )

        self.assertEqual(right_cent2.shape, (4,2) )
        self.assertEqual(left_cent2.shape,  (4,2) )

    def test_searchNearestBoundingBoxes(self):
        
        bboxes = self.Nearest_Box.searchNearestBoundingBoxes(self.box_coordinates, self.box_indexes, self.img)
        
        self.assertIn(bboxes[0], self.box_coordinates)
        self.assertIn(bboxes[1], self.box_coordinates)
        self.assertIn(bboxes[2], self.box_coordinates)
        self.assertIn(bboxes[3], self.box_coordinates)
        
       
        
    
        




if __name__ == '__main__':
    unittest.main()
