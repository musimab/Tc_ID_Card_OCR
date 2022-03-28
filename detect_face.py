import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt


class FindFaceID:
    """
        It takes the image and sends it to the face detection model 
        by rotating it at 15 degree intervals and returning the original image 
        according to that angle which has the highest probability of faces in the image.
    """
    def __init__(self, detection_method = "ssd", rot_interval = 30) -> None:
        self.method  = detection_method
        self.rot_interval = rot_interval
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        self.modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "model/deploy.prototxt.txt"
        self.FaceNet = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
    
    def changeOrientationUntilFaceFound(self,image):


        if(self.method == "dlib"):
            rotated_image = self.__searchFaceDlib(image)
            return rotated_image
        
        elif(self.method == "ssd"):
            rotated_image = self.__searchFaceSsd(image)
            return rotated_image
            
        else:
            print("Select dlib or ssd")
            return None

    def __findFaceDlib(self, image):
        
        faces = self.detector(image)
        num_of_faces = len(faces)
        print("Number of Faces:", num_of_faces )
        if(num_of_faces):
            return True
        return False
    
    def __searchFaceDlib(self, image):
        
        img = image.copy()
        angle_max = 0

        for angle in range(0,360, self.rot_interval):
            img_rotated = self.__rotate_bound(img, angle)
            is_face_available = self.__findFaceDlib(img_rotated)
            if(is_face_available):
                return img_rotated
        

        return None
            

    def __searchFaceSsd(self, image):
        """
        It takes the image and sends it to the face detection model 
        by rotating it at 15 degree intervals and returning the original image 
        according to that angle which has the highest probability of faces in the image.
        """
        img = image.copy()
        face_conf = []
        
        for angle in range(0,360, self.rot_interval):
            img_rotated = self.__rotate_bound(img, angle)
            face_conf.append((self.__detectFace(img_rotated), angle))

        face_confidence = np.array(face_conf)
        face_arg_max = np.argmax(face_confidence, axis=0)
        angle_max = face_confidence[face_arg_max[0]][1]
        #print("Maximum face confidence score at angle: ", angle_max)
        rotated_img = self.__rotate_bound(image, angle_max)
        
        return rotated_img

    @staticmethod
    def __rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
    
    def __detectFace(self,img):

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        self.FaceNet.setInput(blob)
        faces = self.FaceNet.forward()
        
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.6:
                #print("Confidence:", confidence)
                return confidence
            return 0