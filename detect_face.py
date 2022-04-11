from abc import abstractmethod
import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
from abc import ABC

class FaceDetector(ABC):

    @abstractmethod
    def changeOrientationUntilFaceFound(self, image, rot_interval):
        pass

    @abstractmethod
    def findFace(self,img):
        pass

    @abstractmethod
    def rotate_bound(self,image, angle):
        pass


class DlibFaceDetector(FaceDetector):        
    
    def changeOrientationUntilFaceFound(self, image, rot_interval):
        
        img = image.copy()
        angle_max = 0

        for angle in range(0,360, rot_interval):
            
            img_rotated = self.rotate_bound(img, angle)
            is_face_available = self.findFace(img_rotated)
            
            if(is_face_available):
                return img_rotated
        

        return None

    def findFace(self, image):
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        faces = detector(image)
        num_of_faces = len(faces)
        print("Dlib Number of Faces:", num_of_faces )
        if(num_of_faces):
            return True
        return False
    
    
    def rotate_bound(self, image, angle):
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


class SsdFaceDetector(FaceDetector):
    
    
    def changeOrientationUntilFaceFound(self,image, rot_interval):
        """
        It takes the image and sends it to the face detection model 
        by rotating it at 15 degree intervals and returning the original image 
        according to that angle which has the highest probability of faces in the image.
        """
        img = image.copy()
        face_conf = []
        
        for angle in range(0, 360, rot_interval):
            img_rotated = self.rotate_bound(img, angle)
            face_conf.append((self.findFace(img_rotated), angle))

        face_confidence = np.array(face_conf)
        face_arg_max = np.argmax(face_confidence, axis=0)
        angle_max = face_confidence[face_arg_max[0]][1]

        rotated_img = self.rotate_bound(image, angle_max)
        
        return rotated_img
    
    def findFace(self,img):
        
        modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "model/deploy.prototxt.txt"
        FaceNet = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        
        FaceNet.setInput(blob)
        faces = FaceNet.forward()
        
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.6:
                #print("Confidence:", confidence)
                return confidence
            return 0
    
    
    def rotate_bound(self,image, angle):
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

class HaarFaceDetector(FaceDetector):
    
    def changeOrientationUntilFaceFound(self,image, rot_interval):
        img = image.copy()

        for angle in range(0,360, rot_interval):
            
            img_rotated = self.rotate_bound(img, angle)
            is_face_available = self.findFace(img_rotated)
            
            if(is_face_available):
                return img_rotated
        

        return None

    def findFace(self,img):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_of_faces = len(faces)
            
        if(num_of_faces ):
            print("Haar Number of Faces:", num_of_faces)
            return True
        
        return False
            
        
    def rotate_bound(self,image, angle):
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

class FaceFactory(ABC):
    
    @abstractmethod
    def get_face_detector(self) -> FaceDetector:
        """ Returns new face detector """

class DlibModel(FaceFactory):
    
    def get_face_detector(self) -> FaceDetector:
        return DlibFaceDetector()

class SsdModel(FaceFactory):
    
    def get_face_detector(self) -> FaceDetector:
        return SsdFaceDetector()

class HaarModel(FaceFactory):
    
    def get_face_detector(self) -> FaceDetector:
        return HaarFaceDetector()


def face_factory(face_model = "ssd")->FaceFactory:
    """Constructs an face detector factory based on the user's preference."""
    
    factories = {
        "dlib": DlibModel(),
        "ssd" : SsdModel(),
        "haar": HaarModel()
    }
    return factories[face_model]

