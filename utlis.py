import cv2
from matplotlib import pyplot as plt
import pytesseract
import numpy as np
from craft_text_detector import Craft
from math import atan2, cos, sin, sqrt, pi
import os

def displayMachedBoxes(img, new_bboxes):
    
    for box in new_bboxes:
        x1, w, y1, h = box
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,0,255), 2)
        cX = round(int(x1) + w/2.0)
        cY = round(int(y1) + h/2.0)
        cv2.circle(img, (cX, cY), 7, (0, 255, 255), -1)
        
    return img

def createHeatMapAndBoxCoordinates(image):
    """
    takes the ID image and sends it to the craft model. 
    Craft returns the character density map and
    the box coordinates of the characters in the image.
    """
    input_image = image.copy()
    craft = Craft(output_dir='outputs', crop_type="poly", cuda=True)
    prediction_result = craft.detect_text(input_image)
    heatmaps = prediction_result["heatmaps"]
   
    return heatmaps["text_score_heatmap"], prediction_result["boxes"]


def readBBoxCordinatesAndCenters(coordinates_txt):
    boxes = []
    centers = []
    with open(coordinates_txt,"r+") as file:
        for line in file:
            x1,y1, x2, y2, x3, y3, x4, y4 = np.int0(line.split(','))

            x = min(x1, x3)
            y = min(y1, y2)
            w = abs(min(x1,x3) - max(x2, x4))
            h = abs(min(y1,y2) - max(y3, y4))

            cX = round(int(x) + w/2.0)
            cY = round(int(y) + h/2.0)
            centers.append((cX, cY))
            bbox = (int(x), w, int(y), h)
            boxes.append(bbox)
    print("number of boxes", len(boxes))
    return np.array(boxes), np.array(centers)

def findOrientationofLines(mask):

    """
    The masks of 4 regions were found at the Unet output, 
    the orientation angles of the largest region within 
    these lines are determined and the orientation angle in degrees is returned.

    """
    cntrs ,hiarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(len(cntrs) == 0):
        return None
    
    areas = [cv2.contourArea(c) for c in cntrs]
    max_index = np.argmax(areas)
    cnt = cntrs[max_index]
   
    angle_pca = getOrientation(cnt,mask)

    return angle_pca

def rotateImage(orientation_angle, final_img):
    """
    The image is rotated according to the given angle
    """
    (h, w) = final_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), orientation_angle, 1.0)
    
    return cv2.warpAffine(final_img, M, (w, h))


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    #drawAxis(img, cntr, p1, (0, 255, 0), 1)
    #drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return np.rad2deg(angle)

def correctPerspective(img):
    """
    it takes the original image and crop 
    the id card with perspective transform
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur,80,80)
    ret, thresh  = cv2.threshold(imgBlur , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate( thresh, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation , kernel, iterations=1)

    cntrs ,hiarchy = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #areas = [cv2.contourArea(c) for c in cntrs]
    #max_index = np.argmax(areas)
    #cnt = cntrs[max_index]
    cnt_max = max(cntrs, key = cv2.contourArea)
 
    approx = cv2.approxPolyDP(cnt_max, 0.02 * cv2.arcLength(cnt_max, True), True)
  
    (heigth_q, width_q) = img.shape[:2]
    
    warped_img = warpImg(img, approx ,  width_q, heigth_q)
    
    #plt.title("original image")
    #plt.imshow(img)
    #plt.show()

    
    #plt.title("processed image")
    #plt.imshow(img_erosion)
    #plt.show()

    #plt.title("warped image")
    #plt.imshow(warped_img)
    #plt.show()
    #cv2.imwrite("warped_img.jpg", warped_img)


    return warped_img

def reorder(myPoints):

    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis = 1)
    
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def warpImg(img, points, w, h):

    points = reorder(points)
    #print("point after reorder:", points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix =  cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))

    return imgWarp

def rotate_bound(image, angle):
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

def rotate_bbox(bb, cx, cy, h, w, theta):
    
    new_bb = np.zeros_like(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    
    return new_bb

def displayAllBoxes(img, rect):
    
    for rct in rect:
        x1, w, y1, h = rct
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,255,0), 2)
        cX = round(int(x1) + w/2.0)
        cY = round(int(y1) + h/2.0)
        cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)
    
    return img

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        print("filename:", filename)
        if img is not None:
            images.append(img)
    
    return images
