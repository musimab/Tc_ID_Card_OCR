import cv2
from matplotlib import pyplot as plt
import pytesseract
import numpy as np
from zmq import device
import utlis
from pytorch_unet import unet_predict
import craft
from craft_text_detector import Craft

def ocrOutputs(img, bbox):
    
    LowerThr = 5
    UpperThr = 5
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in bbox :
        x, w, y, h = box
        crop_img = img_rgb[y-LowerThr:y+h+UpperThr, x-LowerThr:x+w+UpperThr]
        #processed_img = utlis.denoiseImage(crop_img)
        
        plt.title("OCR image")
        plt.imshow(crop_img, cmap='gray')
        plt.show()
        print(pytesseract.image_to_string(crop_img))


def getCenterRatios(img, centers):

    if(len(img.shape) == 2):
        img_h, img_w = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios
    else :
        img_h, img_w,_ = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios


def displayMachedBoxes(img, new_bboxes):
    
    for box in new_bboxes:
        x1, w, y1, h = box
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,0,255), 2)
        cX = round(int(x1) + w/2.0)
        cY = round(int(y1) + h/2.0)
        cv2.circle(img, (cX, cY), 7, (0, 255, 255), -1)
        
    return img


def matchCenters(ratios1, ratios2):

    bbb0 = np.zeros_like(ratios2)
    bbb1 = np.zeros_like(ratios2)
    bbb2 = np.zeros_like(ratios2)
    bbb3 = np.zeros_like(ratios2)

    for i , r2 in enumerate(ratios2):
        bbb0[i] = abs(ratios1[0] - r2)
        bbb1[i] = abs(ratios1[1] - r2)
        bbb2[i] = abs(ratios1[2] - r2)
        bbb3[i] = abs(ratios1[3] - r2)

    sum_b0 = np.sum(bbb0, axis = 1)
    sum_b0 = np.reshape(sum_b0, (-1, 1))
    arg_min_b0 = np.argmin(sum_b0, axis=0)

    sum_b1 = np.sum(bbb1, axis = 1)
    sum_b1 = np.reshape(sum_b1, (-1, 1))
    arg_min_b1 = np.argmin(sum_b1, axis=0)

    sum_b2 = np.sum(bbb2, axis = 1)
    sum_b2 = np.reshape(sum_b2, (-1, 1))
    arg_min_b2 = np.argmin(sum_b2, axis=0)

    sum_b3 = np.sum(bbb3, axis = 1)
    sum_b3 = np.reshape(sum_b3, (-1, 1))
    arg_min_b3 = np.argmin(sum_b3, axis=0)

    return np.squeeze(arg_min_b0), np.squeeze(arg_min_b1), np.squeeze(arg_min_b2),np.squeeze(arg_min_b3)         


def getCenterOfMasks(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_centers = []
    img_h, img_w = thresh.shape

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        detected_centers.append((cX, cY))
        cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)
        if w > 10 and h > 10:
            cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2)
    #plt.imshow(thresh, cmap='gray')
    #plt.show()
    return np.array(detected_centers)


def getExtendedBoxCoordinates(box1, box1_r):
    
    DISTANCE_THRES = 10
    new_box = np.zeros_like(box1)
    new_box[0] = box1[0] if(box1[0] < box1_r[0]) else box1_r[0]
    new_box[1] = box1_r[1] + box1[1]  + DISTANCE_THRES 
    new_box[2] = box1[2]
    new_box[3] = box1[3] if(box1[3] > box1_r[3]) else box1_r[3]
    return new_box


def searchNearestBoundingBoxes(box_coordinates, box_indexes, img):
    
    DISTANCE_THRES = 10
    
    right_centers, left_centers, right_centers_box_full, left_centers_box_full = utlis.getRightAndLeftBoxCenters(box_coordinates, box_indexes)
    
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

    box1_r_neighbours = np.where(np.all(right_centers_distance1>0, axis=1 ) & np.all(right_centers_distance1 < [DISTANCE_THRES], axis=1))
    box2_r_neighbours = np.where(np.all(right_centers_distance2>0, axis=1 ) & np.all(right_centers_distance2 < [DISTANCE_THRES], axis=1))
    box3_r_neighbours = np.where(np.all(right_centers_distance3>0, axis=1 ) & np.all(right_centers_distance3 < [DISTANCE_THRES], axis=1))
    box4_r_neighbours = np.where(np.all(right_centers_distance4>0, axis=1 ) & np.all(right_centers_distance4 < [DISTANCE_THRES], axis=1))

    if(box1_r_neighbours[0].size):
        box_index = 0
        box1_r_indexes = np.squeeze(box1_r_neighbours)
        box1_r = box_coordinates[box1_r_indexes]
        new_box1 = getExtendedBoxCoordinates(box1, box1_r)

        print("box1:", box1)
        print("right box1:", box1_r)
        print("new box1:", new_box1)
        img = utlis.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box1_r_neighbours, img)
        box1 = new_box1
    
    if(box2_r_neighbours[0].size):
        box_index = 1
        box2_r_indexes = np.squeeze(box2_r_neighbours)
        box2_r = box_coordinates[box2_r_indexes]
        new_box2 = getExtendedBoxCoordinates(box2, box2_r)
     
        print("box2:", box2)
        print("right box2:",box2_r)
        print("new box2:", new_box2)
        img = utlis.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box2_r_neighbours, img)
        box2 = new_box2
    
    if(box3_r_neighbours[0].size):
        box_index = 2
        box3_r_indexes = np.squeeze(box3_r_neighbours)
        box3_r = box_coordinates[box3_r_indexes]
        new_box3 = getExtendedBoxCoordinates(box3, box3_r)

        print("box3:", box3)
        print("right box3:",box3_r)
        print("new box3:", new_box3)
        img = utlis.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box3_r_neighbours, img)
        box3 = new_box3

    if(box4_r_neighbours[0].size):
        box_index = 3
        box4_r_indexes = np.squeeze(box4_r_neighbours)
        box4_r = box_coordinates[box4_r_indexes]
        new_box4 = getExtendedBoxCoordinates(box4, box4_r)

        print("box4:", box4)
        print("right box4:",box4_r)
        print("new box4:", new_box4)
        img = utlis.drawlineBetweenBox(box_index, right_centers, left_centers_box_full, box4_r_neighbours, img)
        box4 = new_box4

    box1_l_neighbours = np.where(np.all(left_centers_distance1>0, axis=1 ) & np.all(left_centers_distance1 < [DISTANCE_THRES], axis=1))
    box2_l_neighbours = np.where(np.all(left_centers_distance2>0, axis=1 ) & np.all(left_centers_distance2 < [DISTANCE_THRES], axis=1))
    box3_l_neighbours = np.where(np.all(left_centers_distance3>0, axis=1 ) & np.all(left_centers_distance3 < [DISTANCE_THRES], axis=1))
    box4_l_neighbours = np.where(np.all(left_centers_distance4>0, axis=1 ) & np.all(left_centers_distance4 < [DISTANCE_THRES], axis=1))

    if(box1_l_neighbours[0].size):
        
        box_index = 0
        box1_l_indexes = np.squeeze(box1_l_neighbours)
        box1_l = box_coordinates[box1_l_indexes]
        new_box1 = getExtendedBoxCoordinates(box1, box1_l)
        
        print("box1:", box1)
        print("left box1:", box1_l)
        print("new box1:", new_box1)
        img = utlis.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box1_l_neighbours, img)
        box1 = new_box1

    if(box2_l_neighbours[0].size):
        
        box_index = 1
        box2_l_indexes = np.squeeze(box2_l_neighbours)
        box2_l = box_coordinates[box2_l_indexes]
        new_box2 = getExtendedBoxCoordinates(box2, box2_l)
        print("box2:", box2)
        print("left box2:", box1_l)
        print("new box2:", new_box2)
        img = utlis.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box2_l_neighbours, img)
        box2 = new_box2

    if(box3_l_neighbours[0].size):
        
        box_index = 2
        box3_l_indexes = np.squeeze(box3_l_neighbours)
        box3_l = box_coordinates[box3_l_indexes]
        new_box3 = getExtendedBoxCoordinates(box3, box3_l)
        print("box3:", box3)
        print("left box3:", box3_l)
        print("new box3:", new_box3)
        img = utlis.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box3_l_neighbours, img)
        box3 = new_box3

    if(box4_l_neighbours[0].size):
        
        box_index = 3
        box4_l_indexes = np.squeeze(box4_l_neighbours)
        box4_l = box_coordinates[box4_l_indexes]
        new_box1 = getExtendedBoxCoordinates(box4, box4_l)
        print("box4:", box4)
        print("left box4:", box4_l)
        print("new box4:", new_box4)
        img = utlis.drawlineBetweenBox(box_index, left_centers, right_centers_box_full, box4_l_neighbours, img)
        box4 = new_box4

    return box1, box2, box3, box4


def changeOrientationUntilFaceFound(image):
    
    img = image.copy()
    face_conf = []
    
    for angle in range(0,360, 15):
        rotated_img = utlis.rotate_bound(img.copy(), angle)
        face_conf.append((utlis.detectFace(rotated_img), angle))

    face_confidence = np.array(face_conf)
    face_arg_max = np.argmax(face_confidence, axis=0)
    angle_max = face_confidence[face_arg_max[0]][1]
    print("Maximum face confidence score at angle: ", angle_max)
    rotated_img = utlis.rotate_bound(image, angle_max)
    
    return rotated_img

def getBoxRegions(regions):
    boxes = []
    centers = []
    for box_region in regions:

        x1,y1, x2, y2, x3, y3, x4, y4 = np.int0(box_region.reshape(-1))
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

def createHeatMapAndBoxCoordinates(image):
    
    input_image = image.copy()
    craft = Craft(output_dir='outputs', crop_type="poly", cuda=True)
    prediction_result = craft.detect_text(input_image)
    heatmaps = prediction_result["heatmaps"]
   
    return heatmaps["text_score_heatmap"], prediction_result["boxes"]



if '__main__' == __name__:
    
    img1 = cv2.imread("images/ori5.jpg")
    
    plt.title("Original image")
    plt.imshow(img1)
    plt.show()
    
    final_img = changeOrientationUntilFaceFound(img1)
    
    #final_img = utlis.correctPerspective(final_img)
    
    txt_heat_map, regions = createHeatMapAndBoxCoordinates(final_img)

    predicted_mask = unet_predict.unet_segment_model(txt_heat_map)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    plt.show()
    bbox_coordinates , box_centers = getBoxRegions(regions)
   
    mask_centers = getCenterOfMasks(predicted_mask)
   
    # centers ratio for 4 boxes
    centers_ratio_mask = getCenterRatios(predicted_mask, mask_centers) 

    # centers ratio for all boxes
    centers_ratio_all = getCenterRatios(final_img, box_centers) 
  
    matched_box_indexes = matchCenters(centers_ratio_mask , centers_ratio_all)

    new_bboxes = searchNearestBoundingBoxes(bbox_coordinates, matched_box_indexes, final_img.copy() )
    
    ocrOutputs(final_img, new_bboxes) 
    
    displayMachedBoxes(final_img, new_bboxes)
    
    #utlis.displayAllBoxes(final_img, bbox_coordinates)
    
    plt.figure()
    plt.title("final_img")
    plt.imshow(final_img)
    plt.figure()
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    #plt.figure()
    #plt.imshow(img_res)
    plt.show()
   
        

