
import cv2
from matplotlib import pyplot as plt
import numpy as np
import utlis
from find_nearest_box import NearestBox
from pytorch_unet.unet_predict import UnetModel
from extract_words import Image2Text
import os


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


def getCenterOfMasks(thrsh):
    
    thresh = thrsh.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][1], reverse=False))

    print("len of countours:", len(cnts))
    
    detected_centers = []
    indx = 0
    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        if w > 10 and h > 5:
            cv2.rectangle(thrsh, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cX = round(int(x) + w/2.0)
            cY = round(int(y) + h/2.0)
            detected_centers.append((cX, cY))
            cv2.circle(thrsh, (cX, cY), 7, (255, 0, 0), -1)
            indx = indx + 1
        if(indx == 4):
            break
    print("len of detected centers:", len(detected_centers))
    #plt.imshow(thresh, cmap='gray')
    #plt.show()
    return np.array(detected_centers)


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


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        print("filename:", filename)
        if img is not None:
            images.append(img)
    
    return images


if '__main__' == __name__:
    
    
    #img1 = cv2.imread("images/me.jpeg")
    #img1 = cv2.imread("images/tcocr.jpeg")
    #img1 = cv2.imread("images/wifetc.JPG")
    
    ORI_THRESH = 3
    model = UnetModel("resnet34", "cuda")
    nearestBox = NearestBox(distance_thresh = 10, draw_line=True)
    
    folder = "tc"

    for filename in sorted(os.listdir(folder)):
        
        img = cv2.imread(os.path.join(folder,filename))
        img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
       
        final_img = changeOrientationUntilFaceFound(img1)
        
        #final_img = utlis.correctPerspective(final_img)
        
        #final_img = cv2.resize(final_img, (480,640))
    

        txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
        txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
        
        predicted_mask = model.predict(txt_heat_map)
    
        orientation_angle = utlis.findOrientationofLines(predicted_mask.copy())
        print("orientation_angle is ", orientation_angle)
        
        if ( abs(orientation_angle) > ORI_THRESH ):
            
            print("absulute orientation_angle is greater than {}".format(ORI_THRESH)  )
            
            final_img = utlis.rotateImage(orientation_angle, final_img)
        
            txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
            txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
            predicted_mask = model.predict(txt_heat_map)
        
        
        bbox_coordinates , box_centers = getBoxRegions(regions)
    
        mask_centers = getCenterOfMasks(predicted_mask)
        
        # centers ratio for 4 boxes
        centers_ratio_mask = getCenterRatios(predicted_mask, mask_centers) 

        # centers ratio for all boxes
        centers_ratio_all = getCenterRatios(final_img, box_centers) 
    
        matched_box_indexes = matchCenters(centers_ratio_mask , centers_ratio_all)
        
        
        new_bboxes = nearestBox.searchNearestBoundingBoxes(bbox_coordinates, matched_box_indexes, final_img)
        
        ocrResult = Image2Text(ocr_method="Easy", lw_thresh=5, up_thresh=5, denoising=False, file_name=filename)
        PersonInfo = ocrResult.ocrOutput(final_img, new_bboxes)
        
        for id, val in PersonInfo.items():
            print(id,':' ,val)
        
        #utlis.displayMachedBoxes(final_img, new_bboxes)
        
        #utlis.displayAllBoxes(final_img, bbox_coordinates)
        
        #plt.figure()
        #plt.title("final_img")
        #plt.imshow(final_img)
    
        #plt.figure()
        #plt.title("Predicted Mask")
        #plt.imshow(predicted_mask, cmap='gray')
        #plt.show()

   
        

