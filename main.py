
import cv2
from matplotlib import pyplot as plt
import numpy as np
import utlis
from find_nearest_box import NearestBox
from pytorch_unet.unet_predict import UnetModel
from extract_words import Image2Text
import os
from detect_face import FindFaceID
import time

def getCenterRatios(img, centers):
    """
    Calculates the position of the centers of all boxes 
    in the ID card image and Unet Mask relative to the width and height of the image 
    and returns these ratios as a numpy array.
    """
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
    """
    It takes the ratio of the centers of the regions 
    included in the mask and CRAFT result on the image 
    and maps them according to the absolute distance. 
    Returns the index of the centers with the lowest absolute difference accordingly
    """

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
    """
    Find centers of 4 boxes in mask from top to bottom with unet model output and return them
    """
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #sort the contours according to size from min to max
    contours = sorted(contours, key = cv2.contourArea, reverse=False)
    
    contours = contours[-4:] # get 4 biggest contour

    #print("size of cnt", [cv2.contourArea(cnt) for cnt in contours])
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort to 4 biggest contours from top to bottom
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][1], reverse=False))
    
    detected_centers = []
    #indx = 0
    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        detected_centers.append((cX, cY))
        #cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)
        #indx = indx + 1
        #if(indx == 4):
        #    break
    #print("len of detected centers:", len(detected_centers))
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    return np.array(detected_centers)


def getBoxRegions(regions):
    """
    The coordinates of the texts on the id card are converted 
    to x, w, y, h type and the centers and coordinates of these boxes are returned.
    """
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



if '__main__' == __name__:
    
    Folder = "tc"
    ORI_THRESH = 3
    model = UnetModel("resnet34", "cuda")
    nearestBox = NearestBox(distance_thresh = 10, draw_line=True)
    findFaceID = FindFaceID(detection_method = "ssd", rot_interval= 30)
    
    
    start = time.time()

    for filename in sorted(os.listdir(Folder)):
        
        img = cv2.imread(os.path.join(Folder,filename))
        img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
  
        final_img = findFaceID.changeOrientationUntilFaceFound(img1)
        
        final_img = utlis.correctPerspective(final_img)
    
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
        
        ocrResult = Image2Text(ocr_method="Easy", lw_thresh = 5, up_thresh= 5, denoising=False, file_name=filename)
        
        PersonInfo = ocrResult.ocrOutput(final_img, new_bboxes)
        print(" ")
        for id, val in PersonInfo.items():
            print(id,':' ,val)
        print(" ")
        #utlis.displayMachedBoxes(final_img, new_bboxes)
        
        #utlis.displayAllBoxes(final_img, bbox_coordinates)
        
      
        #plt.title("final_img")
        #plt.imshow(final_img)
        #plt.show()
    
       
        #plt.title("Predicted Mask")
        #plt.imshow(predicted_mask, cmap='gray')
        #plt.show()
    
    end = time.time()
    print("Execution Time:", (end -start))
   
        

