import easyocr
import keras_ocr
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import os

TesseractOcrInfo = {

        "Tc": ' ',
        "Surname": ' ',
        "Name": ' ',
        "DateofBirth":' '
}

KerasOcrInfo = {

        "Tc": ' ',
        "Surname": ' ',
        "Name": ' ',
        "DateofBirth":' '
}

CardInfo = {}


EasyOcrInfo = {}


class SaveOcrInfo:

    def __init__(self) -> None:
        
        self.EasyOcrInfo = {}
        self.data_path = 'test/predictions_json'
        self.dict_path = self.data_path +  '/data.json'
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def saveDict(self, dict_name):
        
        with open(self.dict_path, 'w', encoding='utf-8') as fp:
            json.dump(dict_name, fp, ensure_ascii = False)
    
    def loadDict(self):
        
        with open(self.dict_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            print(data)
    


class Image2Text():
    
    def __init__(self, ocr_method, lw_thresh = 3, up_thresh = 3, denoising = False, file_name=None) -> None:
        
        self.ocr_method = ocr_method
        self.lw_thresh = lw_thresh
        self.up_thresh = up_thresh
        self.denoise = denoising
        self.img_name = file_name
        self.crop_img = list()
        self.crop_img_names = []
        #self.pipeline = keras_ocr.pipeline.Pipeline()
        self.reader = easyocr.Reader(['tr','en'])

        self.IdCardInfo = SaveOcrInfo()
    
    def ocrOutput(self, img, bbox):
        
        ocr_output = None

        if(self.ocr_method == "Tesseract"):
            ocr_output = self.tesserctOcr(img, bbox)
        
        elif (self.ocr_method == "Keras"):
            ocr_output = self.kerasOcr(img, bbox)
       
        elif (self.ocr_method == "Easy"):
            ocr_output = self.easyOcr(img, bbox)
        
        else:
            print("Select Keras Easy or Tesseract")
            return
        
        return ocr_output

    def cropRoi(self, img, bbox):
       
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        id_infos= ["Tc", "Surname", "Name", "DateofBirth"]
       
        for  info, box in zip(id_infos, bbox):
            x, w, y, h = box
            crop_img = img_rgb[y-self.lw_thresh :y+h+self.up_thresh, x-self.lw_thresh:x + w + self.up_thresh]
            
            if self.denoise:
                crop_img = self.denoiseImage(crop_img)
            
            self.crop_img.append(crop_img)
            crop_name = "outputs/target_crops/" + str(info) +".jpg"
            plt.imsave(crop_name,crop_img)
            self.crop_img_names.append(crop_name)

    
    def easyOcr(self, img, bbox):

        self.cropRoi(img, bbox)
        id_infos= ["Tc", "Surname", "Name", "DateofBirth"]

        for info, img  in zip(id_infos, self.crop_img_names):
            result = self.reader.readtext(img)
            if(len(result)):
                box, text, prob = result[0]
                self.IdCardInfo.EasyOcrInfo[info] = text
        
        self.IdCardInfo.EasyOcrInfo["DateofBirth"] = self.getonlyDigits(self.IdCardInfo.EasyOcrInfo["DateofBirth"])
        
        CardInfo[self.img_name] = self.IdCardInfo.EasyOcrInfo
        self.IdCardInfo.saveDict(CardInfo)
        
        
        return self.IdCardInfo.EasyOcrInfo


    def tesserctOcr(self,img, bbox):
        
        self.cropRoi(img, bbox)
        
        for info, crop_img in zip(TesseractOcrInfo, self.crop_img):
            TesseractOcrInfo[info] = pytesseract.image_to_string(crop_img)

        return TesseractOcrInfo

    """
    def kerasOcr(self,img, bbox):
       
        self.cropRoi(img, bbox)
        
        images = [keras_ocr.tools.read(img) for img in self.crop_img_names]
    
        prediction_groups = self.pipeline.recognize(images)
        for info, img_region in zip(KerasOcrInfo, prediction_groups):
            new_text = ""
            if(len(img_region) == 1):
                for text, box in img_region:
                    KerasOcrInfo[info] = text
            else:
                for text, box in img_region:
                    new_text = text + new_text
                KerasOcrInfo[info] = new_text

        return KerasOcrInfo 
    
    """
    def denoiseImage(self, img):
    
        img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        imgGray = cv2.cvtColor(img_denoise , cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
        ret, imgf = cv2.threshold(imgBlur , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #imgf contains Binary image
        
        kernel = np.ones((1,1), np.uint8)
        img_dilation = cv2.dilate( imgf, kernel, iterations=1)
        img_erosion = cv2.erode(img_dilation , kernel, iterations=1)
        
        #img_erosion = cv2.resize(img_erosion ,(img_erosion.shape[1]+10, img_erosion.shape[0]+5))
        return img_erosion
    
    def getonlyDigits(self,inp_str):
        
        print("Original String : " + inp_str) 
        num = ""
        for c in inp_str:
            if c.isdigit():
                num = num + c
        return num

            