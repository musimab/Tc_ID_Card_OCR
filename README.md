# Usage

## TODOs
1. deep learning based (Yolo SSD Faster Rcnn) identity card recognition model will be developed


## Algorithm Pipeline
![ocr_pip_update1](https://user-images.githubusercontent.com/47300390/158075210-1ed286a2-f8ee-4007-b158-51fcc342add3.jpg)

## Input image
![tcocr](https://user-images.githubusercontent.com/47300390/158068982-1eb2394c-ae25-480c-9e2f-d0f1ebe69361.jpeg)

## Rotated image
![final_img](https://user-images.githubusercontent.com/47300390/158069046-333e582c-0b98-469a-a908-3e210e4c8b3a.jpg)


## CRAFT Character Density Map
![txt_heat_map](https://user-images.githubusercontent.com/47300390/158069076-5b91e198-f65a-4599-ac61-322b8a9b3e23.jpg)

## Unet Output for character density map
![predicted_mask_fin](https://user-images.githubusercontent.com/47300390/158069105-e6b330ee-4f62-4431-96a5-9f5a09d97cf1.jpg)

## Craft Output
![image_text_detection](https://user-images.githubusercontent.com/47300390/158069125-5480a4fd-31b9-401d-b00b-081ed761ee09.png)

## Matched Boxes

![final_img_](https://user-images.githubusercontent.com/47300390/158069152-af8848d4-510e-4073-85d2-0a72e3d2f35c.jpg)

## Cropped Roi
![DateofBirth](https://user-images.githubusercontent.com/47300390/158069210-f2567e85-1635-41d1-b389-62e37d02bb63.jpg)
![Name](https://user-images.githubusercontent.com/47300390/158069223-583bee9a-534e-414f-aaff-32d595c899eb.jpg)
![Surname](https://user-images.githubusercontent.com/47300390/158069227-81ffb2ee-7c36-4e0a-861d-47b2260ace9b.jpg)
![Tc](https://user-images.githubusercontent.com/47300390/158069231-0d051bd9-5e57-483f-a67d-bd1087873f97.jpg)

## Ocr Output

Tc : 12345678901
Surname : YILMAZ
Name : MEHMET
DateofBirth : 01.01.1990

## Ocr Evaluation

The accuracy of the optical character system was evaluated according to 2 different criteria. The first of these is accuracy at the word level and the other is accuracy at the character level.

The evaluate.py function retrieves the predicted and actual values in json format

###  Character Level Comparision  
1. tc: 1303 / 1327  => 98.19 %
2. surname: 805 / 816 => 98.65 %
3. name: 742 / 746 => 99.46 % 
4. dateofbirth: 976 / 976 => 100.0 % 

###  Word Level Comparision  
1. tc : 0.96 %
2. surname : 0.91 %
3. name : 0.95 %
4. date: 1.0 %

### Easy Ocr
https://github.com/sarra831/EasyOCR
