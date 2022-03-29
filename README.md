# Usage
### Arguments
* `--folder_name`: folder path
* `--neighbor_box_distance`: Nearest box distance 
* `--face_recognition`: Face recognition method 
* `--rotation_interval`: Id card rotation interval in degrees

Create a folder and put the ID card images in that folder
``` 
git clone git@github.com:musimab/Tc_ID_Card_OCR.git
mkdir images
``` 

``` 
python3 main.py --folder_name "images" --neighbor_box_distance 60 --face_recognition ssd --rotation_interval 60
```

```
pip install opencv-python-headless==4.5.3.56
pip install craft-text-detector
pip install easyocr
```

The result image and cropped regions will be saved to `./outputs` by default.
The json data will be saved to `./test` by default.

## TODOs
1. deep learning based (Yolo SSD Faster Rcnn) identity card recognition model will be developed


## Algorithm Pipeline
![ocr_pip_update1](https://user-images.githubusercontent.com/47300390/158075210-1ed286a2-f8ee-4007-b158-51fcc342add3.jpg)

## Input image
![ori_img](https://user-images.githubusercontent.com/47300390/160566593-3ed44b00-5da6-4d9d-a445-30f8981526e9.jpg)

## Warped image
![warped_img](https://user-images.githubusercontent.com/47300390/160566682-b75ce5c3-ee6f-4dc2-b671-17c8ca003a70.jpg)


## CRAFT Character Density Map
![txt_heat_map ](https://user-images.githubusercontent.com/47300390/160567945-05428b53-cb64-4a39-b230-1b0ef95ef3da.jpg)

## Unet Output for character density map
![maskem](https://user-images.githubusercontent.com/47300390/160568696-2f9b1b82-8be4-462d-85dd-cfa4373997e1.png)

## Craft Output(red boxes) and Matched Boxes(blue boxes)

![final_imgp](https://user-images.githubusercontent.com/47300390/160568107-0ac52940-797e-4a00-9702-610d8e4f305c.jpg)



## Ocr Output

Tc : 12345678909
Surname : MUSTAFA ALİ
Name : YILMAZ
DateofBirth : 07071999

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
