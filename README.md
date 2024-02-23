# Usage
### Arguments
* `--folder_name`: folder path
* `--neighbor_box_distance`: Nearest box distance 
* `--face_recognition`: Face recognition method (dlib, ssd, haar)
* `--rotation_interval`: Id card rotation interval in degrees
* `--ocr_method`: ocr method (EasyOcr and TesseractOcr)

In Dlib and Haar face detection model, it is better to choose a rotation angle of less than 30 degrees, otherwise no face may be detected due to image inversion.
Create a folder and put the ID card images in that folder
``` 
git clone git@github.com:musimab/Tc_ID_Card_OCR.git
mkdir images
``` 

``` 
python3 main.py --folder_name "images" --neighbor_box_distance 60 --face_recognition ssd --ocr_method EasyOcr --rotation_interval 60
```
## create python3 virtual enviroment and install dependencies
```
python3 -m venv card_id_ocr_venv
```

```
source card_id_ocr_venv/bin/activate
```

```
pip3 install -r requirements.txt
```

The result image and cropped regions will be saved to `./outputs` by default.
The json data will be saved to `./test` by default.

## TODOs
1. deep learning based (Yolo SSD Faster Rcnn) identity card recognition model will be developed


## Algorithm Pipeline
![ocr_pip_update1](https://user-images.githubusercontent.com/47300390/158075210-1ed286a2-f8ee-4007-b158-51fcc342add3.jpg)

## Input image
![ori14_m2rot](https://user-images.githubusercontent.com/47300390/160571547-236fd7ed-bbf0-40f9-b01a-d8e9d2edfc11.jpg)

## Warped image
![warped_img](https://user-images.githubusercontent.com/47300390/160571705-44b19acb-920b-4c2e-b792-7983a8a35bff.jpg)


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
