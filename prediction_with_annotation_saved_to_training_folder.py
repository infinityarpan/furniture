#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import json
import base64
import re
import os
import cv2
import pickle
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

######## Load path to testing images
in_path = "/home/arpan/Downloads/mix2.jpeg"
######## Load the model weight path
filename = '/home/arpan/Downloads/test_detectron/detectitems/config.pkl'
cfg = get_cfg()
with open(filename, 'rb') as f:
    cfg = pickle.load(f)
predictor = DefaultPredictor(cfg)

######## Set the path to save the new image along with the annotation file
out_path = "/home/arpan/Downloads/test_detectron/items/train/"

data = {}
######## Create lists for storing the filenames of the stored images in the training dataset 
######## to avoid overwriting of data while saving annotations 
bed = []
chair = []
dresser = []
lamps = []
sofa = []
table = []

inp_img = Image.open(in_path)
img = np.array(inp_img)[:,:,::-1]
height = img.shape[0]
width = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
outputs = predictor(img)
list = outputs['instances']
# print(list.get_fields())
pred_classes = list.get_fields()["pred_classes"]
pred_masks = list.get_fields()["pred_masks"]
classes = pred_classes.to("cpu").numpy()

######## search the class from the pred_classes tensor value
item_dict = {0:'bed', 1:'chair', 2:'dresser', 3:'lamps', 4:'sofa', 5:'table'}
# print(item_dict)
pred_value = None
for key, value in item_dict.items():
    if key == classes:
        pred_value = value

new_masker = pred_masks.to("cpu").numpy().astype(np.uint8)[0]
if new_masker.size == 0:
    print("No items detected")
######## If something is predicted
else:
    print(f"It's a {pred_value}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(new_masker.shape[0]):        #for each column
        for j in range(new_masker.shape[1]):    #For each row
            if new_masker[i,j] == 0:
                img[i,j] = 255    #set the background of the detected item to black
                

    contours, hierarchy = cv2.findContours(new_masker[0] ,cv2.RETR_TREE, cv2. CHAIN_APPROX_SIMPLE)

######## LESS number of contour points are generated for JSON and easy to modify annotation
    list_points = []
    for cnt in contours : 
  
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
#         print(approx)
#         print(type(approx))
        for points in approx:
            flatten_points = points.flatten().tolist()
            list_points.append(flatten_points)
#             print(flatten_points)
#     print(list_points)
########

######## draws boundary of contours for LESS NUMBER OF POINTS 
#         cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)  
########    

######## draws boundary of contours for MORE NUMBER OF POINTS
#     cv2.drawContours(img, contours, 0, (0, 0, 255), 5)  

######## MORE number of contour points are generated for JSON but difficult to modify annotation
#     n_points = contours[0]
#     for points in n_points:
#         flatten_list = points.flatten().tolist()
#         d.append(flatten_list)
########
        
#     cv2.imshow('Contours', img) 
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows() 

    encoded = base64.b64encode(open(in_path, "rb").read())
    decoded = encoded.decode("utf-8")

    data = {}
    data["version"] = "3.16.7"
    data["flags"] = {}
    data["shapes"] = [{"label": pred_value, "line_color": None,"fill_color": None,"points": list_points, "shape_type": "polygon", "flags": {}}]
    data["lineColor"] = [0,255,0,128]
    data["fillColor"] = [255,0,0,128]
    data["imagePath"] = f"{pred_value}.jpeg"
    data["imageData"] = decoded
    data["imageHeight"] = height
    data["imageWidth"] = width
    # print(data)

####### for saving the files

    for file in os.listdir(out_path):
        file = file.split(".")
        if file[1] == 'json':
#         print(file)
            temp = re.compile("([a-zA-Z]+)([0-9]+)") 
            res = temp.match(file[0]).groups()
            if res[0] == 'bed':
                bed.append(int(res[1]))
            elif res[0] == 'sofa':
                sofa.append(int(res[1]))
            elif res[0] == 'dressers':
                dresser.append(int(res[1]))
            elif res[0] == 'table':
                table.append(int(res[1]))
            elif res[0] == 'lamp':
                lamps.append(int(res[1]))
            elif res[0] == 'chair':
                chair.append(int(res[1]))

######## save image and annotations to the training folder with filename followed by the highest number in serial (example: sofa62)
    index_items = ['bed', 'chair', 'sofa', 'lamps', 'dresser', 'table']
    list_items = [bed, chair, sofa, lamps, dresser, table]
    for itms in index_items:
        if pred_value == itms:
            index_pos = index_items.index(itms)
            max_val = max(list_items[index_pos])

    inp_img.save(f"/home/arpan/Downloads/test_detectron/items/train/{pred_value}{max_val+1}.jpeg")        

    with open(f"{out_path}/{pred_value}{max_val+1}.json", 'w') as outfile:
        json.dump(data, outfile, indent=4)
########

# resize image according to the sub models for different styles of furnitures
    print(f"New image {pred_value}{max_val+1}.jpeg and annotations {pred_value}{max_val+1}.json is saved at {out_path}")
    img = cv2.resize(img, (224, 224))
#     print(img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




