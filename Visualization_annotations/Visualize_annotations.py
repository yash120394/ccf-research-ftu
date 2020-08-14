# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:59:02 2020

@author: yash1
"""

# Loading required libraries
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from read_roi import read_roi_file
from read_roi import read_roi_zip
from PIL import Image, ImageDraw
import itertools
from scipy.ndimage.measurements import label


# Opening annotated json file from Qupath
annots_jash = open('VAN0008Jash.json') 
annots_leah = open('VAN0008Leah.json') 

# convert into dictionary object
data_jash = json.load(annots_jash) 
data_leah = json.load(annots_leah) 

# saving all coordinates in a list
coords_jash=[]
coords_leah=[]


for i in data_jash:
    coords_jash.extend(i['geometry']['coordinates'])
    
    
for i in data_leah:
    coords_leah.extend(i['geometry']['coordinates'])
    
    
# Opening annotated .roi object from ImageJ
# to handle decompression bomb
Image.MAX_IMAGE_PIXELS = None

#reading roi files
rois_sumeet = read_roi_zip("VAN0008Sumeet.zip")
rois_yash = read_roi_zip("VAN0008Yash.zip")



# Read corresponding images
img = cv2.imread('VAN0008-RK-403-100-PAS_registered.ome.jpg')


# For Counting number of glomeruli
# Create binary mask array while iterating over the glomeruli points
jash_array = np.zeros(img.shape[:2])
leah_array = np.zeros(img.shape[:2])
yash_array = np.zeros(img.shape[:2])
sumeet_array = np.zeros(img.shape[:2])

for i in coords_jash:
    pts_jash = np.array(i, np.int32)
    pts_jash = pts_jash.reshape((-1,1,2))
    cv2.fillPoly(jash_array,[pts_jash],1)

for i in coords_leah:
    pts_leah = np.array(i, np.int32)
    pts_leah = pts_leah.reshape((-1,1,2))
    cv2.fillPoly(leah_array,[pts_leah],1)
    
yash_pil = Image.fromarray(yash_array)
draw = ImageDraw.Draw(yash_pil)
for key in rois_yash.keys():
    left = rois_yash[key]['left']
    top = rois_yash[key]['top']
    width = rois_yash[key]['width']
    height = rois_yash[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill=1)
yash_array1 = np.array(yash_pil)


sumeet_pil = Image.fromarray(sumeet_array)
draw = ImageDraw.Draw(sumeet_pil)
for key in rois_sumeet.keys():
    left = rois_sumeet[key]['left']
    top = rois_sumeet[key]['top']
    width = rois_sumeet[key]['width']
    height = rois_sumeet[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill=1)
sumeet_array1 = np.array(sumeet_pil)


# Create combined array by adding all arrays
comb_array = yash_array1 + sumeet_array1 + jash_array + leah_array

# Counting number of glomeruli for all areas
comb_one = (comb_array>=1)*255 
comb_two = (comb_array>=2)*255 
comb_three = (comb_array>=3)*255 
comb_four = (comb_array>=4)*255 

structure = np.ones((3, 3), dtype=np.int)

# Max ground truth annotated glomeruli : 174 (Leah)

labeled_one, ncomponents_one = label(comb_one, structure)
print("Number of Glomeruli for atleast outermost overlap:", ncomponents_one)
# 179

labeled_two, ncomponents_two = label(comb_two, structure)
print("Number of Glomeruli for second outermost overlap:", ncomponents_two)
# 165

labeled_three, ncomponents_three = label(comb_three, structure)
print("Number of Glomeruli for third outermost overlap:", ncomponents_three)
# 131

labeled_four, ncomponents_four = label(comb_four, structure)
print("Number of Glomeruli for innermost overlap:", ncomponents_four)
# 126








# Visualization
# Create binary/mask images for each human annotation
bin_mask_j = np.zeros(img.shape[:2])
bin_mask_l = np.zeros(img.shape[:2])
bin_mask_y = np.zeros(img.shape[:2])
bin_mask_s = np.zeros(img.shape[:2])

mask_j = img.copy()
mask_l = img.copy()
mask_y = img.copy()
mask_s = img.copy()

# Creating binary mask for Jashjeet's annotation
# Creating overlay mask on original image for Jashjeet's annotation (green)
for i in coords_jash:
    pts_jash = np.array(i, np.int32)
    pts_jash = pts_jash.reshape((-1,1,2))
    cv2.fillPoly(bin_mask_j,[pts_jash],1)
    cv2.fillPoly(mask_j,[pts_jash],color=[0,255,0])
    
bin_mask_j = bin_mask_j*255
bin_mask_j = bin_mask_j.astype(np.uint8)
    
# Plot binary and mask and save corresponding image
plt.imshow(bin_mask_j)
plt.imshow(mask_j)

cv2.imwrite('bin_VAN008_j.jpg',bin_mask_j)
cv2.imwrite('VAN008_j.jpg',mask_j)


# Creating binary mask for Leah's annotation
# Creating overlay mask on original image for Leah's annotation (red)
for i in coords_leah:
    pts_leah = np.array(i, np.int32)
    pts_leah = pts_leah.reshape((-1,1,2))
    cv2.fillPoly(bin_mask_l,[pts_leah],1)
    cv2.fillPoly(mask_l,[pts_leah],color=[0,0,255])
    
bin_mask_l = bin_mask_l*255
bin_mask_l = bin_mask_l.astype(np.uint8)
    
# Plot binary and mask and save corresponding image
plt.imshow(bin_mask_l)
plt.imshow(mask_l)

cv2.imwrite('bin_VAN008_l.jpg',bin_mask_l)
cv2.imwrite('VAN008_l.jpg',mask_l)


# Creating binary mask for Yash's annotation
# Creating overlap mask on original image for Yash's annotation (blue)
mask_y_pil = Image.fromarray(mask_y)
bin_mask_y = bin_mask_y.astype(np.uint8)
bin_mask_y_pil = Image.fromarray(bin_mask_y)

draw = ImageDraw.Draw(mask_y_pil)
bin_draw = ImageDraw.Draw(bin_mask_y_pil)

for key in rois_yash.keys():
    left = rois_yash[key]['left']
    top = rois_yash[key]['top']
    width = rois_yash[key]['width']
    height = rois_yash[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill='blue')
    bin_draw.ellipse(bbox, fill='white')    




mask_y_pil.save("VAN008_y.jpg")
bin_mask_y_pil.save("bin_VAN008_y.jpg")


# Creating binary mask for Sumeet's annotation
# Creating overlap mask on original image for Sumeet's annotation (yellow)
mask_s_pil = Image.fromarray(mask_s)
bin_mask_s = bin_mask_s.astype(np.uint8)
bin_mask_s_pil = Image.fromarray(bin_mask_s)

draw = ImageDraw.Draw(mask_s_pil)
bin_draw = ImageDraw.Draw(bin_mask_s_pil)

for key in rois_sumeet.keys():
    left = rois_sumeet[key]['left']
    top = rois_sumeet[key]['top']
    width = rois_sumeet[key]['width']
    height = rois_sumeet[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill='yellow')
    bin_draw.ellipse(bbox, fill='white')    




mask_s_pil.save("VAN008_s.jpg")
bin_mask_s_pil.save("bin_VAN008_s.jpg")



# Create an overlay of all annotation onto one image (Using Pillow)
mask_comb = img.copy()
mask_comb_pil = Image.fromarray(mask_comb)
draw = ImageDraw.Draw(mask_comb_pil)

for i in coords_jash:
    pts_jash_list = list(itertools.chain(*i))
    draw.polygon(pts_jash_list,fill = (0,255,0))

for i in coords_leah:
    pts_leah_list = list(itertools.chain(*i))
    draw.polygon(pts_leah_list,fill = (255,0,0))
    
for key in rois_sumeet.keys():
    left = rois_sumeet[key]['left']
    top = rois_sumeet[key]['top']
    width = rois_sumeet[key]['width']
    height = rois_sumeet[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill = (255,255,0))
    
for key in rois_yash.keys():
    left = rois_yash[key]['left']
    top = rois_yash[key]['top']
    width = rois_yash[key]['width']
    height = rois_yash[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, fill = (0,0,255))


mask_comb_pil.save("VAN008_comb.jpg")


# Create an overlay outlines of all annotation onto one image (Using Pillow)
mask_comb = img.copy()
mask_comb_pil = Image.fromarray(mask_comb)
draw = ImageDraw.Draw(mask_comb_pil)

for i in coords_jash:
    pts_jash_list = list(itertools.chain(*i))
    draw.polygon(pts_jash_list,outline = (0,255,0))

for i in coords_leah:
    pts_leah_list = list(itertools.chain(*i))
    draw.polygon(pts_leah_list,outline = (255,0,0))
    
for key in rois_sumeet.keys():
    left = rois_sumeet[key]['left']
    top = rois_sumeet[key]['top']
    width = rois_sumeet[key]['width']
    height = rois_sumeet[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = (255,255,0),width=10)
    
for key in rois_yash.keys():
    left = rois_yash[key]['left']
    top = rois_yash[key]['top']
    width = rois_yash[key]['width']
    height = rois_yash[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = (0,0,255),width=5)


mask_comb_pil.save("VAN008_comb_outline.jpg")



# Create an overlay outlines of all annotation into one image (Using OpenCV)
mask_comb = img.copy()

for i in coords_jash:
    pts_jash = np.array(i, np.int32)
    pts_jash = pts_jash.reshape((-1,1,2))
    cv2.polylines(mask_comb,[pts_jash],True,(0,255,0),10)

for i in coords_leah:
    pts_leah = np.array(i, np.int32)
    pts_leah = pts_leah.reshape((-1,1,2))
    cv2.polylines(mask_comb,[pts_jash],True,(0,0,255),10)
    
for key in rois_sumeet.keys():
    left = rois_sumeet[key]['left']
    top = rois_sumeet[key]['top']
    width = rois_sumeet[key]['width']
    height = rois_sumeet[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    cv2.ellipse(mask_comb, (centreX,centreY), (widthX,heightY), 0, 0, 360, (0,255,255), 15)
    
for key in rois_yash.keys():
    left = rois_yash[key]['left']
    top = rois_yash[key]['top']
    width = rois_yash[key]['width']
    height = rois_yash[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    cv2.ellipse(mask_comb, (centreX,centreY), (widthX,heightY), 0, 0, 360, (255,0,0), 10)
    
    


cv2.imwrite("VAN008_comb_outline1.jpg",mask_comb)


# Find number of glomeruli with number of overlaps
img_jash = np.zeros(img.shape[:2])
img_leah = np.zeros(img.shape[:2])
img_yash = np.zeros(img.shape[:2])
img_sumeet = np.zeros(img.shape[:2])



# Create binary image 
































    