# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:13:38 2020

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
import skimage.measure as sk
import pandas as pd


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


# Visualize glom size of Jash annotation
# Create binary mask array while iterating over the glomeruli points
jash_array = np.zeros(img.shape[:2])
leah_array = np.zeros(img.shape[:2])

for i in coords_jash:
    pts_jash = np.array(i, np.int32)
    pts_jash = pts_jash.reshape((-1,1,2))
    cv2.fillPoly(jash_array,[pts_jash],1)

for i in coords_leah:
    pts_leah = np.array(i, np.int32)
    pts_leah = pts_leah.reshape((-1,1,2))
    cv2.fillPoly(leah_array,[pts_leah],1)





# Get properties of Jash annotation of Glomeruli
jash_annots = (jash_array*255).astype('uint8')
jash_label = sk.label(jash_annots)
jash_prop = sk.regionprops(label_image=jash_label)


jash_table = pd.DataFrame(index = range(len(jash_prop)), columns = ['area','major_axis','minor_axis'])
for i in range(len(jash_prop)):
    jash_table.iloc[i,0] = jash_prop[i].area
    jash_table.iloc[i,1] = jash_prop[i].major_axis_length
    jash_table.iloc[i,2] = jash_prop[i].minor_axis_length
    
jash_table1 = jash_table.copy()

jash_table1['major_axis'] = jash_table['major_axis'].astype('float64') 
jash_table1['minor_axis'] = jash_table['minor_axis'].astype('float64') 
jash_table1['area'] = jash_table['area'].astype('float64') 




# Histogram plots    
plt.hist(jash_table['major_axis'],bins=20)
plt.savefig('./Viz_jash/hist_max_len.jpg')

plt.hist(jash_table['minor_axis'],bins=20)
plt.savefig('./Viz_jash/hist_min_len.jpg')

plt.hist(jash_table['area'],bins=20)
plt.savefig('./Viz_jash/hist_area.jpg')


# Box plots
plt.boxplot(jash_table['major_axis'])
plt.savefig('./Viz_jash/box_max_len.jpg')

plt.boxplot(jash_table['minor_axis'])
plt.savefig('./Viz_jash/box_min_len.jpg')

plt.boxplot(jash_table['area'])
plt.savefig('./Viz_jash/box_area.jpg')





# Get properties of Leah annotation of Glomeruli
leah_annots = (leah_array*255).astype('uint8')
leah_label = sk.label(leah_annots)
leah_prop = sk.regionprops(label_image=leah_label)


leah_table = pd.DataFrame(index = range(len(leah_prop)), columns = ['area','major_axis','minor_axis'])
for i in range(len(leah_prop)):
    leah_table.iloc[i,0] = leah_prop[i].area
    leah_table.iloc[i,1] = leah_prop[i].major_axis_length
    leah_table.iloc[i,2] = leah_prop[i].minor_axis_length


leah_table1 = leah_table.copy()

leah_table1['major_axis'] = leah_table['major_axis'].astype('float64') 
leah_table1['minor_axis'] = leah_table['minor_axis'].astype('float64') 
leah_table1['area'] = leah_table['area'].astype('float64') 


leah_table1.describe()

    

# Histogram plots    
plt.hist(leah_table['major_axis'],bins=20)
plt.savefig('./Viz_leah/hist_max_len.jpg')

plt.hist(leah_table['minor_axis'],bins=20)
plt.savefig('./Viz_leah/hist_min_len.jpg')

plt.hist(leah_table['area'],bins=20)
plt.savefig('./Viz_leah/hist_area.jpg')


# Box plots
plt.boxplot(leah_table['major_axis'])
plt.savefig('./Viz_leah/box_max_len.jpg')

plt.boxplot(leah_table['minor_axis'])
plt.savefig('./Viz_leah/box_min_len.jpg')

plt.boxplot(leah_table['area'])
plt.savefig('./Viz_leah/box_area.jpg')



# Visualize glom size of Yash annotation
# Create binary mask array while iterating over the glomeruli points
yash_array = np.zeros(img.shape[:2])
sumeet_array = np.zeros(img.shape[:2])


yash_img = Image.fromarray(yash_array)

yash_draw = ImageDraw.Draw(yash_img)

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
    yash_draw.ellipse(bbox, fill='white')    

yash_array1 = np.array(yash_img)
yash_array1[yash_array1 > 0] = 1



sumeet_img = Image.fromarray(sumeet_array)

sumeet_draw = ImageDraw.Draw(sumeet_img)

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
    sumeet_draw.ellipse(bbox, fill='white')    

sumeet_array1 = np.array(sumeet_img)
sumeet_array1[sumeet_array1 > 0] = 1



# Get properties of Yash annotation of Glomeruli
yash_annots = (yash_array1*255).astype('uint8')
yash_label = sk.label(yash_annots)
yash_prop = sk.regionprops(label_image=yash_label)


yash_table = pd.DataFrame(index = range(len(yash_prop)), columns = ['area','major_axis','minor_axis'])
for i in range(len(yash_prop)):
    yash_table.iloc[i,0] = yash_prop[i].area
    yash_table.iloc[i,1] = yash_prop[i].major_axis_length
    yash_table.iloc[i,2] = yash_prop[i].minor_axis_length
    
    
yash_table1 = yash_table.copy()

yash_table1['major_axis'] = yash_table['major_axis'].astype('float64') 
yash_table1['minor_axis'] = yash_table['minor_axis'].astype('float64') 
yash_table1['area'] = yash_table['area'].astype('float64') 


yash_table1.describe()



    

# Histogram plots    
plt.hist(yash_table['major_axis'],bins=20)
plt.savefig('./Viz_yash/hist_max_len.jpg')

plt.hist(yash_table['minor_axis'],bins=20)
plt.savefig('./Viz_yash/hist_min_len.jpg')

plt.hist(yash_table['area'],bins=20)
plt.savefig('./Viz_yash/hist_area.jpg')


# Box plots
plt.boxplot(yash_table['major_axis'])
plt.savefig('./Viz_yash/box_max_len.jpg')

plt.boxplot(yash_table['minor_axis'])
plt.savefig('./Viz_yash/box_min_len.jpg')

plt.boxplot(yash_table['area'])
plt.savefig('./Viz_yash/box_area.jpg')





# Get properties of Sumeet annotation of Glomeruli
sumeet_annots = (sumeet_array1*255).astype('uint8')
sumeet_label = sk.label(sumeet_annots)
sumeet_prop = sk.regionprops(label_image=sumeet_label)


sumeet_table = pd.DataFrame(index = range(len(sumeet_prop)), columns = ['area','major_axis','minor_axis'])
for i in range(len(sumeet_prop)):
    sumeet_table.iloc[i,0] = sumeet_prop[i].area
    sumeet_table.iloc[i,1] = sumeet_prop[i].major_axis_length
    sumeet_table.iloc[i,2] = sumeet_prop[i].minor_axis_length
    
    
sumeet_table1 = sumeet_table.copy()

sumeet_table1['major_axis'] = sumeet_table['major_axis'].astype('float64') 
sumeet_table1['minor_axis'] = sumeet_table['minor_axis'].astype('float64') 
sumeet_table1['area'] = sumeet_table['area'].astype('float64') 


sumeet_table1.describe()
    

# Histogram plots    
plt.hist(sumeet_table['major_axis'],bins=20)
plt.savefig('./Viz_sumeet/hist_max_len.jpg')

plt.hist(sumeet_table['minor_axis'],bins=20)
plt.savefig('./Viz_sumeet/hist_min_len.jpg')

plt.hist(sumeet_table['area'],bins=20)
plt.savefig('./Viz_sumeet/hist_area.jpg')


# Box plots
plt.boxplot(sumeet_table['major_axis'])
plt.savefig('./Viz_sumeet/box_max_len.jpg')

plt.boxplot(sumeet_table['minor_axis'])
plt.savefig('./Viz_sumeet/box_min_len.jpg')

plt.boxplot(sumeet_table['area'])
plt.savefig('./Viz_sumeet/box_area.jpg')




# Reading predicted binary mask for VAN0008
img_pred = cv2.imread('VAN0008_pred.jpg',cv2.IMREAD_GRAYSCALE)
img_pred[img_pred <=246 ] = 0
img_pred[img_pred > 246 ] = 1


# Get properties of predicted annotation
pred_annots = (img_pred*255).astype('uint8')
pred_label = sk.label(pred_annots)
pred_prop = sk.regionprops(label_image=pred_label)


pred_table = pd.DataFrame(index = range(len(pred_prop)), columns = ['area','major_axis','minor_axis'])
for i in range(len(pred_prop)):
    pred_table.iloc[i,0] = pred_prop[i].area
    pred_table.iloc[i,1] = pred_prop[i].major_axis_length
    pred_table.iloc[i,2] = pred_prop[i].minor_axis_length
    

# Histogram plots    
plt.hist(pred_table['major_axis'],bins=20)
plt.savefig('./Viz_preds/hist_max_len.jpg')

plt.hist(pred_table['minor_axis'],bins=20)
plt.savefig('./Viz_preds/hist_min_len.jpg')

plt.hist(pred_table['area'],bins=20)
plt.savefig('./Viz_preds/hist_area.jpg')


# Box plots
plt.boxplot(pred_table['major_axis'])
plt.savefig('./Viz_preds/box_max_len.jpg')

plt.boxplot(pred_table['minor_axis'])
plt.savefig('./Viz_preds/box_min_len.jpg')

plt.boxplot(pred_table['area'])
plt.savefig('./Viz_preds/box_area.jpg')



























    


















