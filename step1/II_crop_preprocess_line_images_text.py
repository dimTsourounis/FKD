import os
import torch
from skimage import io
import numpy
import matplotlib.pyplot as plt

import preprocessing_functions_text

db = 'MCYT75'
# db info
if (db == 'GPDS960BW'): 
    # aspect_ratio: size(image, 1) / size(image, 0)
    aspect_ratio = 2.2258
elif (db == 'MCYT75'):
    # aspect_ratio: size(image, 1) / size(image, 0)
    aspect_ratio = 2.2818
elif (db == 'CEDAR'):
    # aspect_ratio: size(image, 1) / size(image, 0)
    aspect_ratio = 2.7210
elif (db == 'UTSIG'):
    # aspect_ratio: size(image, 1) / size(image, 0)
    aspect_ratio = 1.6757


#tr_val = 'train' # 'validation'
tr_val = 'validation' # 'validation'
path_data = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/line_images", str(tr_val)+"_line_images")

path_save_cropped_images = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/line_images",str(tr_val)+"_crop_images")
img_type = '.tif'

files = os.listdir(path_data)
cnt = 0

for idx in range(len(files)):

    # load line image
    img_name = os.path.join(path_data, files[idx])
    img = io.imread(img_name, plugin='matplotlib')

    rows = numpy.size(img, 0)
    cols = numpy.size(img, 1)

    Num_crops = numpy.floor(cols / (rows*aspect_ratio))

    for j in range(int(Num_crops)):
        cnt = cnt + 1 # counts images
        width = round(rows*aspect_ratio)
        startpoint = j * width
        endpoint = startpoint + width
        image = img[:,startpoint:endpoint]

        image_out_path = os.path.join(path_save_cropped_images, str('writer_')+str(files[idx][7:10])+str('_text_image_')+str(cnt)+str(img_type))
        io.imsave(image_out_path, image, compress=6)
