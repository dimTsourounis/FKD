import os
import torch
from skimage import io
import numpy
import matplotlib.pyplot as plt

import preprocessing_functions_text


#tr_val = 'train' # 'validation'
tr_val = 'validation' # 'validation'
path_data = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/", tr_val)

path_train_lines = "/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/lines_train.npy"
path_validation_lines = "/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/lines_validation.npy"

train_lines = numpy.load(path_train_lines)
validation_lines = numpy.load(path_validation_lines)

path_save_line_images = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/line_images", str(tr_val)+"_line_images")
img_type = '.tif'

files = os.listdir(path_data)

for idx in range(len(files)):

    # load text form
    img_name = os.path.join(path_data, files[idx])
    img = io.imread(img_name, plugin='matplotlib')

    # load indices of text form for indicating lines
    if (tr_val == 'train'):
        lines_img = train_lines[idx]
    elif (tr_val == 'validation'):
        lines_img = validation_lines[idx]

    # use only the horizontal dimension for rejecting lines with width shorter than 220 pixels
    output_img_size = (220, 150)

    # line images
    # line image : line_images_of_form[i]
    line_images_of_form = preprocessing_functions_text.Isolate_text_lines(img, lines_img, output_img_size)

    for i in range(len(line_images_of_form)):
        """
        plt.imshow(line_images_of_form[i], cmap ='gray')
        plt.show()
        """

        line_img_out_path = os.path.join(path_save_line_images, str('writer_')+str(files[idx][1:4])+str('_text_form_')+str(idx)+str('_line_img_')+str(i)+str(img_type))
        line_img = line_images_of_form[i]
        io.imsave(line_img_out_path, line_img, compress=6)
