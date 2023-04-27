import os
import torch
from skimage import io
from skimage import img_as_ubyte
from normalize import preprocess_signature

db = 'MCYT75'
# db info
if (db == 'GPDS960BW'): # Scratch & GPDS300BW
    canvas_size = (952, 1360) # max size
elif (db == 'MCYT75'):
    canvas_size = (600, 850) # max size
elif (db == 'CEDAR'):
    canvas_size = (730, 1042) # max size
elif (db == 'UTSIG'):
    canvas_size = (1440, 1825) # max size


tr_val = 'train' # 'validation'
#tr_val = 'validation' # 'validation'
path_data = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/line_images", str(tr_val)+"_crop_images")

path_save_processed_images = os.path.join("/home/ellab_dl/Desktop/OSV_SigNet_text_project/1_Train_SigNet_text/database/line_images",str(db), str(tr_val)+"_processed_dataset")
img_type = '.tif'

files = os.listdir(path_data)
cnt = 0

for idx in range(len(files)):

    cnt = cnt + 1

    # load cropped image
    img_name = os.path.join(path_data, files[idx])
    img = img_as_ubyte(io.imread(img_name, as_gray=True))

    # preprocess (img-> uint8 grayscale)
    image = preprocess_signature(img,canvas_size)

    # save processed image
    image_out_path = os.path.join(path_save_processed_images, str('writer_')+str(files[idx][7:10])+str('_text_image_')+str(cnt)+str(img_type))
    io.imsave(image_out_path, image, compress=6)
