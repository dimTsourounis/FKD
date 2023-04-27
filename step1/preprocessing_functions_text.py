import warnings
import numpy
import skimage
from skimage.color import rgb2gray


def Isolate_text_lines(imgray, Threshold_var, output_size):
    """Removes spaces and cuts an image with text into greyscale segments.

    Parameters
    ----------
    imgray : The image to process, in singe-channel greyscale.

    Threshold_var: Array containing line markings per line of pixels. False indicates a blank line, True otherwise.

    output_size: Size of returned images, (width, height) tuple.

    Returns
    -------
    out : Numpy image.
    """

    rows = numpy.size(imgray, 0)
    cols = numpy.size(imgray, 1)

    #loop variables
    line_index=0
    current_line=False
    lines=[]
    samples = []

    #iterate through the lines in Threshold_var
    for index in range(rows):
        if (Threshold_var[index] != current_line):
            current_line = Threshold_var[index]
            #if last line was False and current line is True, save current line's index
            if (current_line):
                start_line = index
            else:
                #if last line was True and current line is False, store all the lines
                #between start_line and current line as an image
                testline = imgray[start_line:index, 0:cols]

                #error checking: if the stored image is too small it was
                #probably an error, do not save it
                #otherwise, save the image under lines[line_index]
                if (numpy.size(testline,0) > 20):
                    lines.append(testline)
                    line_index=line_index+1



    #___space killer___
    #takes each image in lines and removes spaces between words
    #this should not be an issue as we are only interested in handwritting,
    #not reading the actual text

    for i in range(line_index):
        # calculate the standard deviation value per column of image
        var_line = numpy.std(lines[i],0)
        # set threshold to 10% of max
        thresh_spaces = 0.1*numpy.max(var_line)
        # select a threshold in order to isolate blank space
        thresh_line = numpy.greater(var_line, thresh_spaces)

        # keep only columns with more than blank space
        lines[i] = lines[i][:,thresh_line]



    #___sample___
    #desired output size
    S_width = output_size[0]
    S_height = output_size[1]

    # #discard lines shorter than S_width
    # for i, line in enumerate(lines):
    #     if (numpy.size(line,1) < S_width):
    #         del lines[i]
    #         #print('deleted line with length %i' %(numpy.size(line,1)))
    lines[:] = [line for line in lines if (numpy.size(line,1) > S_width)]


    # pick all lines one-by-one
    # select each line and return it
    for line in range(len(lines)):
        # each sample comes from one line

        line_height = numpy.size(lines[line],0)

        # the height of the line can be smaller or larger than the one expected

        startpoint_height = 0
        endpoint_height = line_height
        # first_crop: the line as it was
        first_crop = lines[line][ startpoint_height:endpoint_height ,:]

        #second crop: after samples have been picked remove spaces horizontaly from them again
        #this helps remove more space from images that could not be cleanly cut into lines

        # calculate the standard deviation value per column of image
        variance_crop2 = numpy.std(first_crop,0)
        # set threshold to 1% of max
        thresh_crop2 = 0.01*numpy.max(variance_crop2)
        # select a threshold in order to isolate blank space
        thresh_line_crop2 = numpy.greater(variance_crop2, thresh_crop2)


        # second_crop: keep only columns of line with more than blank space
        second_crop = first_crop[ : ,thresh_line_crop2]

        ##displays the random samples
        # import matplotlib.pyplot as plt
        # plt.subplot(1,1,2)
        # plt.imshow(first_crop, cmap='gray')
        # plt.subplot(2,1,3)
        # plt.imshow(second_crop, cmap='gray')
        samples.append(second_crop)


    return samples


def Invert(img):
    inverted = 255 - img
    #inverted = skimage.util.invert(img)
    inverted = skimage.img_as_float(inverted) # image data type
    inverted = numpy.float32(inverted)
    return inverted
