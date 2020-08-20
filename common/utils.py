import datetime
import time

import numpy as np
from datasets import NumpyDataset
import os


def get_current_time_iso() -> str:
    """ Get current time in ISO 8601 """
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')


def get_current_datetime() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d %H:%M:%S.%f')[:-3]


def unnormalise(img) -> np.array:
    min, max = float(img.min()), float(img.max())
    image = img.clamp_(min=min, max=max)
    image = image.add_(-min).div_(max - min + 1e-5)
    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return image

# converting the pixel_array values to Hounsfield Units (HU), standard density unit for CT scans.
# outputs a dicom scan in np array
# We should verify that this does what it says it does
def get_pixels_hu(image):
    slope = image.RescaleSlope # getting slope
    intercept = image.RescaleIntercept # getting intercept
    image = image.pixel_array.astype(np.int16)

    # I am suspicious of this line, why the magic number?
    image[image == -2000] = 0

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def clean_and_threshold(image, threshold, under=True):
    air = -1000

    if under:
        image[image < threshold] = air
    else:
        image[image > threshold] = air
    return image

def pad_height(arr, height):
    # Calculate amount to pad
    curr = arr.shape[0]
    diff = height - curr

    # Add air (-1000) to top and buttom, centering existing values
    return np.pad(arr, ((diff // 2, diff - diff // 2), (0, 0), (0, 0)), constant_values=-1000)

def crop_neck(image3D): 
    #Crop to the top of the head and the top of the neck
    top_height = None
    neck_height = None
    neck_width = None
    height = image3D.shape[0]

    for i in range(height):
        slit = image3D[i,:,:]
        x_s, y_s = np.where(slit>-500) #Gaussian blur magic number...

        min_x = None
        max_x = None
        min_y = None
        max_y = None

        # Find rectangle boundaries
        for x in x_s:
            if not max_x or x > max_x:
                max_x = x
            if not min_x or x < min_x:
                min_x = x
        for y in y_s:
            if not max_y or y > max_y:
                max_y = y
            if not min_y or y < min_x:
                min_y = y

        # Calculate area of rectangle
        if max_x and max_y and min_x and min_y:
            count = abs(max_x-min_x)*abs(max_y-min_y)
        else:
            count = 0

        # Eliminate stray blurred pixels
        if count>20:
            if top_height == None:
                top_height = i
            if i > height/2:
                if neck_width == None or count < neck_width:
                    neck_height = i
                    neck_width = count

    return image3D[top_height-1:neck_height-1,:,:]
            
def scipy_gaussian(image3D):
    #Apply a SD 1 gaussian blur 
    image3D = crop_neck(scipy.ndimage.filters.gaussian_filter(image3D,1))
    return image3D