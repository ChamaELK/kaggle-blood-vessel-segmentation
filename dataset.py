import torch
from torch.utils.data import Dataset
import os 
import glob
import numpy as np 
import skimage
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from scipy.ndimage import rotate
import random
import scipy

circle = skimage.morphology.disk(3)
def multi_dil(im, num, element=circle):
    for i in range(num):
        im = skimage.morphology.dilation(im, element)
    return im

def multi_ero(im, num, element=circle):
    for i in range(num):
        im = skimage.morphology.erosion(im, element)
    return im
def foreground_mask(image):
    sample_c  = multi_ero(multi_dil(image,5),5)
    thresh_mask = sample_c > skimage.filters.threshold_otsu(sample_c)
    global_mask = scipy.ndimage.binary_fill_holes(thresh_mask)
    return global_mask

def train_test_valid_split(*arrays, test_size: float, valid_size: float, **kwargs):
    first_split = train_test_split(*arrays, test_size=test_size, **kwargs)
    testing_data = first_split[1::2]
    if valid_size == 0:
        training_data = first_split[::2]
        validation_data = []
    else:
        training_validation_data = train_test_split(*first_split[::2], test_size=(valid_size / (1 - test_size)),
                                                    **kwargs)
        training_data = training_validation_data[::2]
        validation_data = training_validation_data[1::2]

    return training_data + testing_data + validation_data



def patches(image, tile_shape, padding):
    w, h = image.shape
    bytelength = image.nbytes // image.size
    padding_w , padding_h = padding
    padded_image = np.pad(image,((0,padding_w),(0,padding_h)))
    tile_height, tile_width = tile_shape
    h, w= padded_image.shape
    
    tiled_array =  np.lib.stride_tricks.as_strided( padded_image, shape = (h // tile_height,
                                                            w //  tile_width,
                                                            tile_height,
                                                            tile_width), 
                                                        strides = ( w* tile_height*bytelength,
                                                            tile_width*bytelength,
                                                            w*bytelength,
                                                            bytelength))
    return tiled_array


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0

    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img

def broadcast(patches):
    tw, th , pw,ph = patches.shape
    w, h = pw*tw, ph*th
    broadcasted_array = np.zeros((w,h)) 
    for i in range(tw):
        for j in range(th):
            broadcasted_array[pw*i:pw*(i+1),ph*j:ph*(j+1)] = patches[i,j,:,:]
    return broadcasted_array

def next_remainder(n, m):
    return ((n - 1) // m + 1) * m - n 

def normalize(image):
    return (image - image.min())/(image.max() -image.min())


class ImageDataset(Dataset):
    def __init__(self, image_files, label_files, dilate_label = True, remove_background = False, tile_shape = (192,192)): #image_folder, mask_folder):
        self.image_files = image_files
        self.label_files = label_files
        self.dilate_label = dilate_label
        self.remove_background = remove_background
        self.tile_shape = tile_shape
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        #print("index ")
        #print(index)
        image = skimage.io.imread(self.image_files[index])
        angle = random.randint(0,45)
        image_equalized = skimage.exposure.equalize_adapthist(image)
        if self.remove_background:
            image_equalized = image_equalized*foreground_mask(image_equalized)
        #image_equalized = rotate(image_equalized, angle, reshape=False)

        label = skimage.io.imread(self.label_files[index]).astype(np.uint8)
        
        descriptor = { "image_path": self.image_files[index],
                           "label_path":self.label_files[index],
                             "shape":image.shape }
        return descriptor, image_equalized*256, label

def gray_to_tiled_tensor(image, tile_shape):
    tw, th = tile_shape
    w,h = image.shape
    w_padding = next_remainder(w, tw)
    h_padding = next_remainder(h,th)
    image_patches = patches(image, (tw,th),(w_padding,h_padding)) 
    image_tensor = torch.from_numpy(image_patches)
    return image_tensor

def image_feature_tensor(image, tile_shape):
    image_tensor = gray_to_tiled_tensor(image, tile_shape)
    w,h, a,b = image_tensor.shape
    n_image_channels = 3
    features_tensor = torch.zeros(w,h,n_image_channels,a,b)
    for i in range(w):
        for j in range(h):
            features_tensor[i,j][0] = image_tensor[i,j]
            features_tensor[i,j][1] = image_tensor[i,j]
            features_tensor[i,j][2] = image_tensor[i,j]
    return features_tensor

def label_feature_tensor(label, tile_shape, dilate_label= False):
    binary_label = np.where(label > 0, 1, 0)
    label_float = binary_label.copy().astype(np.float32)
    borders_float = binary_label.copy().astype(np.float32)
    if dilate_label :
        clear = label.copy()
        erode = multi_ero(clear,2)
        dilate = multi_dil(clear,4)
        borders = np.logical_and(dilate , ~erode)
        borders_float = normalize(borders.astype(np.float32))
        label_float = normalize(dilate.astype(np.float32))
        
    label_tensor = gray_to_tiled_tensor(label_float, tile_shape)
    borders_tensor  = gray_to_tiled_tensor(borders_float, tile_shape)
    n_label_channels = 2
    w,h, a,b = label_tensor.shape
    labels_tensor = torch.zeros(w,h,n_label_channels,a,b)
    for i in range(w):
        for j in range(h):
            labels_tensor[i,j,0] = label_tensor[i,j]     
            labels_tensor[i,j,1] = borders_tensor[i,j]
    return labels_tensor


