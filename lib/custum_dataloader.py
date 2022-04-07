"""
tensorflow custom data loader.
"""

import os
from os import listdir
from os.path import isfile, join

import math
import random

from datetime import datetime

import cv2
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from tensorflow.keras.utils import Sequence

# set image augumentation
sometimes02 = lambda aug: iaa.Sometimes(0.2, aug)
sometimes05 = lambda aug: iaa.Sometimes(0.5, aug)
sometimes08 = lambda aug: iaa.Sometimes(0.8, aug)

seq_img_filter = iaa.Sequential([
        iaa.Fliplr(0.5),
        sometimes08(iaa.Affine(
            scale={"x": (0.9, 1.0), "y": (0.9, 1.05)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
        ),
                   ),
])

noise_fillter = iaa.Sequential([
                sometimes05(iaa.OneOf([
                    iaa.GaussianBlur((0, 0.3)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ])),
                sometimes05([
                    #iaa.SaltAndPepper(0.01, 0.05),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.GammaContrast((0.5, 2.0), per_channel=True)
                ]),

])


def get_filelist(file_dir):
    """
    Get a list of files in a path.
    Args:
        file_dir: dataset path
    Returns:
        file_list: list of files the dataset path
        class_list: list of classname the dataset path
    """
    included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']

    class_list = []
    file_list = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(file_dir)):
        if i==0:
            class_list = dirnames
        file_list += [os.path.join(dirpath, fn) for fn in filenames
                       if any(fn.lower().endswith(ext) for ext in included_extensions)]
        
    class_list.sort()
    return file_list, class_list

def get_word2int(class_list):
    """
    A dictionary that converts text classes to numbers.
    Args:
        class_list: [char1, char2, char3,...,charN], (example) ["a0","b1","c1","bc1"]
    Returns:
        char_to_int: {char: int, ...}
        int_to_char: {int: char, ...}
    """
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(class_list))
    int_to_char = dict((i, c) for i, c in enumerate(class_list))
    return char_to_int, int_to_char

def one_hot(a, num_classes):
    """
    One-hot encoding conversion of integers.
    Args:
        a: class number converted to integer
        num_classes: Total number of classes
    Returns:
        One-hot encoding results
    """
    return np.eye(num_classes)[a]

def create_dir(dir_name):
    """Creating directory
    Args:
        dir_name: directory name
    """
    if os.path.isdir(dir_name)==False:
        os.makedirs(dir_name)
        print("create directory: ", dir_name)
        
def get_class_data_num(file_list, class_list, class_char2int):
    """
    Set the number of dataset files by class
    Args:
         file_list: List of dataset file paths
         class_list: List of dataset classes
         class_char2int: a dictionary taking the string class as an int
     Returns:
         class_dict: A dictionary of the number of dataset files by class
    """
    
    class_dict = {}
    for class_name in class_list:
        class_dict[class_char2int[class_name]]=0
        
    for fn in file_list:
        class_name = fn.split("/")[-2]
        class_dict[class_char2int[class_name]]+=1
    return class_dict

def create_class_weight(labels_dict, mu=0.15):
    """
    tensorflow class weighting function
    """
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

class CustomDataloader(Sequence):
    def __init__(self, 
                 file_dir, 
                 batch_size, 
                 img_size,
                 out_size=12,
                 aug_op=False,
                 shuffle=True,
                 multi_op=False,
                 sample_n=1024,
                 ann_dir = "datasets/ann_4"
                 
                ):
        """
        Custum tensorflow dataloader
        Args:
            file_dir: Dataset directory path
            batch_size: batch size
            img_size: input image size
            out_size: size of the output image
            aug_op: use data augmentation boolean
            shuffle: data shuffle data augmentation after 1 epoch
            multi_op: ouput type -> True: [class label, eye nose mouth heat map], False: class label
            sample_n: number of samples to use as data
        
        """
        
        self.ann_dir = ann_dir
        
        self.file_list, self.class_list = get_filelist(file_dir)
        random.shuffle(self.file_list)
        
        self.class_char2int, self.class_int2char = get_word2int(self.class_list)
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.out_size = out_size
        
        self.aug_op = aug_op
        self.shuffle = shuffle
        self.multi_op = multi_op
        self.sample_n = sample_n
        
        self.file_list = self.file_list[:self.sample_n]
        
        self.file_n = len(self.file_list)
        self.class_n = len(self.class_list)
        self.class_data_num = get_class_data_num(self.file_list, self.class_list, self.class_char2int)
        self.on_epoch_end()
 
    def __len__(self):
        return math.ceil(self.file_n / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_x = []
        batch_y1 = []
        batch_y2 = []
        self.batch_fn = []
        for i in indices:
            try:
                img = cv2.imread(self.file_list[i], cv2.IMREAD_COLOR)
                
                if len(img)==0:
                    continue
                    
                fine_name = self.file_list[i].split("/")[-1]
                if self.multi_op:
                    heatmap_path = join(self.ann_dir, "ann_"+fine_name)
                    if isfile(heatmap_path):
                        heatmap_img = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

                    else:
                        heatmap_img = np.zeros([self.out_size,self.out_size],dtype=np.uint8)

                if self.aug_op:
                    #seq_img_filter = seq_img_filter.localize_random_state()
                    
                    # 회전, 크롭 등
                    seq_img_f = seq_img_filter.to_deterministic()
                    
                    img = seq_img_f.augment_image(img)
                    if self.multi_op:
                        heatmap_img = seq_img_f.augment_image(heatmap_img)
                    
                    # 블러, 노이즈 등
                    img = noise_fillter.augment_image(img)

                w,h = img.shape[:2]
                if w!=self.img_size or h!=self.img_size:
                    img = cv2.resize(img, (self.img_size,self.img_size))

                # input x
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255. #-0.5
                batch_x.append(img)
                self.batch_fn.append([self.file_list[i], i])
                
                # label class name
                class_name = self.file_list[i].split("/")[-2]
                class_idx = self.class_char2int[class_name]
                class_one_hot = one_hot(class_idx,self.class_n)
                batch_y1.append(class_one_hot)
                
                if self.multi_op:
                    # label heatmap 
                    heatmap_img = cv2.resize(heatmap_img,(self.out_size,self.out_size))
                    heatmap_img = heatmap_img.reshape(self.out_size*self.out_size)
                    heatmap_img = heatmap_img/255.
                    batch_y2.append(heatmap_img)
                
            except Exception as e:
                print("e",e)
            
        batch_x = np.array(batch_x)
        batch_y1 = np.array(batch_y1)
        batch_y2 = np.array(batch_y2)
        
        if self.multi_op:
            return batch_x, [batch_y1, batch_y2]
        else:
            return batch_x, batch_y1
    
    def on_epoch_end(self):
        self.indices = np.arange(self.file_n)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
        
        