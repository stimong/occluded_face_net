"""
Convert h5 to tflite
"""

from os.path import join

import pickle
import argparse

import cv2
import numpy as np

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


input_shape = (112,112)

def img_2_inputx(img, input_shape):
    """
    convert opencv image to input x
    Args:
        img: opencv image
    Returns:
        input_x: model input x
    """
    img = cv2.resize(img,input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #-0.5
    input_x = np.expand_dims(img,axis=0)/255.
    return input_x

def main(traininfo_dir):
    """
    convert tflite
    Args: 
        traininfo_dir: traininfo path    
    """
    h5_path = join(traininfo_dir, "best_model.h5")
    load_model = tf.keras.models.load_model(h5_path)

    load_model.trainable=False
    converter = tf.lite.TFLiteConverter.from_keras_model(load_model)
    tflite_model = converter.convert()
    
    # Save tflite.
    tflite_model_path = join(traininfo_dir, "best_model.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print("Saved!", tflite_model_path)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--traininfo_dir', "-t", default="./train_info", help="train info directory 경로")
    parser.add_argument('--dir_name', "-d",required=True, default="0320_2248", help="train info directory name")

    args = parser.parse_args()
    traininfo_dir = join(args.traininfo_dir, args.dir_name)
    main(traininfo_dir)