""""
simple CNN based Non-occluded Model 
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout 


def simplecnn_occluded_model(input_shape=(112,112,3),  num_class=5, drop=0.2):
    """
    Non-occluded classification model based on MobileNetV2.
     Args:
         input_shape: the input shape of the model
         num_class: number of outputs
     Returns:
         model: Non-occluded classification model
    """
    
    act_func = "swish"
    kernel_reg = tf.keras.regularizers.l2(0.001)
    kernel_init = tf.keras.initializers.he_normal(seed=1) 
    
    # Stage 1 #
    img_input = Input(shape=input_shape)
    
    ## Block 1 ##
    x = Conv2D(32, (3,3), strides=(1,1), name='base_conv1', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(img_input)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv1_act')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='base_conv1_pool')(x)

    ## Block 2 ##
    x = Conv2D(64, (3,3), strides=(1,1), name='base_conv2', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv2_act')(x)
    x = Conv2D(64, (3,3), strides=(1,1), name='base_conv3', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv3_act')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='base_conv3_poo1')(x)

    ## Block 3 ##
    x = Conv2D(64, (3,3), strides=(1,1), name='base_conv4', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv4_act')(x)
    x = Conv2D(64, (3,3), strides=(1,1), name='base_conv5', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv5_act')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='base_conv5_poo1')(x)
        
    ## Block 4 ##
    x = Conv2D(256, (3,3), strides=(1,1), name='base_conv6', kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation(act_func, name='base_conv6_act')(x)
    x = Dropout(0.5)(x)
    
    ## Block 5 ##
    x = Flatten(name='base_flatten')(x)
    x = Dense(1024, activation=act_func, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, activation=act_func, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x)
    x = Dropout(0.2)(x)
   
    # Activation function을 적용하면 output의 범위를 [0,1]로 제한하게 되므로 
    #x = Dense(output_size, activation="sigmoid", name="out1")(x)
    x = Dense(num_class, activation="softmax", name="out1")(x)
    model = Model(img_input, x)
    
    return model