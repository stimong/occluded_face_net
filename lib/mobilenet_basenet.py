""""
MobileNetV2 based Non-occluded Model 
"""

import tensorflow as tf


def mobilenet_occluded_model(input_shape=(112,112,3), num_class = 5, drop=0.2):
    """
    Non-occluded classification model based on MobileNetV2.
     Args:
         input_shape: the input shape of the model
         num_class: number of outputs
     Returns:
         model: Non-occluded classification model
    """
    act_f = "swish"
    kernel_reg = tf.keras.regularizers.l2(0.001)
    kernel_init = tf.keras.initializers.he_normal(seed=1)
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    
    inputs = tf.keras.Input(shape=input_shape)
    x_base = base_model(inputs, training=True)
    x_avg1 = tf.keras.layers.GlobalAveragePooling2D()(x_base)
    
    x1 = tf.keras.layers.Dropout(drop)(x_avg1)
    x1 = tf.keras.layers.Dense(1024, activation=act_f, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x1)
    x1 = tf.keras.layers.Dropout(drop)(x1)
    x1 = tf.keras.layers.Dense(512, activation=act_f, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(x1)
    
    outputs1 = tf.keras.layers.Dense(num_class, activation='softmax',name='output_1')(x1)
    
    model = tf.keras.Model(inputs,outputs1)
    
    return model
