# Non-Occluded Face Networks

This is a Non-Occluded Face classification model. The mobile version can be converted.

## Installation
1. Install Python3.8 and Tensorflow >= 2.8.0
2. Clone this repository.
```
git clone https://github.com/stimong/Non_occluded_face_net.git
cd Non_occluded_face_net
```
3. Install the dependencies in requirements.txt. Please install additional packages manually.
```
pip install -r requirements.txt
```

## Demo
predict video sample
```
python run_video.py -tf True -n mobilenet_best -v sample/sample_video.mp4
```

predict wabcam sample
```
python run_wabcam.py -tf True -n mobilenet_best -v sample/sample_video.mp4
```

## dataset

1. Put face_datasets_5 under folder `data`. The folder structure should look like this:
````
Non-Occluded Face Networks
-- data
   |-- face_datasets_5
       |-- train
           |-- a0   
           |-- b1  
           |-- c1   
           |-- bc1 
           |-- bg             
       |-- val
       |-- test
-- lib
-- model_structure
-- plot_images
-- pre_w
-- result
-- sample
-- train_info
-- convert_tflite_example.ipynb
-- data_eda.ipynb
-- dataset_split_sample_with_bg.ipynb
-- train_example.ipynb
-- requirements.txt
-- run_video.py
-- run_webcam.py
````

## Training
````
-- lib
   |-- cnn_basenet.py
   |-- convert_tflite.py
   |-- custum_dataloader.py
   |-- mobilenet_basenet.py
   |-- train.py
````
```
python ./lib/train.py -e 20 -m "mobilenet"

```