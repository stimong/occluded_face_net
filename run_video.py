from os.path import join

import pickle
import timeit

import numpy as np
import cv2

import tensorflow as tf

import argparse


input_shape = (112,112)
font = cv2.FONT_HERSHEY_SIMPLEX 

def img_2_inputx(img):
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

def predict_web_cam(model, class_int2char, video_path):
    """
    Predict Non-occluded_face from video with tensorflow h5 format model
    Args:
         model: the loaded h5 model
         class_int2char: Convert information dictionary to predict result string
         video_path: Path to the video file to load
    """
    
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    fh, fw = frame.shape[:2]
    x1,y1,x2,y2 = int(fw*0.2), int(fh*0.2), int(fw*0.8), int(fh*0.9)
    

    # 웹캠으로 찰영한 영상을 저장하기
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    v_name = video_path.split("/")[-1]
    out = cv2.VideoWriter('result/{}'.format(v_name), fourcc, 20, (fw, fh))
    
        
    while(cap.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        start_t = timeit.default_timer()
        ret, frame = cap.read()
        if ret:
            
            crop_frame = frame[y1:y2,x1:x2].copy()
            
            input_x = img_2_inputx(crop_frame)
            predict_y = model.predict(input_x)[0]
            #print("predict_y",predict_y.shape,np.argmax(predict_y), predict_y)
            conf_y = int(np.max(predict_y)*100)
            str_y = class_int2char[np.argmax(predict_y)]

            f_txt = "{} {}%".format(str_y, conf_y)
            cv2.putText(frame, f_txt,(20,int(fh*0.2)), font, 1, (0,255,0),2)
            
            # 알고리즘 종료 시점
            terminate_t = timeit.default_timer()

            FPS = int(1./(terminate_t - start_t ))
            cv2.putText(frame, "FPS: "+str(FPS),(20, int(fh*0.9)), font, 1, (0,255,0),2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            #frame = cv2.resize(frame,(320,240))

            cv2.imshow('video', frame)
            out.write(frame)

        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def predict_web_cam_tflite(tf_list, class_int2char, video_path):
    """
    Predicting video with tensorflow tflite format model
    Args:
         tf_list: the loaded tflite model, input inform, output inform
         class_int2char: Convert information dictionary to predict result string
         video_path: Path to the video file to load
    """
    interpreter, input1, output1 = tf_list
    
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    fh, fw = frame.shape[:2]
    x1,y1,x2,y2 = int(fw*0.2), int(fh*0.2), int(fw*0.8), int(fh*0.9)

    # 웹캠으로 찰영한 영상을 저장하기
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    v_name = video_path.split("/")[-1]
    out = cv2.VideoWriter('result/{}'.format(v_name), fourcc, 20, (fw, fh))
        
    while(cap.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        start_t = timeit.default_timer()
        ret, frame = cap.read()
        if ret:
            
            crop_frame = frame[y1:y2,x1:x2].copy()
            
            input_x = img_2_inputx(crop_frame)
            input_x = np.array(input_x, dtype=np.float32)
            interpreter.set_tensor(input1['index'], input_x)
            interpreter.invoke()
            
            predict_y = interpreter.get_tensor(output1['index'])[0]
            conf_y = int(np.max(predict_y)*100)
            str_y = class_int2char[np.argmax(predict_y)]

            f_txt = "{} {}%".format(str_y, conf_y)
            cv2.putText(frame, f_txt,(20,30), font, 1, (0,255,0),2)
            
            # 알고리즘 종료 시점
            terminate_t = timeit.default_timer()

            FPS = int(1./(terminate_t - start_t ))
            cv2.putText(frame, "FPS: "+str(FPS),(20, int(fh*0.9)), font, 1, (0,255,0),2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            #frame = cv2.resize(frame,(320,240))

            cv2.imshow('video', frame)
            out.write(frame)

        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', "-r",required=False, 
                                   default="./", 
                                    help="프로젝트 경로")
    parser.add_argument('--name', "-n",required=False, default='cnn_best', help="불러올 학습데이터 폴더명(날짜)")
    parser.add_argument('--video', "-v",required=False, default='sample/sample_video.mp4', help="불러올 동영상파일 경로")
    parser.add_argument('--tflite', "-tf",required=False, default=False, help="불러올 동영상파일 경로")

    args = parser.parse_args()
    
    root_dir = args.root  # 
    load_date= args.name # "0320_1935"
    video_name = args.video  # "sample/sample_video.mp4"
    tflite_op = args.tflite
    print("loaded tflite: ", tflite_op)
    
    
    load_traininfo_dir = "{}/train_info/{}".format(root_dir, load_date)
    video_path = "{}/{}".format(root_dir, video_name)
    
    load_class_int2char_path = join(load_traininfo_dir,"class_int2char.pkl")

    # class_int2char = {0: 'a0', 1: 'b1', 2: 'bc1', 3: 'bg', 4: 'c1'}
    with open(load_class_int2char_path, 'rb') as handle:
        class_int2char = pickle.load(handle)
    print("class_int2char: ", class_int2char)
    
    
    if tflite_op==False:
        pretrain_w_path = join(load_traininfo_dir,"best_model.h5")
        model = tf.keras.models.load_model(pretrain_w_path)
        print("model loaded!")
        predict_web_cam(model, class_int2char, video_path)
    else:
        tflite_w_path = join(load_traininfo_dir,"best_model.tflite")
        interpreter = tf.lite.Interpreter(model_path=tflite_w_path)
        interpreter.allocate_tensors()

        input1 = interpreter.get_input_details()[0]  # Model has single input.
        output1 = interpreter.get_output_details()[0]  # Model has single output.
        
        tf_list = [interpreter, input1,output1]
        predict_web_cam_tflite(tf_list, class_int2char, video_path)