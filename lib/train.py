"""
Non-occluded face classification training example code
"""

import pickle
import argparse

from datetime import datetime

from custum_dataloader import *
from mobilenet_basenet import mobilenet_occluded_model as build_model_mb
from cnn_basenet import simplecnn_occluded_model as build_model_cnn

import tensorflow as tf
config = tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def get_dataset(dataset_dir, img_size, batch_size, aug_op=True, out_size=12):
    
    multi_output = False
    
    train_file_dir = join(dataset_dir,"train")
    val_file_dir = join(dataset_dir,"val")
    test_file_dir = join(dataset_dir,"test")

    train_loader = CustomDataloader(train_file_dir, batch_size, img_size, aug_op=aug_op, sample_n=-1, multi_op=multi_output)
    val_loader = CustomDataloader(val_file_dir, batch_size, img_size, aug_op=False, sample_n=-1, multi_op=multi_output)
    test_loader = CustomDataloader(test_file_dir, batch_size, img_size, aug_op=False, sample_n=-1, multi_op=multi_output)

    return train_loader, val_loader, test_loader

def run_train(project_dir, 
              dataset_dir, 
              img_size=112, 
              batch_size=16, 
              epochs=30, 
              aug_op=True, 
              pre_w_path="",
              base_model="moblienet"
              ):
    
    print("-----------\n")
    train_loader, val_loader, test_loader = get_dataset(dataset_dir, img_size, batch_size, aug_op)
    
    class_int2char = train_loader.class_int2char
    print("class_int2char: ",class_int2char)

    train_data_num = train_loader.file_n
    print("train_data_num: ",train_data_num)

    class_data_num = train_loader.class_data_num
    print("class_data_num: ", class_data_num)
    
    NUM_CLASS = train_loader.class_n
    IMG_SHAPE = (img_size, img_size, 3)

    print("NUM_CLASS: ", NUM_CLASS)
    print("IMG_SHAPE: ", IMG_SHAPE)
    print("-----------\n")

    # build model
    if base_model =="moblienet":
        model = build_model_mb(IMG_SHAPE, NUM_CLASS)
    else:
        model = build_model_cnn(IMG_SHAPE, NUM_CLASS)
    print("\nbuild base model: ", base_model)
    
    # model compile
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, 
                  loss=["categorical_crossentropy"], 
                  metrics=["accuracy"])
    
    if len(pre_w_path)>0:  
        model = tf.keras.models.load_model(pre_w_path)
        print("loaded: ", pre_w_path)
        
    # set checkpoint
    now = datetime.today().strftime('%m%d_%H%M')
    ckp_filepath = '{}/pre_w/mobilenet_5class_best_{}.h5'.format(project_dir, now)
    print("ckp_filepath: ",ckp_filepath)

    my_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_filepath, 
                                 monitor="val_loss",
                                 verbose=2, 
                                 save_best_only=True,
                                 mode="min")
    

    history = model.fit(train_loader, 
                       validation_data=val_loader, 
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=0,
                       workers=4,# multi로 처리할 개수
                       callbacks=[my_checkpoint],
                      )
    
    print("traing done!")
    ## load best model
    load_ckp_filepath = ckp_filepath
    load_model=tf.keras.models.load_model(load_ckp_filepath)
    evaluate_result = load_model.evaluate(train_loader)
    
    
    ## save model sturucture and trained weight
    now = datetime.today().strftime('%m%d_%H%M')

    save_traininfo_dir = "{}/train_info".format(project_dir)
    save_traininfo_dir = join(save_traininfo_dir, now)
    create_dir(save_traininfo_dir)

    save_model_path = join(save_traininfo_dir, "best_model.h5")  # ckp_filepath.replace("pre_w",save_traininfo_dir)
    print("save_best_model_path: ", save_model_path)

    load_model.save(save_model_path)

    ## save label dictionary
    save_class_int2char_path = join(save_traininfo_dir,"class_int2char.pkl")
    print("save_class_int2char_path: ",save_class_int2char_path)

    with open(save_class_int2char_path, 'wb') as handle:
        pickle.dump(class_int2char, handle)

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    # print(a == b)
    
    print("Saved!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project_dir', "-p", default=".", help="프로젝트 경로")
    parser.add_argument('--dataset_dir', "-d", default='./data/face_datasets_5', help="불러올 학습데이터 경로")
    parser.add_argument('--img_size', "-s", default=112, type=int , help="input image size")
    parser.add_argument('--batch_size', "-b", default=32, type=int, help="batch size")
    parser.add_argument('--epochs', "-e", default=30, type=int, help="epoch 수")
    parser.add_argument('--aug_op', "-a", default=True, help="데이터 증강 사용 여부")
    parser.add_argument('--pre_w', "-w", default="", help="pretrained model weight")
    parser.add_argument('--base_model', "-m", default="moblienet", help="base model: moblienet, cnn")

    args = parser.parse_args()
    
    project_dir = args.project_dir  # 
    dataset_dir = args.dataset_dir  # 
    img_size = args.img_size  # 112
    batch_size = args.batch_size  # 32
    epochs = args.epochs  # 10
    aug_op = args.aug_op  # True
    pre_w_path = args.pre_w  # True
    base_model = args.base_model  # "moblienet"
    
    run_train(project_dir=project_dir,
              dataset_dir=dataset_dir, 
              img_size=img_size, 
              batch_size=batch_size, 
              epochs=epochs, 
              aug_op=aug_op,
              pre_w_path=pre_w_path,
              base_model=base_model
              )
    
    