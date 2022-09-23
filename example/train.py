import os   
import random
import skimage
import json
import sys
import datetime
import tensorflow as tf
import numpy as np
import imgaug

sys.path.append("/path/to/mask/rcnn/repo")
import mrcnn.model as modellib

from mask_rcnn_conf import SpaceNet_config
from spacenet_dataset import Spacenet_dataset

#DEFINE NEEDED DIRECTORY
MODEL_DIR = "/path/to/your/pre-trained/model"
LOG_DIR = "/path/to/your/model/log"
DATASET_DIR = "/path/to/your/dataset"

#init training with COCO, last trained, or pre-trained model
init_model = "last"

#LOAD TRAINING CONFIG
training_config = SpaceNet_config()


#LOAD TRAINING DATASET
training_dataset = Spacenet_dataset()
training_dataset.load_dataset(DATASET_DIR, "train")
training_dataset.prepare()
#LOAD VALIDATION DATASET
validation_dataset = Spacenet_dataset()
validation_dataset.load_dataset(DATASET_DIR, "validation")
validation_dataset.prepare()


#SETUP TRAINING CALLBACK
##SETUP TRAINING LOG
logdir = os.path.join(
    LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, write_graph=True, write_images=False)
##SETUP EARLY STOP
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)


#CREATE IMAGE AUGMENTATION
img_augmentation = imgaug.augmenters.Sequential([
                                                 imgaug.augmenters.Fliplr(0.5),
                                                 imgaug.augmenters.Flipud(0.5) 
                                                 ], random_order=True)


#CREATE MODEL INSTANCE
model = modellib.MaskRCNN(
    mode="training",
    config=training_config,
    model_dir=MODEL_DIR
)

#LOAD WEIGHTS
if init_model == "coco":
    coco_weight_path = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
    model.load_weights(
        coco_weight_path,
        by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    )
elif init_model == "last":
    model.load_weights(model.find_last(), by_name=True)
elif init_model == "pre-trained":
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_full_0150.h5")
    model.load_weights(model_path, by_name=True)
elif init_model == "pre-trained-off-nadir":
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_off_nadir_pre.h5")
    model.load_weights(model_path, by_name=True)
elif init_model == "feature-extract":
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_coco_head_0030.h5")
    model.load_weights(model_path, by_name=True)
elif init_model == "fine-tune-1":
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_2_finetune_1.h5")
    model.load_weights(model_path, by_name=True)
elif init_model == "fine-tune-2":
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_2_finetune_2.h5")
    model.load_weights(model_path, by_name=True)


#TRAIN MODEL (HEAD LAYER ONLY)
model.train(
    training_dataset,
    validation_dataset,
    epochs=40,
    layers='heads',
    custom_callbacks = [tensorboard_callback, early_stop_callback],
    learning_rate=training_config.LEARNING_RATE,
    augmentation = img_augmentation
)

#TRAIN MODEL (LAYER 4+)
model.train(
    training_dataset,
    validation_dataset,
    epochs=120,
    layers='4+',
    custom_callbacks = [tensorboard_callback, early_stop_callback],
    learning_rate=training_config.LEARNING_RATE/10,
    augmentation = img_augmentation
)

#TRAIN MODEL (ALL LAYER)
model.train(
    training_dataset,
    validation_dataset,
    epochs=200,
    layers='all',
    custom_callbacks = [tensorboard_callback, early_stop_callback],
    learning_rate=training_config.LEARNING_RATE/100,
    augmentation = img_augmentation
)