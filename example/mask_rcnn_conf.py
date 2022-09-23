import sys

sys.path.append("/path/to/mask/rcnn/repo")
from mrcnn.config import Config

class SpaceNet_config(Config):
    BACKBONE = 'resnet101' #backbone used in training
    GPU_COUNT = 1 #number of gpu used
    IMAGES_PER_GPU = 2 #number of images processed per gpu
    NUM_CLASSES = 1 + 1 #number of class (building and background/non-building)
    # IMAGE_MIN_DIM = 640 #minimum image dimension
    # IMAGE_MAX_DIM = 640 #maximum image dimension
    # MEAN_PIXEL = np.array([83.07730001, 79.89838819, 79.71916675])
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) #RPN anchor side pixel
    RPN_ANCHOR_RATIOS = [0.25, 1, 4]
    RPN_NMS_THRESHOLD = 0.9
    USE_MINI_MASK = True #enable/disable mini mask
    LEARNING_RATE = 0.0001
    NAME='SpaceNet'