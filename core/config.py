#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/widerface.names"
__C.YOLO.ANCHORS                = "./data/anchors/widerface_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [4, 8, 16, 32, 8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
#__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
#__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_test_loss=33.8890.ckpt-19"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_test_loss=33.8890_demo.ckpt-19"



# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "./data/dataset/wider_train_yolo3.txt"
__C.TRAIN.BATCH_SIZE            = 4
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 100
__C.TRAIN.SECOND_STAGE_EPOCHS   = 250
__C.TRAIN.SMOOTH_DETA           = 0.01
__C.TRAIN.ONE_DETA           = 0.2
__C.TRAIN.TINY_DETA           = 0.25
#"./checkpoint/yolov3_test_loss=21.1034.ckpt-26"
__C.TRAIN.INITIAL_WEIGHT        = "/media/yons/7AD02E63D02E263B1/res/MSNFD/yolov3_2_test_loss=18.6413.ckpt-140"
__C.TRAIN.RESTORE_FORMAT = "origin" #from yolov3 backbone="origin" else from now ="now"
__C.TRAIN.RESTORE_PART = ['darknet']
__C.TRAIN.UPDATE_PART = ['yolo_body', 'enhance_body', 'tiny_face']

# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "./data/dataset/wider_val_yolo3.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=9.2099.ckpt-5"
__C.TEST.SHOW_LABEL             = False
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45






