#! /usr/bin/env python
# coding=utf-8
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_ssbbox/concat_2:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0", "pred_sbbox_ori/concat_2:0", "pred_mbbox_ori/concat_2:0", "pred_lbbox_ori/concat_2:0"]
pb_file         = "./yolov3_widerface_1080ti.pb"
image_path      = "./docs/small/4.jpg"
num_classes     = 1
input_size      = 416
input_size_0    = 320
input_size_1    = 608
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

flip_image = cv2.flip(original_image, 1)
flip_image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
flip_image_size = flip_image.shape[:2]
flip_image_data = utils.image_preporcess(np.copy(flip_image), [input_size, input_size])
flip_image_data = flip_image_data[np.newaxis, ...]


image_0_data = utils.image_preporcess(np.copy(original_image), [input_size_0, input_size_0])
image_0_data = image_0_data[np.newaxis, ...]


image_1_data = utils.image_preporcess(np.copy(original_image), [input_size_1, input_size_1])
image_1_data = image_1_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

ptime = time.time()
with tf.Session(graph=graph) as sess:
    ptime = time.time()
    pred_ssbbox, pred_sbbox, pred_mbbox, pred_lbbox, pred_sbbox_ori, pred_mbbox_ori, pred_lbbox_ori = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3],return_tensors[4], return_tensors[5], return_tensors[6],return_tensors[7]],
                feed_dict={ return_tensors[0]: image_data})
    pred_ssbbox_flip, pred_sbbox_flip, pred_mbbox_flip, pred_lbbox_flip, pred_sbbox_ori_flip, pred_mbbox_ori_flip, pred_lbbox_ori_flip = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3],return_tensors[4], return_tensors[5], return_tensors[6],return_tensors[7]],
                feed_dict={ return_tensors[0]: flip_image_data})
    pred_ssbbox_0, pred_sbbox_0, pred_mbbox_0, pred_lbbox_0, pred_sbbox_ori_0, pred_mbbox_ori_0, pred_lbbox_ori_0 = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3],return_tensors[4], return_tensors[5], return_tensors[6],return_tensors[7]],
                    feed_dict={ return_tensors[0]: image_0_data})
    pred_bbox_0 = np.concatenate([np.reshape(pred_ssbbox_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_sbbox_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_mbbox_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_lbbox_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_sbbox_ori_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_mbbox_ori_0, (-1, 5 + num_classes)),
                                  np.reshape(pred_lbbox_ori_0, (-1, 5 + num_classes))], axis=0)
    pred_bbox_0[:, 0] = pred_bbox_0[:, 0] * (input_size / input_size_0)
    pred_bbox_0[:, 1] = pred_bbox_0[:, 1] * (input_size / input_size_0)
    pred_bbox_0[:, 2] = pred_bbox_0[:, 2] * (input_size / input_size_0)
    pred_bbox_0[:, 3] = pred_bbox_0[:, 3] * (input_size / input_size_0)

    pred_ssbbox_1, pred_sbbox_1, pred_mbbox_1, pred_lbbox_1, pred_sbbox_ori_1, pred_mbbox_ori_1, pred_lbbox_ori_1 = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3],return_tensors[4], return_tensors[5], return_tensors[6],return_tensors[7]],
                    feed_dict={ return_tensors[0]: image_1_data})
    pred_bbox_1 = np.concatenate([np.reshape(pred_ssbbox_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_sbbox_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_mbbox_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_lbbox_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_sbbox_ori_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_mbbox_ori_1, (-1, 5 + num_classes)),
                                     np.reshape(pred_lbbox_ori_1, (-1, 5 + num_classes))], axis=0)
    pred_bbox_1[:, 0] = pred_bbox_1[:, 0] * (input_size / input_size_1)
    pred_bbox_1[:, 1] = pred_bbox_1[:, 1] * (input_size / input_size_1)
    pred_bbox_1[:, 2] = pred_bbox_1[:, 2] * (input_size / input_size_1)
    pred_bbox_1[:, 3] = pred_bbox_1[:, 3] * (input_size / input_size_1)
    #print(str(pred_ssbbox.shape))

    pred_bbox_flip = np.concatenate([np.reshape(pred_ssbbox_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_sbbox_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_sbbox_ori_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox_ori_flip, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox_ori_flip, (-1, 5 + num_classes))], axis=0)
    #flip bbox
    pred_bbox_flip[:, 0] = flip_image_data.shape[1] - pred_bbox_flip[:, 0]
    pred_bbox_ori = np.concatenate([np.reshape(pred_ssbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_sbbox_ori, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox_ori, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox_ori, (-1, 5 + num_classes))], axis=0)

pred_bbox = np.concatenate([pred_bbox_flip,pred_bbox_ori])

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
bboxes = utils.nms(bboxes, 0.2, method='nms')
image = utils.draw_bbox(original_image, bboxes,show_label=False)
ctime = time.time()
exec_time = ctime - ptime
info = "time: %.2f ms" %(1000*exec_time)
#cv2.imwrite('./docs/pic_res/fddb/img_631.jpg',image.astype(np.uint8))
print(info)

image = Image.fromarray(image)

image.save('./docs/pic_res/small_r/4.jpg')
image.show()



