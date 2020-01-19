#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
from tensorflow.python.client import device_lib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(device_lib.list_local_devices())
'''
widerface_result_path = './widerface_result/firstlevel/'
picture_root_path = '/media/yons/7AD02E63D02E263B1/workplace/facedataset/widerface/widerface/WIDER_test/images/'
widerface_obj_path = '/media/yons/7AD02E63D02E263B1/workplace/facedataset/widerface/widerface/wider_face_split/wider_face_test_filelist.txt'
'''
widerface_result_path = './widerface_result/enhanced/'
picture_root_path = '/media/yons/7AD02E63D02E263B1/workplace/facedataset/widerface/widerface/WIDER_val/images/'
widerface_obj_path = '/media/yons/7AD02E63D02E263B1/workplace/facedataset/widerface/widerface/wider_face_split/wider_face_val_bbx_gt.txt'

#return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
return_elements = ["input/input_data:0", "pred_ssbbox/concat_2:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0", "pred_sbbox_ori/concat_2:0", "pred_mbbox_ori/concat_2:0", "pred_lbbox_ori/concat_2:0"]

pb_file         = "./yolov3_widerface_1080ti.pb"
num_classes     = 1
input_size      = 416
input_size_0    = 320
input_size_2    = 512
input_size_1    = 608
graph           = tf.Graph()
nms_thresold = 0.45
#result_obj = open(widerface_result_path+'1.txt', 'a+')
file_obj = open(widerface_obj_path,'r')
#all_lines = file_obj.readlines()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

is_show = False
is_write_txt = True

#for line in all_lines:
line = file_obj.readline()
with  tf.Session(graph=graph) as sess:
    while line:
        #print(line)
        single_pic_path = picture_root_path + line
        print("path:"+str(single_pic_path))
        groups = line.split('/')
        dir_name = groups[0]
        pic_name = groups[1].replace('.jpg','')
        #print("dir_name:"+dir_name)
        #print("pic_name:"+pic_name)
        num = int(file_obj.readline())
        #print('num'+str(num))
        for i in range(0,num):
            line = file_obj.readline()
            #print("line"+line)
            # image_path = picture_root_path + image_name
        original_image = cv2.imread(single_pic_path.replace('\n', ''))
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

        image_2_data = utils.image_preporcess(np.copy(original_image), [input_size_2, input_size_2])
        image_2_data = image_2_data[np.newaxis, ...]
        pred_ssbbox, pred_sbbox, pred_mbbox, pred_lbbox, pred_sbbox_ori, pred_mbbox_ori, pred_lbbox_ori = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3], return_tensors[4], return_tensors[5],
             return_tensors[6], return_tensors[7]],
            feed_dict={return_tensors[0]: image_data})
        pred_ssbbox_flip, pred_sbbox_flip, pred_mbbox_flip, pred_lbbox_flip, pred_sbbox_ori_flip, pred_mbbox_ori_flip, pred_lbbox_ori_flip = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3], return_tensors[4], return_tensors[5],
             return_tensors[6], return_tensors[7]],
            feed_dict={return_tensors[0]: flip_image_data})
        pred_ssbbox_0, pred_sbbox_0, pred_mbbox_0, pred_lbbox_0, pred_sbbox_ori_0, pred_mbbox_ori_0, pred_lbbox_ori_0 = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3], return_tensors[4], return_tensors[5],
             return_tensors[6], return_tensors[7]],
            feed_dict={return_tensors[0]: image_0_data})
        pred_bbox_0 = np.concatenate([np.reshape(pred_ssbbox_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_ori_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_ori_0, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_ori_0, (-1, 5 + num_classes))
                                      ], axis=0)
        pred_bbox_0[:, 0] = pred_bbox_0[:, 0] * (input_size / input_size_0)
        pred_bbox_0[:, 1] = pred_bbox_0[:, 1] * (input_size / input_size_0)
        pred_bbox_0[:, 2] = pred_bbox_0[:, 2] * (input_size / input_size_0)
        pred_bbox_0[:, 3] = pred_bbox_0[:, 3] * (input_size / input_size_0)

        pred_ssbbox_1, pred_sbbox_1, pred_mbbox_1, pred_lbbox_1, pred_sbbox_ori_1, pred_mbbox_ori_1, pred_lbbox_ori_1 = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3], return_tensors[4], return_tensors[5],
             return_tensors[6], return_tensors[7]],
            feed_dict={return_tensors[0]: image_1_data})
        pred_bbox_1 = np.concatenate([np.reshape(pred_ssbbox_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_ori_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_ori_1, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_ori_1, (-1, 5 + num_classes))
                                    ], axis=0)
        pred_bbox_1[:, 0] = pred_bbox_1[:, 0] * (input_size / input_size_1)
        pred_bbox_1[:, 1] = pred_bbox_1[:, 1] * (input_size / input_size_1)
        pred_bbox_1[:, 2] = pred_bbox_1[:, 2] * (input_size / input_size_1)
        pred_bbox_1[:, 3] = pred_bbox_1[:, 3] * (input_size / input_size_1)
        # print(str(pred_ssbbox.shape))

        pred_ssbbox_2, pred_sbbox_2, pred_mbbox_2, pred_lbbox_2, pred_sbbox_ori_2, pred_mbbox_ori_2, pred_lbbox_ori_2 = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3], return_tensors[4], return_tensors[5],
             return_tensors[6], return_tensors[7]],
            feed_dict={return_tensors[0]: image_2_data})
        pred_bbox_2 = np.concatenate([np.reshape(pred_ssbbox_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_sbbox_ori_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_mbbox_ori_2, (-1, 5 + num_classes)),
                                      np.reshape(pred_lbbox_ori_2, (-1, 5 + num_classes))
                                    ], axis=0)
        pred_bbox_2[:, 0] = pred_bbox_2[:, 0] * (input_size / input_size_2)
        pred_bbox_2[:, 1] = pred_bbox_2[:, 1] * (input_size / input_size_2)
        pred_bbox_2[:, 2] = pred_bbox_2[:, 2] * (input_size / input_size_2)
        pred_bbox_2[:, 3] = pred_bbox_2[:, 3] * (input_size / input_size_2)


        pred_bbox_flip = np.concatenate([np.reshape(pred_ssbbox_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_sbbox_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_mbbox_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_lbbox_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_sbbox_ori_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_mbbox_ori_flip, (-1, 5 + num_classes)),
                                         np.reshape(pred_lbbox_ori_flip, (-1, 5 + num_classes))
                                         ], axis=0)
        # flip bbox
        pred_bbox_flip[:, 0] = flip_image_data.shape[1] - pred_bbox_flip[:, 0]
        pred_bbox_ori = np.concatenate([np.reshape(pred_ssbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_sbbox_ori, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox_ori, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox_ori, (-1, 5 + num_classes))
                                        ], axis=0)

        pred_bbox = np.concatenate([pred_bbox_flip, pred_bbox_ori,pred_bbox_0,pred_bbox_1,pred_bbox_2])

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.33)
        bboxes = utils.nms(bboxes, nms_thresold, method='nms')
        if is_show:
            image = utils.draw_bbox(original_image, bboxes)
            image = Image.fromarray(image)
            image.show()
        if is_write_txt:
            dir_path = widerface_result_path + dir_name
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            if os.path.exists(dir_path):
                file_path = dir_path + '/' + pic_name.replace('\n','') + '.txt'
                result_file = open(file_path, 'a+')
                name_str = pic_name
                print(name_str)

                result_file.write(name_str)
                # result_file.write('\n')
                face_number = len(bboxes)
                face_number_str = str(face_number)
                print(face_number_str)
                result_file.write(face_number_str)
                result_file.write('\n')
                for box in bboxes:
                    left_x = box[0]
                    left_y = box[1]
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    conf = box[4]
                    write_str = str(left_x) + ' ' + str(left_y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(conf)
                    print(write_str)
                    result_file.write(write_str)
                    result_file.write('\n')
                result_file.close()
        #detect_image(dir_name, pic_name, single_pic_path.replace('\n', ''), is_show=False, nms_thresold=0.45, is_write_txt=True)
        line = file_obj.readline()

file_obj.close()





