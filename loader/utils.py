import os
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import tensorflow as tf


class Bbox:
    def __init__(self, x, y, w, h, cat):
        self.x = x
        self.x = x
        self.w = w
        self.h = h
        self.cat = cat

    def __getitem__(self,key):
        return getattr(self, key)

def train_test_split(data_path, *args):
    if len(args) == 0:
        percentage_test = 10
    else:
        percentage_test = args[0]
    img_count = len(os.listdir(data_path+"images/apple"))
    file_train = open(data_path + 'train.txt', 'w+')  
    file_test = open(data_path + 'test.txt', 'w+')
    counter = 0
    index_test = round(img_count * percentage_test / 100)
    all_items = glob.glob(data_path+"images/apple/*.jpg")
    for i in all_items:
        if counter < index_test:
            file_test.write(i+"\n")
            counter += 1
        else:
            file_train.write(i+"\n")
def box_plot(box):
    # a = (box[0][:].split(","))
    # print (a)

    fig, ax = plt.subplots(1)
    im = np.zeros((100,100),dtype='float64')
    ax.imshow(im)
    for i in range(3):
        k = 0
        s = patches.Rectangle((box[i][k], box[i][k+1]), box[i][k+2], box[i][k+3], linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(s)
    plt.show()

def to_NCHW_format(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        # change the channel to gpu tensorflow format
        to_NCHW = tf.transpose(bottom, [0, 3, 1, 2])

        reshaped = tf.reshape(to_NCHW,
                                    tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf
def bbox_transform_inv(boxes, regr):
    if boxes.shape[0] == 0:
        return np.zeros((0, regr.shape[1]), dtype=regr.dtype)

    boxes = boxes.astype('float32', copy=False)

    width = boxes[:, 2] - boxes[:, 0] + 1.0
    height = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * width
    ctr_y = boxes[:, 1] + 0.5 * height

    dx = regr[:, 0::4]
    dy = regr[:, 1::4]
    dw = regr[:, 2::4]
    dh = regr[:, 3::4]

    pred_ctr_x = dx * width[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * height[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = tf.exp(dw) * width[:, np.newaxis]
    pred_h = tf.exp(dh) * height[:, np.newaxis]

    pred_boxes = np.zeros(regr.shape, dtype=object)

    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w

    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
def clip_boxes(boxes, im_shape):
    # im_shape = tf.convert_to_tensor(im_shape, dtype=object)
    boxes[:, 0::4] = tf.maximum(tf.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = tf.maximum(tf.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = tf.maximum(tf.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = tf.maximum(tf.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)


    return boxes