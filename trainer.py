from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import loader.utils as utils
import cv2
import sys
from models import final_layers
from loader.DataLoader import load
import cv2
from models.rpn import RPN




dataset_path = sys.argv[1]
data_loader = load(dataset_path)

rpn_net = RPN()
rpn_cls, rpn_bbox, net = rpn_net.vgg_16()
x, gt_boxes = rpn_net.getPlaceholders()

data = data_loader.data_batch()
img, gt_box, labels = data[0][0], data[0][1], data[0][2]
img_info = float(img.shape[1]), float(img.shape[2])
loss = rpn_net.setup(net, rpn_cls, rpn_bbox, img_info, data)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(2):
        sess.run(train_step, feed_dict={x:img, gt_boxes:gt_box})
        ls_val = sess.run(loss, feed_dict={x:img, gt_boxes:gt_box})
        print ('loss : {}'.format(ls_val))

