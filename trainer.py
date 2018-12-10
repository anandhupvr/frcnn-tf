from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import loader.utils as utils
import cv2
import sys
from models import final_layers
from loader.DataLoader import load
import cv2
# from models.rpn import RPN
from models.net import network



num_epo = 500
dataset_path = sys.argv[1]
data_loader = load(dataset_path)
net = network()
rois, cls_prob, bbox_pred, cls_score = net.build_network()
x, gt_boxes = net.getPlaceholders()

x, gt_boxes = net.getPlaceholders()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# rpn_net = RPN()
# rpn_cls, rpn_bbox, net = rpn_net.vgg_16()
# x, gt_boxes = rpn_net.getPlaceholders()

saver = tf.train.Saver()

# init_op = tf.global_variables_initializer()
with tf.Session(config = config) as sess:
    for i in range(num_epo):
        for _ in range(len(open("train.txt", "r").readlines())):
            data = data_loader.data_batch()
            img, gt_box, labels = data[0][0], data[0][1], data[0][2]
            loss = net.losses(bbox_pred, cls_score)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            sess.run(tf.global_variables_initializer()) 
            sess.run(train_step, feed_dict={x:img, gt_boxes:gt_box})
            ls_val = sess.run(loss, feed_dict={x:img, gt_boxes:gt_box})
            print ('loss : {}       --> : {}'.format(ls_val, _))
        print ('loss : {}      epoch --> : {}'.format(ls_val, i))
    if i%100 == 0:
        save_path = saver.save(sess, 'weights/'+"model_{}.ckpt".format(i))
        print ("Model at {} epoch saved at {}".format(i, save_path))

