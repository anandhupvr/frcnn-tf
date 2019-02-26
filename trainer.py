from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import sys
from loader.DataLoader import load
from models.net import network
import lib.loss as losses
from config.parameters import Config


import lib.ls as lss


C = Config()
def get_img_output_length(width, height):
    return (int(width/16),int(height/16))



rois = tf.placeholder(tf.float32, shape=[None, 5])


num_epo = 500
dataset_path = sys.argv[1]
load = load(dataset_path)


data = load.get_data()
num_anchors = 9
data_gen = load.get_anchor_gt(data, C, get_img_output_length, mode='train')


net = network()
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
# feature = net.build_network()
# rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, feature = net.build_network()
rpn_out = net.build_network()
x, cls_plc, box_plc = net.getPlaceholders()

lsr = lss.rpn_loss_cls_org(9)
lgr = lss.rpn_loss_regr_org(9)
# rg = lss.rpn_loss_regr(9)
los_c = lsr(cls_plc, rpn_out[0])
los_b = lgr(box_plc, rpn_out[1])
rpn_loss = los_c + los_b
# rpn_loss = losses.rpn()

# classification = net.build_predictions(rpn_out[3], rois, initializer, initializer_bbox)

saver = tf.train.Saver()


with tf.Session() as sess:
    for i in range(num_epo):
        for _ in range(284):
            X, Y, image_data, debug_img, debug_num_pos = next(data_gen)

            sess.run(tf.global_variables_initializer())
            # loss_ = sess.run(rpn_loss, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            train_step = optimizer.minimize(rpn_loss)
            sess.run(train_step, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            ls_val = sess.run(rpn_loss, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            print ("epoch : %s   loss  %s "%(i,ls_val))

    if i == 100:
        save_path = saver.save(sess, ''+"model_{}.ckpt".format(i))
        print ("epoch : %s   saved at  %s "%(i,save_path))





saver = tf.train.Saver()

# init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epo):
        for _ in range(len(open("train.txt", "r").readlines())):
            data = data_loader.data_batch()
            img, gt_box, im_info = data[0][0], data[0][1], data[0][2]
            loss = losses(rpn_cls_score_reshape, rpn_labels, rpn_bbox_pred, \
                            rpn_bbox_targets, rpn_bbox_inside_weights, \
                            rpn_bbox_outside_weights, cls_score, labels, \
                            bbox_prediction, bbox_targets, bbox_inside_weights, \
                            bbox_outside_weights)
            #train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            train_step = optimizer.minimize(loss)
            sess.run(train_step, feed_dict={x:img, gt_boxes:gt_box, im_dims:im_info})
            ls_val = sess.run(loss, feed_dict={x:img, gt_boxes:gt_box, im_dims:(im_info)})
            print ('loss : {}       --> : {}'.format(ls_val, _))
        print ('loss : {}      epoch --> : {}'.format(ls_val, i))
    if i%5 == 0:
        save_path = saver.save(sess, 'weights/'+"model_{}.ckpt".format(i))
        print ("Model at {} epoch saved at {}".format(i, save_path))

