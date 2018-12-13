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
from models import vgg

num_epo = 500
import pdb; pdb.set_trace()
dataset_path = sys.argv[1]
data_loader = load(dataset_path)
x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
net = network()
# cls_score, cls_prob, bbox_pred = net.build_network()
# test = net.build_network()
im = np.expand_dims(cv2.imread('/home/christie/junk/frcnn-tf/dog.jpg'), axis=0)
vgg_16 = vgg.ConvNetVgg16('/home/christie/junk/frcnn-tf/vgg16.npy')
cnn = vgg_16.inference(x)
features = vgg_16.get_features()
# test = net.build_network(features)


# x, gt_boxes, im_dims = net.getPlaceholders()

# x, gt_boxes, im_dims = net.getPlaceholders()
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'

# rpn_net = RPN()
# rpn_cls, rpn_bbox, net = rpn_net.vgg_16()
# x, gt_boxes = rpn_net.getPlaceholders()

# saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(init_op)
    for i in range(num_epo):
        for _ in range(len(open("train.txt", "r").readlines())):
            # data = data_loader.data_batch()
            # img, gt_box, im_info = data[0][0], data[0][1], data[0][2]
            # loss = net.losses()
            # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            import pdb; pdb.set_trace()
            feat = sess.run(features, feed_dict={x:im})
            # sess.run(train_step, feed_dict={x:img, gt_boxes:gt_box, im_dims:im_info})
            # ls_val = sess.run(test, feed_dict={x:img, gt_boxes:gt_box, im_dims:(im_info)})
            # print ('loss : {}       --> : {}'.format(ls_val, _))
    #     print ('loss : {}      epoch --> : {}'.format(ls_val, i))
    # if i%100 == 0:
    #     save_path = saver.save(sess, 'weights/'+"model_{}.ckpt".format(i))
    #     print ("Model at {} epoch saved at {}".format(i, save_path))

