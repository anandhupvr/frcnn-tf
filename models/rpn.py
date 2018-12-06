import tensorflow as tf
import loader.utils as utils
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from loader import get_anchor
from models import net_vgg
# slim = tf.contrib.slim
from lib.anchor_pre import generate_anchors_pre
from lib.proposal_layer import proposal_layer
from lib.anchor_target import anchor_target_layer




anchor_scales = [8, 16, 32]

checkpoints_dir = 'vgg_16_2016_08_28/vgg16.ckpt'


class RPN:
    def __init__(self):
        self._batch_size = 1

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        # self.im_info = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 2])
        self.box = []
        # self.im_info = self.x.shape[1], self.x.shape[2]
        self.feat_stride = [16,]
        self._anchor_targets = {}
        # self.img = cv2.imread('/home/food/Music/frcnn-tf/dataset/images/apple/apple_10.jpg')
    def _softmax(self, rpn_cls, name):
        if name == 'rpn_cls_softmax':
            shape = tf.shape(rpn_cls)
            reshape_ = tf.reshape(rpn_cls, [-1, shape[-1]])
            reshaped_score = tf.nn.softmax(reshape_, name=name)
            return tf.reshape(reshaped_score, shape)

    def _reshape(self, rpn_cls, num, name):
        with tf.variable_scope(name):
            to_caffe = tf.transpose(rpn_cls, [0, 3, 1, 2])
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num, -1], [tf.shape(rpn_cls)[2]]]))
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box
    def vgg_16(self):
        num_anchors = 9
        
        conv1 = tf.layers.conv2d(self.x,
                                    filters=64,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_1")
        conv2 = tf.layers.conv2d(conv1,
                                    filters=64,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_2")
        pool1 = tf.layers.max_pooling2d(conv2,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_1")
        conv3 = tf.layers.conv2d(pool1,
                                    filters=128,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_3")
        conv4 = tf.layers.conv2d(conv3,
                                    filters=128,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_4")

        pool2 = tf.layers.max_pooling2d(conv4,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_2")

        conv5 = tf.layers.conv2d(pool2,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_5")

        conv6 = tf.layers.conv2d(conv5,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_6")
        conv7 = tf.layers.conv2d(conv6,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_7")

        pool3 = tf.layers.max_pooling2d(conv7,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_3")


        conv8 = tf.layers.conv2d(pool3,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_8")
        conv9 = tf.layers.conv2d(conv8,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_9")
        conv10 = tf.layers.conv2d(conv9,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_10")

        pool3 = tf.layers.max_pooling2d(conv10,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_4")

        conv11 = tf.layers.conv2d(pool3,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_11")
        conv12 = tf.layers.conv2d(conv11,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_12")
        conv13 = tf.layers.conv2d(conv12,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_13")

        rpn1 = tf.layers.conv2d(conv13,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    kernel_initializer ='normal' ,
                                    name='npn_conv/3x3')
        rpn_cls = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name="rpn_out_class")
        rpn_bbox = tf.layers.conv2d(rpn1,
                                    filters=num_anchors * 4,
                                    kernel_size=(1, 1),
                                    activation='linear',
                                    kernel_initializer='uniform',
                                    name='rpn_out_regre')
        # rpn_shape = rpn_cls.shape
        num = 2
        rpn_cls_ = self._reshape(rpn_cls, num, 'rpn_cls_scores_reshape')
        
        rpn_cls_score = self._softmax(rpn_cls_, 'rpn_cls_softmax')
        rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors * 2, "rpn_cls_prob")
        # rpn_cls = tf.reshape(rpn_cls, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], num_anchors, 2])

        # rpn_bbox = tf.reshape(rpn_bbox, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], num_anchors, 4])
        
        return rpn_cls_prob, rpn_bbox, conv13




    def smooth_l1(x):
        l2 = 0.5 * (x**2.0)
        l1 = tf.abs(x) - 0.5

        condition = tf.less(tf.abs(x), 1.0)
        loss = tf.where(condition, l2, l1)
        return loss

    def losses(self, fg_inds, bg_inds, rpn_bbox, rpn_cls, box):
        elosion = 0.00001
        true_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls+elosion), fg_inds))
        false_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls+elosion), bg_inds))
        obj_loss = tf.add(true_obj_loss, false_obj_loss)
        cls_loss = tf.div(obj_loss, 16)

        bbox_loss = smooth_l1(tf.subtract(rpn_bbox, box))
        bbox_loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(bbox_loss), fg_inds))
        bbox_loss = tf.multiply(tf.div(bbox_loss, 1197), 100)
        total_loss = tf.add(cls_loss, bbox_loss)

        return total_loss


    def setup(self, net, rpn_cls, rpn_bbox, img):
        # height = tf.to_int32(tf.ceil(int(img[1]) / np.float32(self.feat_stride[0])))
        # width = tf.to_int32(tf.ceil(int(img[2]) / np.float32(self.feat_stride[0])))
        # height, width = rpn_cls.shape[1:3]
        # img_info = height, width
        anchors, length, img_info = tf.py_func(generate_anchors_pre,
                                    [rpn_cls, self.feat_stride],
                                    [tf.float32, tf.int32, tf.float32], name="generate_anchors")
        # im_info = [tf.to_int32(int(img[1])), tf.to_int32(int(img[2]))]
        anchors.set_shape([None, 4])
        length.set_shape([])
        img_info.set_shape([None, None])
        anchors = anchors
        self._anchors = anchors
        self._anchor_length = length
        img_info = img_info
        
        rois, rpn_scores = tf.py_func(proposal_layer,
                                        [rpn_cls, rpn_bbox, img_info, self._anchors, self._anchor_length],
                                        [tf.float32, tf.float32])
        
        
        rois.set_shape([None, 4])
        rpn_scores.set_shape([None, 1])
        num = 9

        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
                                                                [rpn_cls, self._gt_boxes, img_info, img, self.feat_stride, self._anchors, num],
                                                                [tf.float32, tf.float32, tf.float32, tf.float32])
        
        rpn_labels.set_shape([1, 1, None, None])
        # rpn_labels.set_shape([None])
        rpn_bbox_targets.set_shape([1, None, None, 9 * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, 9 * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, 9 * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        rpn_cls_score = tf.reshape(rpn_cls, [-1, 2])
        rpn_label_los = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label_los, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label_los = tf.reshape(tf.gather(rpn_label_los, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label_los))

        # RPN , bbox loss
        rpn_bbox_pred = rpn_bbox
        rpn_bbox_targets_los = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights_los = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights_los = self._anchor_targets['rpn_bbox_outside_weights']

        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets_los, rpn_bbox_inside_weights_los,
                            rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])
        # loss = losses(fg_inds, bg_inds, rpn_bbox, rpn_cls, box)

        loss = rpn_loss_box + rpn_cross_entropy

        return loss


    def getPlaceholders(self):
        return self.x, self._gt_boxes




