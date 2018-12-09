import numpy as np
import tensorflow as tf
from lib.setup import arrange

class network(arrange):
    def __init__(self, batch_size=1):
        arrange.__init__(self)
        self._batch_size = 1

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        # self.im_info = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 2])
        self.box = []
        self.class_num = 1
        # self.im_info = self.x.shape[1], self.x.shape[2]
        self.feat_stride = [16,]
        # self.rois_ = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 4])
        self._anchor_targets = {}

    def build_network():
        with tf.variable_scope('vgg_16'):
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # vgg net
            net = self.backbone()

            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, initializer)

            # build proposals
            rois = self.build_proposals(rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # build predictions 
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, initializer, initializer_bbox)


            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            return rois, cls_prob, bbox_pred, cls_score

    def backbone(self):
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
                                    filters=64,
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
        return conv13


    def build_rpn(self, net):
        rpn1 = tf.layers.conv2d(net,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    kernel_initializer = initializer,
                                    name='npn_conv/3x3')
        rpn_cls_score = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer = initializer,
                                    name="rpn_out_class")
        rpn_bbox_pred = tf.layers.conv2d(rpn1,
                                    filters=num_anchors * 4,
                                    kernel_size=(1, 1),
                                    activation='linear',
                                    kernel_initializer = initializer,
                                    name='rpn_out_regre')
        # rpn_shape = rpn_cls.shape
        num = 2
        rpn_cls_score_reshape = self._reshape(rpn_cls_score, num, 'rpn_cls_scores_reshape')
        
        rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors * 2, "rpn_cls_prob")

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)

        rpn_labels = self._anchor_targets(rpn_cls_score)

        with tf.control_dependencies([rpn_labels]):
            rois, _ = self._proposal_target_layer([rois, roi_scores])

        return rois

    def build_predictions(self, net, rois, initializer, initializer_bbox):

        pool5 = self._crop_pool_layer(net, rois)
        fc6 = tf.layers.conv2d(pool5, 4096, [7, 7], padding='VALID')
        fc7 = tf.layers.conv2d(fc6, 4096, [1, 1])
        cls_score = tf.layers.conv2d(fc7,
                                    filters=1,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name='rpn_out_classification')
        cls_prob = tf.nn.sotmax(cls_score)

        bbox_prediction = tf.layers.conv2d(fc6,
                                    filters=4,
                                    kernel_size=(1, 1),
                                    activation='linear',
                                    kernel_initializer='uniform',
                                    name='rpn_out_regression')
        
        return cls_score, cls_prob, bbox_prediction

    def getPlaceholders(self):
        return self.x, self._gt_boxes

    def get_loss(self):
        self._losses