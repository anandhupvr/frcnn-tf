import numpy as np
import tensorflow as tf
# from lib.anchor_pre import generate_anchors_pre
# from lib.proposal_layer import proposal_layer
# from lib.anchor_target import anchor_target_layer
from lib.proposal_target_layer import proposal_target_layer_py
import numpy.random as npr
from lib.targets import anchor_target_layer_python
from lib.proposal_layer import proposal_layer_py
slim = tf.contrib.slim

class network():
    def __init__(self, batch_size=1):
        self._batch_size = 1

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.im_dims = tf.placeholder(tf.float32, shape=[2])
        # self.im_info = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 2])
        self.box = []
        self.class_num = 1
        # self.im_info = self.x.shape[1], self.x.shape[2]
        self.feat_stride = [16,]
        # self.rois_ = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 4])
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._predictions = {}
        self._losses = {}
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


    def rpn_cls(self):
        rpn_cls_score = tf.reshape(self.rpn_cls_score_reshape, [-1, 2])
        rpn_label = tf.reshape(self.rpn_labels, [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        return rpn_cross_entropy

    def rpn_bbox(self):
        rpn_bbox_pred = self.rpn_bbox_pred
        rpn_bbox_targets = self.rpn_bbox_targets
        rpn_bbox_inside_weights = self.rpn_bbox_inside_weights
        rpn_bbox_outside_weights = self.rpn_bbox_outside_weights

        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                            rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

        return rpn_loss_box

    def rcnn_cls_loss(self):
        cls_score = self._predictions["cls_score"]
        label = tf.reshape(self._predictions["labels"], [-1])


        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(cls_score, [-1, self.class_num]), labels=label))
        return cross_entropy



    def rcnn_bbox_los(self):
        bbox_pred = self._predictions["bbox_pred"]
        bbox_targets = self._predictions["bbox_targets"]
        bbox_inside_weights = self._predictions["bbox_inside_weights"]
        bbox_outside_weights = self._predictions["bbox_outside_weights"]

        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        return loss_box

    def losses(self):
        rpn_cls_loss = self.rpn_cls()
        rpn_bbox_loss = self.rpn_bbox()
        
        rcnn_bbox = self.rcnn_bbox_los()
        rcnn_cls = self.rcnn_cls_loss()
        # loss = rpn_bbox_loss + rpn_cls_loss + rcnn_bbox + rcnn_cls
        loss = rpn_cls_loss + rpn_bbox_loss + rcnn_bbox + rcnn_cls
 
        return loss


    def anchor_target_layer(self, rpn_cls_score, _gt_boxes, im_dims, feat_stride):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        tf.py_func(anchor_target_layer_python, [rpn_cls_score, _gt_boxes, im_dims, feat_stride],
            [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    def proposal_layer(self, rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, feat_strides):
        blob = tf.py_func(proposal_layer_py,
                        [rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, feat_strides],
                        [tf.float32])
        blob = tf.reshape(blob, [-1, 5])

        return blob

    def proposal_target_layer(self, rpn_rois, _gt_boxes, num_classes):
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func( proposal_target_layer_py,
                                                                                            [ rpn_rois, _gt_boxes, num_classes],
                                                                                            [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32])
        rois = tf.reshape( rois, [-1, 5], name = 'rois')
        labels = tf.convert_to_tensor( tf.cast(labels, tf.int32),name = 'labels')
        bbox_targets = tf.convert_to_tensor( bbox_targets, name = 'bbox_targets')
        bbox_inside_weights = tf.convert_to_tensor( bbox_inside_weights, name = 'bbox_inside_weights')
        bbox_outside_weights = tf.convert_to_tensor( bbox_outside_weights, name = 'bbox_outside_weights')
        
        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def build_network(self):
        with tf.variable_scope('vgg_16'):
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # vgg net
            net = self.backbone()

            self.rpn_cls_prob, self.rpn_bbox_pred, self.rpn_cls_score, self.rpn_cls_score_reshape = self.build_rpn(net, initializer)

            self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                self.anchor_target_layer( self.rpn_cls_score, self._gt_boxes, self.im_dims, self.feat_stride)

            blob = self.proposal_layer(self.rpn_cls_prob, self.rpn_bbox_pred, self.im_dims,self.feat_stride)


            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.proposal_target_layer(blob, self._gt_boxes, self.class_num)


            # pooled = self._crop_pool_layer(net, rois)

            # cls_score, cls_prob, bbox_prediction = self.build_predictions(pooled, initializer, initializer_bbox)


            # self._predictions["cls_score"] = cls_score
            # self._predictions["bbox_pred"] = bbox_prediction

            # self._predictions["labels"] = labels
            # self._predictions["bbox_targets"] = bbox_targets
            # self._predictions["bbox_inside_weights"] = bbox_inside_weights
            # self._predictions["bbox_outside_weights"] = bbox_outside_weights


            # return cls_score, cls_prob, bbox_prediction
            return rois

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


    def build_rpn(self, net, initializer):
        num_anchors = 9
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



    def _crop_pool_layer(self, bottom, rois):
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # Get the normalized coordinates of bboxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        # Won't be backpropagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = 7 * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')


    def build_predictions(self, pooled, initializer, initializer_bbox):

        fc7 = tf.contrib.layers.flatten(pooled)
        # cls_score = tf.layers.conv2d(fc7,
        #                             filters=1,
        #                             kernel_size=(1, 1),
        #                             activation='sigmoid',
        #                             kernel_initializer=initializer,
        #                             name='rpn_out_classification')
        # cls_prob = tf.nn.softmax(cls_score)

        # bbox_prediction = tf.layers.conv2d(fc7,
        #                             filters=4,
        #                             kernel_size=(1, 1),
        #                             activation='linear',
        #                             kernel_initializer=initializer_bbox,
        #                             name='rpn_out_regression')
        
        # return cls_score, cls_prob, bbox_prediction
        return fc7


    def getPlaceholders(self):
        return self.x, self._gt_boxes, self.im_dims
