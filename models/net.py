import numpy as np
import tensorflow as tf
# from lib.anchor_pre import generate_anchors_pre
# from lib.proposal_layer import proposal_layer
# from lib.anchor_target import anchor_target_layer
from lib.proposal_target_layer import proposal_target_layer_py
import numpy.random as npr
from lib.targets import anchor_target_layer_python
from lib.proposal_layer import proposal_layer_py
from models import vgg


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

        # self.feature_vector     = feature_vector
        # self.ground_truth       = ground_truth
        # self.im_dims            = im_dims
        # self.anchor_scale       = anchor_scale

        self.RPN_OUTPUT_CHANNEL = 512
        self.RPN_KERNEL_SIZE    = 3
        self.feat_stride        = 16

        self.weights            = {
        'w_rpn_conv1'     : tf.Variable(tf.random_normal([ self.RPN_KERNEL_SIZE, self.RPN_KERNEL_SIZE, 512, self.RPN_OUTPUT_CHANNEL ], stddev = 0.01)),
        'w_rpn_cls_score' : tf.Variable(tf.random_normal([ 1, 1, self.RPN_OUTPUT_CHANNEL, 18  ], stddev = 0.01)),
        'w_rpn_bbox_pred' : tf.Variable(tf.random_normal([ 1, 1, self.RPN_OUTPUT_CHANNEL, 36  ], stddev = 0.01))
        }




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



    def build_network(self):
        with tf.variable_scope('vgg_16'):
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            vgg_16 = vgg.ConvNetVgg16('/home/christie/junk/frcnn-tf/vgg16.npy')
            cnn = vgg_16.inference(x)
            features = vgg_16.get_features()

            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(feature, initializer)
            # rpn_cls_score, rpn_bbox_pred = self.build_rpn(feature)

            # self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
            #     self.anchor_target_layer( self.rpn_cls_score, self._gt_boxes, self.im_dims, self.feat_stride)

            # blob = self.proposal_layer(self.rpn_cls_prob, self.rpn_bbox_pred, self.im_dims,self.feat_stride)

            # rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.proposal_target_layer(blob, self._gt_boxes, self.class_num)


            # pooled = self._crop_pool_layer(net, rois)

            # cls_score, cls_prob, bbox_prediction = self.build_predictions(pooled, initializer, initializer_bbox)


            # self._predictions["cls_score"] = cls_score
            # self._predictions["bbox_pred"] = bbox_prediction

            # self._predictions["labels"] = labels
            # self._predictions["bbox_targets"] = bbox_targets
            # self._predictions["bbox_inside_weights"] = bbox_inside_weights
            # self._predictions["bbox_outside_weights"] = bbox_outside_weights


            # return cls_score, cls_prob, bbox_prediction
            return rpn_cls_prob




    # def build_rpn(self, feature_vector):

    #     # rpn_conv1
    #     # slide a network on the feature map, for each nxn (n = 3), use a conv kernel to produce another feature map.
    #     # each pixel in this fature map in an anchor 
    #     ksize      = self.RPN_KERNEL_SIZE
    #     feat       = tf.nn.conv2d( feature_vector, self.weights['w_rpn_conv1'], strides = [1, 1, 1, 1], padding = 'SAME' )
    #     feat       = tf.nn.relu( feat )
    #     self.feat  = feat

    #     # for each anchor, propose k anchor boxes, 
    #     # for each box, regress: objectness score and coordinates

    #     # box-classification layer ( objectness scor)
    #     with tf.variable_scope('cls'):
    #         self.rpn_cls_score = tf.nn.conv2d(feat, self.weights['w_rpn_cls_score'], strides = [ 1, 1, 1, 1], padding = 'SAME')

    #     # bounding-box prediction 
    #     with tf.variable_scope('reg'): 
    #         self.rpn_reg_pred  = tf.nn.conv2d(feat, self.weights['w_rpn_bbox_pred'], strides = [1, 1, 1, 1], padding = 'SAME')
       

    #     return self.rpn_cls_score, self.rpn_reg_pred


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
        rpn_shape = rpn_cls.shape
        num = 2
        rpn_cls_score_reshape = self._reshape(rpn_cls_score, num, 'rpn_cls_scores_reshape')
        
        rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors * 2, "rpn_cls_prob")

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape
  





    def getPlaceholders(self):
        return self.x, self._gt_boxes, self.im_dims
