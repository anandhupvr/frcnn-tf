import numpy as np
import tensorflow as tf
from lib.proposal_target_layer import proposal_target_layer_py
import numpy.random as npr
from lib.targets import anchor_target_layer_python
from lib.proposal_layer import proposal_layer_py
from models import vgg
from models.RoiPooling import RoiPoolingConv

slim = tf.contrib.slim

from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed



class network():
    def __init__(self, batch_size=1):
        self._batch_size = 1

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3], name="input_image")
        self.cls_plc = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 18], name="rpn_cls")
        self.box_plc = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 72], name="rpn_box")
        # self.im_info = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 2])
        self.box = []
        self.class_num = 2
        # self.im_info = self.x.shape[1], self.x.shape[2]
        self.feat_stride = [16,]
        # self.rois_ = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 4])
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._predictions = {}
        self._losses = {}
        self.num_classes = 2


        self.p_drop                  = 0.5 
        self.bn1                     = BatchNorm( name='bn1')
        self.bn2                     = BatchNorm( name='bn2')

        self.weights                 = {
            'wfc1':     tf.Variable( tf.random_normal([49*512,  1024], stddev=0.005)), 
            'wfc2':     tf.Variable( tf.random_normal([1024,1024], stddev=0.005)),
            'wfccls':   tf.Variable( tf.random_normal([1024,self.num_classes    ], stddev=0.005)),
            'wfcbbox':  tf.Variable( tf.random_normal([1024,self.num_classes*4  ], stddev=0.005))
        } 

        self.biases                  = {
            'bfc1':     tf.Variable( tf.ones([1024])),
            'bfc2':     tf.Variable( tf.ones([1024])),
            'bfccls':   tf.Variable( tf.ones([self.num_classes])),
            'bfcbbox':  tf.Variable( tf.ones([self.num_classes*4]))
        }




    def build_network(self):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        vgg_16 = vgg.ConvNetVgg16('vgg16.npy')
        cnn = vgg_16.inference(self.x)
        features = vgg_16.get_features()


        rpn_cls_score, rpn_bbox_pred = self.build_rpn(features, initializer)
        return [rpn_cls_score, rpn_bbox_pred, features]


    def build_rpn(self, net, initializer):
        num_anchors = 9
        rpn1 = tf.layers.conv2d(net,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    kernel_initializer = initializer,
                                    name='npn_conv/3x3')
        rpn_cls_score = tf.layers.conv2d(rpn1,
                                    filters=num_anchors,
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
        rpn_cls = tf.reshape(rpn_cls_score, [1, 14, 14, 9], name='rpn_cls_pred')
        rpn_bbox = tf.reshape(rpn_bbox_pred, [1, 14, 14, 36], name='rpn_bbox_pred')

        # num = 2
        # rpn_cls_score_reshape = self._reshape(rpn_cls_score, num, 'rpn_cls_scores_reshape')
        
        # rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        # rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        # rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors , "rpn_cls_prob")

        return rpn_cls, rpn_bbox
  
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

        return tf.layers.max_pooling2d(inputs=crops, pool_size=[2, 2], strides=2)



    def build_predictions(self, feature, rois, initializer, initializer_bbox):

        pooled = self._crop_pool_layer(feature, rois)

        pooled_features = tf.contrib.layers.flatten(pooled)
        # Fully connected layers
        fc1 = tf.nn.dropout( self.fc(pooled_features, self.weights['wfc1'], self.biases['bfc1']), self.p_drop)
        fc2 = tf.nn.dropout( self.fc(fc1, self.weights['wfc2'], self.biases['bfc2']), self.p_drop)
        feature = fc2

        with tf.variable_scope('cls'):
            rcnn_cls_score      = self.fc(feature, self.weights['wfccls'], self.biases['bfccls'] ) 

        with tf.variable_scope('bbox'):
            rcnn_bbox_refine    = self.fc(feature, self.weights['wfcbbox'],self.biases['bfcbbox'])

        # cls_prob = tf.nn.softmax(rcnn_cls_score, name="rcnn_class_prob")
        predictions = [rcnn_cls_score, rcnn_bbox_refine]
        return predictions



    def classifier(self, base_layers, input_rois, num_rois, nb_classes = 1, trainable=False):
        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        if K.backend() == 'tensorflow':
            pooling_regions = 7
            input_shape = (num_rois,7,7,512)
        elif K.backend() == 'theano':
            pooling_regions = 7
            input_shape = (num_rois,512,7,7)
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out_class = TimeDistributed(Dense(nb_classes-1, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
        out_class = tf.reshape(out_class, [1, 8, 1], name='class_prediction')
        out_regr = tf.reshape(out_regr, [1, 8, 4], name='box_prediction')
        return [out_class, out_regr]










    def fc(self,x,W,b):          
        h                            = tf.matmul(x, W) + b
        h                            = tf.nn.relu(h)
        return h


    def getPlaceholders(self):
        return self.x, self.cls_plc, self.box_plc


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name     = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)
