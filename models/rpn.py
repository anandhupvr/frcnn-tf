import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed



# conv(3,3,512,1,1,name='rpn_conv/3x3')
# .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score')

_feat_stride = [16,]
anchor_scales = [8, 16, 32]


def rpn_net(base_layer):
    rpn1 = tf.layers.conv2d(base_layer,
                                        filters=512,
                                        kernel_size=(3, 3),
                                        kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        name='npn_conv/3x3')
    return rpn1

def rpn_k(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # x = tf.layers.conv2d(base_layers, filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal',name='rpn_conv1')
    # x_class = tf.layers.conv2d(x, filters=num_anchors, kernel_size=(1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')
    # x_regr = tf.layers.conv2d(x, filters=num_anchors*4, kernel_size=(1,1), activation='linear', kernel_initializer='zero', name='rpn-out-regrss')
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr]