import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed


from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


# conv(3,3,512,1,1,name='rpn_conv/3x3')
# .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score')
#  tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
_feat_stride = [16,]
anchor_scales = [8, 16, 32]


def rpn_net(base_layer, num_anchors):
    rpn1 = tf.layers.conv2d(base_layer,
                                        filters=512,
                                        kernel_size=(3, 3),
                                        padding='same',
                                        kernel_initializer ='normal' ,
                                        name='npn_conv/3x3')
    x_class = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name="rpn_out_class")
    x_regr = tf.layers.conv2d(rpn1,
                                filters=num_anchors * 4,
                                kernel_size=(1, 1),
                                activation='linear',
                                kernel_initializer='zero',
                                name='rpn_out_regre')

                                
    return [x_class, x_regr]

def rpn_k(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # x = tf.layers.conv2d(base_layers, filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal',name='rpn_conv1')
    # x_class = tf.layers.conv2d(x, filters=num_anchors, kernel_size=(1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')
    # x_regr = tf.layers.conv2d(x, filters=num_anchors*4, kernel_size=(1,1), activation='linear', kernel_initializer='zero', name='rpn-out-regrss')
    x_class = Conv2D(num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr]

def rpn_slim(base_layers, num_anchors):

    x =  tf.nn.conv2d(base_layers, 512, 3, 1, 1, bias=True)
    

    return(x)

def classifier(base_layers, input_rois, num_rois, nb_classes = 1):

    x =  tf.contrib.layers.flatten(base_layers, name="flatten")
    x = tf.layers.dense(x, 4096, activation='relu', name='fc1')
    x = tf.layers.dropout(x, 0.5)
    x = tf.layers.dense(x, 4096, activation='relu', name='fc2')
    x = tf.layers.dropout(x, 0.5)

    out_class = tf.layers.dense(x, nb_classes, activation='softmax', kernel_initializer='zero', name="dense_class")
    out_rgr = tf.layers.dense(out, 4 * (nb_classes - 1), activation='linear', kernel_initializer='zero', name='regresiion')

    return [out_class, out_rgr]