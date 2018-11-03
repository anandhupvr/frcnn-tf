from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
from models import net_vgg
from loader import get_anchor
from models import rpn
# vgg16 = net_vgg.vgg_16()
slim = tf.contrib.slim


checkpoints_dir = 'vgg_16_2016_08_28/vgg16.ckpt'

with tf.Graph().as_default():
    img = tf.read_file('dog.jpg')
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.cast(image, tf.float32)
    resize_fn = tf.image.resize_image_with_crop_or_pad
    image_resized = resize_fn(image, 224, 224)
    processed_images = tf.expand_dims(image_resized, 0)
    # print (processed_images.shape)
    with slim.arg_scope(net_vgg.vgg_arg_scope()):
        logits, _ = net_vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
    net = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
                                            checkpoints_dir,
                                            slim.get_model_variables('vgg_16'))
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("out")
    with tf.Session() as sess:
        init_fn(sess)
        np_image, network_input, net = sess.run([image,
                                                    processed_images,
                                                    net])
        
        # writer = tf.summary.FileWriter("out")
        # writer.add_graph(sess.graph)
    print (net.shape)   #  ithaanu vgg16 nu ingotekk return cheythath 7x7x512 ithene kond enik prosal layer l kodukanam 
    print (network_input.shape)
    anchors = get_anchor.generate_anchors()
    num_anchors = anchors.shape[0]
    _feat_stride = [16,]
    # x = tf.placeholder(tf.float32, shape=(7, 7, 512))
    rp = rpn.rpn_k(net, num_anchors)
    with tf.Session() as sess:
        clss, rg= rsess.run(rp)
    print (clss.shape)