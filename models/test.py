from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
# from models import net_vgg
# vgg16 = net_vgg.vgg_16


checkpoints_dir = 'vgg_16_2016_08_28/vgg16.ckpt'

with tf.Graph().as_default():
    img = tf.read_file('dog.jpg')
    image = tf.image.decode_jpeg(img, channels=3)
    resize_fn = tf.image.resize_image_with_crop_or_pad
    image_resized = resize_fn(image, 224, 224)
    processed_images = tf.expand_dims(processed_images, 0)
    print (processed_images.shape)
