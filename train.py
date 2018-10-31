from loader.DataLoader import load
import sys
from config.parameters import Config
import tensorflow as tf
# import loader.trainer_utils as t_utils
from models.vgg import vgg16
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import matplotlib.pyplot as plt

vgg = nets.vgg



config = Config()
# dataset_path = sys.argv[1]
# data_loader = load(dataset_path)
# data = data_loader.data()

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
predictions = vgg.vgg_16(images)
variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
restorer = tf.train.Saver(variables_to_restore)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess, config.vgg16_path)
    print ("vgg16 restored")
    pred = sess.run(vgg.vgg_16(plt.imread('dog.jpg')))
    print (pred)
