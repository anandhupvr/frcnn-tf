from loader.DataLoader import load
import sys
from config.parameters import Config
import tensorflow as tf
# import loader.trainer_utils as t_utils
from models.vgg import vgg16
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import matplotlib.pyplot as plt
slim = tf.contrib.slim



vgg = nets.vgg



config = Config()
# dataset_path = sys.argv[1]
# data_loader = load(dataset_path)
# data = data_loader.data()
# img = plt.imread('dog.jpg')
# img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
predictions = vgg.vgg_16(images)
variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
restorer = tf.train.Saver(variables_to_restore)
init = tf.initialize_all_variables()
for i in variables_to_restore:
	print (i)
	input()
with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess, config.vgg16_path)
    print ("vgg16 restored")
    writer = tf.summary.FileWriter("out")
    writer.add_graph(sess.graph)



