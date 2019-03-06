import tensorflow as tf
from PIL import Image

import numpy as np



tf.reset_default_graph()

img = Image.open('human.jpg')

im = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
new_graph = tf.Graph()

import pdb; pdb.set_trace()
with tf.Session(graph=new_graph) as sess:
	tf.global_variables_initializer().run()
	saver = tf.train.import_meta_graph('/run/media/user1/disk2/agrima/testing/frcnn-tf/weight/model_400.ckpt.meta')
	checkpoint = tf.train.latest_checkpoint('/run/media/user1/disk2/agrima/testing/frcnn-tf/weight')

	saver.restore(sess, checkpoint)
	print ("model restored")

	img = np.expand_dims(img.resize([224, 224]), axis=0)

	image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')


     