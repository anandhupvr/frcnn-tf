import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

vgg = nets.vgg

# Specify where the Model, trained on ImageNet, was saved.
model_path = 'vgg_16_2016_08_28/vgg16.ckpt'

# Specify where the new model will live:
log_dir = './'

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
predictions = vgg.vgg_16(images)

variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
restorer = tf.train.Saver(variables_to_restore)


for i in variables_to_restore:
	print (i)
	input()
init = tf.initialize_all_variables()

with tf.Session() as sess:
   sess.run(init)
   restorer.restore(sess,model_path)
   print ("model restored")