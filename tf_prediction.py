import tensorflow as tf
from PIL import Image
from models.net import network
import sys
# from loader.DataLoader import load
import lib.utils as utils

import numpy as np
from config.parameters import Config



tf.reset_default_graph()



C = Config()

bbox_threshold = 0.3



# load = load(dataset_path)


# data = load.get_data()
# data_gen = load.get_anchor_gt(data, C, get_img_output_length, mode='test')


img = Image.open('human.jpg')

# im = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
new_graph = tf.Graph()

class_mapping = {1:'human', 0:'bg'}

with tf.Session(graph=new_graph) as sess:
	# X, Y, image_data, debug_img, debug_num_pos = next(data_gen)
	tf.global_variables_initializer().run()
	saver = tf.train.import_meta_graph('weight/model_400.ckpt.meta')
	checkpoint = tf.train.latest_checkpoint('weight')

	saver.restore(sess, checkpoint)
	print ("model restored")
	img = np.expand_dims(img.resize([224, 224]), axis=0)

	image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
	rpn_reg_out = tf.get_default_graph().get_tensor_by_name('rpn_bbox_pred:0')
	rpn_cls_out = tf.get_default_graph().get_tensor_by_name('rpn_cls_pred:0')

	base_layer = tf.get_default_graph().get_tensor_by_name('conv5_3/Relu:0')
	out_cls = tf.get_default_graph().get_tensor_by_name('class_prediction:0')
	out_box = tf.get_default_graph().get_tensor_by_name('box_prediction:0')
	roi = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

	#box_out = tf.get_default_graph().get_tensor_by_name('dense_regress_2/Reshape_1:0')
	#cls_out = tf.get_default_graph().get_tensor_by_name('dense_class_2/Softmax:0')



	P_rpn = sess.run([rpn_cls_out, rpn_reg_out, base_layer], feed_dict={image_tensor:img})


	R = utils.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', overlap_thresh=0.7)

	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break
		if jk == R.shape[0]//C.num_rois:
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded
		P_cls, P_regr = sess.run([out_cls, out_box], feed_dict={image_tensor:img, roi:ROIs})
		import pdb; pdb.set_trace()
		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold:
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]
			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = utils.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))



		print ("kdk")
		print ("kdkkkkkk")
# dense_class_2/Softmax:0
# dense_regress_2/Reshape_1:0

	print ("hellow ")
	print ("hdh")

     

