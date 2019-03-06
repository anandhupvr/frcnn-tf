from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import sys
from loader.DataLoader import load
from models.net import network
import lib.loss as losses
from config.parameters import Config
import lib.utils as utils

import lib.ls as lss


C = Config()
def get_img_output_length(width, height):
    return (int(width/16),int(height/16))



roi_input = tf.placeholder(tf.float32, shape=[1, None, 4])

num_rois = 4


num_epo = 500
dataset_path = sys.argv[1]
load = load(dataset_path)


data = load.get_data()
num_anchors = 9
data_gen = load.get_anchor_gt(data, C, get_img_output_length, mode='train')


net = network()
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
# feature = net.build_network()
# rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, feature = net.build_network()
rpn_out = net.build_network()
x, cls_plc, box_plc = net.getPlaceholders()


lsr = lss.rpn_loss_cls_org(9)
lgr = lss.rpn_loss_regr_org(9)
# rg = lss.rpn_loss_regr(9)
los_c = lsr(cls_plc, rpn_out[0])
los_b = lgr(box_plc, rpn_out[1])
rpn_loss = los_c + los_b
# rpn_loss = losses.rpn()
class_mapping = {'human': 0, 'bg': 1}

classifier = net.classifier(rpn_out[2], roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

lab_cls = tf.placeholder(tf.float32, shape=classifier[0].shape)
lab_reg = tf.placeholder(tf.float32, shape=[1, None, 8])
clf = lss.class_loss_regr(1)
clf_cls = lss.class_loss_cls(lab_cls, classifier[0])
clf_reg = clf(lab_reg, classifier[1])
clf_loss = clf_cls + clf_reg
total_loss = rpn_loss + clf_loss
# classification = net.build_predictions(rpn_out[2], roi_input, initializer, initializer_bbox)

train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
# train__step_cls = tf.train.AdamOptimizer(1e-4).minimize(clf_loss)

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epo):
        # import pdb; pdb.set_trace()
        los = 0
        for _ in range(256):
            X, Y, image_data, debug_img, debug_num_pos = next(data_gen)
            # sess.run(train_step_rpn, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            P_rpn = sess.run(rpn_out, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})

            R = utils.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
            X2, Y1, Y2, IouS = utils.calc_iou(R, image_data, C, class_mapping)

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            if num_rois > 1:
                if len(pos_samples) < num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)
            import pdb; pdb.set_trace()
            sess.run(train_step, feed_dict={rpn_out[2]:P_rpn[2], roi_input:X2[:, sel_samples, :], lab_cls:Y1[:, sel_samples, :], lab_reg:Y2[:, sel_samples, :], x:X, cls_plc:Y[0], box_plc:Y[1]})
            ls_val = sess.run(total_loss, feed_dict={rpn_out[2]:P_rpn[2], roi_input:X2[:, sel_samples, :], lab_cls:Y1[:, sel_samples, :], lab_reg:Y2[:, sel_samples, :], x:X, cls_plc:Y[0], box_plc:Y[1]})
            total_loss = ls_val + los
            los = ls_val
            print (total_loss)
        print ("epoch : %s    ******** losss : %s ***** "%(i,total_loss/256))


















'''


            sess.run(train_step, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            ls_val = sess.run(rpn_loss, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            
            total_loss = ls_val + los
            los = ls_val
            # print ("epoch : %s   loss  %s "%(_,ls_val))
        print ("epoch : %s    ******** losss : %s ***** "%(i,total_loss/256))

        if i%100 == 0:
            save_path = saver.save(sess, 'weight/'+"model_{}.ckpt".format(i))
            print ("epoch : %s   saved at  %s "%(i,save_path))

'''