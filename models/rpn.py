import tensorflow as tf
import loader.utils as utils
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from loader import get_anchor



_feat_stride = [16,]
anchor_scales = [8, 16, 32]


def rpn_net(net, num_anchors, processed_images, data):
    rpn1 = tf.layers.conv2d(net,
                                filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer ='normal' ,
                                name='npn_conv/3x3')
    classe = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name="rpn_out_class")
    reg = tf.layers.conv2d(rpn1,
                                filters=num_anchors * 4,
                                kernel_size=(1, 1),
                                activation='linear',
                                kernel_initializer='uniform',
                                name='rpn_out_regre')

    # rpn_cls_score_reshape = utils.to_NCHW_format(bottom=x_class,
    #                                                             num_dim=2,
    #                                                             name='rpn_cls_score_reshape')
    # rpn_cls_prob_reshape = tf.reshape(tf.nn.softmax(tf.reshape(rpn_cls_score_reshape,
    #                                                             [-1, tf.shape(rpn_cls_score_reshape)[-1]]),
    #                                                 name='rpn_cls_prob_reshape'),
    #                                     tf.shape(rpn_cls_score_reshape))

    # rpn_cls_prob = utils.to_NCHW_format(bottom=rpn_cls_prob_reshape,
    #                                     num_dim=num_anchors * 2,
    #                                     name='rpn_cls_prob')

    # return [rpn_cls_prob, x_regr]

    anchors = get_anchor.generate_anchors()

    num_anchors =  anchors.shape[0]
    width = int(np.shape(net)[1])
    height = int(np.shape(net)[2])
    print (net.shape)
    img_width = int(processed_images.shape[1])
    img_height = int(processed_images.shape[2])

    num_feature_map = width * height

    # Calculate output w, h stride
    w_stride = img_width / width
    h_stride = img_height / height





    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()

    all_anchors = (anchors.reshape( (1, 9, 4)) +
                                    shifts.reshape( (1, num_feature_map, 4) ).transpose((1, 0, 2)) )


    total_anchors = num_feature_map * 9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    # utils.bbox_plot(all_anchors)
    reg = np.reshape(reg, (-1, 4))
    classe = np.reshape(classe, (-1, 1))

    proposals = utils.bbox_transform_inv(all_anchors, reg)

    proposals = utils.clip_boxes(proposals, (np.array([int(processed_images.shape[1]), int(processed_images.shape[2])], dtype='float32')))
    keep = utils.filter_boxes(proposals, 40)
    proposals = proposals[keep, :]
    scores = classe[keep]


    box = np.array([data[0][0].x, data[0][0].y, data[0][0].w, data[0][0].h], dtype='float32')
    box = box.reshape(1, 4)

    overlaps = utils.bbox_overlaps(proposals, box)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(proposals)), gt_assignment]
    qt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[qt_argmax_overlaps,
                                np.arange(overlaps.shape[1])]
    qt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels = np.empty((len(proposals), ), dtype=np.float32)
    labels.fill(-1)

    labels[qt_argmax_overlaps] = 1
    labels[max_overlaps >= .7] = 1
    labels[max_overlaps < .3] = 0 

    fg_inds = np.where(labels == 1)[0]

    num_bg = int(len(fg_inds) * 2)
    bg_inds = np.where(labels == 0)[0]

    if len(bg_inds) > num_bg:
        disble_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disble_inds] = -1

    batch_inds = (proposals[labels != -1])

    batch_inds = (batch_inds / 9).astype(np.int)

    k = [i for i in range(len(proposals))]
    full_labels = utils.unmap(labels, len(proposals), k, fill=-1)

    batch_label_targets = full_labels.reshape(-1, 1, 1, 1 * 9)[batch_inds]

    bbox_targets = np.zeros((len(proposals), 4), dtype=np.float32)

    pos_anchors = proposals[labels == 1]
    bbox_targets = utils.bbox_transform(pos_anchors , box[gt_assignment, :][labels == 1])
    a = [i for i in range(len(labels)) if labels[i]==1]
    bbox_targets = utils.unmap(bbox_targets, len(proposals), a, fill=0)



    batch_bbox_targets = bbox_targets.reshape(-1, 1, 1, 4 * 9)[batch_inds]

    padded_fcmap = np.pad(net, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')

    padded_fcmap = np.squeeze(padded_fcmap)
    batch_tiles=[]
    dd = [i for i in range(len(labels)) if labels[i] != -1]
    for ind in dd:
        x = ind % width
        y = int(ind / width)
        fc_3x3 = padded_fcmap[y:y+3, x:x+3,:]
        batch_tiles.append(fc_3x3)
    # last = np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist()


    return np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist(), proposals

# def rpn_k(base_layers, num_anchors):

#     x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
#     # x = tf.layers.conv2d(base_layers, filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal',name='rpn_conv1')
#     # x_class = tf.layers.conv2d(x, filters=num_anchors, kernel_size=(1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')
#     # x_regr = tf.layers.conv2d(x, filters=num_anchors*4, kernel_size=(1,1), activation='linear', kernel_initializer='zero', name='rpn-out-regrss')
#     x_class = Conv2D(num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
#     x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

#     return [x_class, x_regr]

# def rpn_slim(base_layers, num_anchors):

#     x =  tf.nn.conv2d(base_layers, 512, 3, 1, 1, bias=True)
    

#     return(x)

def classifier(base_layers, input_rois, num_rois, nb_classes = 1):

    x =  tf.contrib.layers.flatten(base_layers, name="flatten")
    x = tf.layers.dense(x, 4096, activation='relu', name='fc1')
    x = tf.layers.dropout(x, 0.5)
    x = tf.layers.dense(x, 4096, activation='relu', name='fc2')
    x = tf.layers.dropout(x, 0.5)

    out_class = tf.layers.dense(x, nb_classes, activation='softmax', kernel_initializer='zero', name="dense_class")
    out_rgr = tf.layers.dense(out, 4 * (nb_classes - 1), activation='linear', kernel_initializer='zero', name='regresiion')

    return [out_class, out_rgr]


def _crop_pool_layer(bottom, rois, name):
    with tf.variable_scope(name):

        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"))
        # Get the normalized coordinates of bboxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(_feat_stride)
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(_feat_stride)
        x1 = tf.slice(rois, [0, 0], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 1], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 2], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 3], [-1, 1], name="y2") / height
        # Won't be backpropagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = 7 * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return tf.layers.max_pooling2d(crops, [2, 2], padding='SAME')

def predictions(base_layers, rois):
    
    pool5 = _crop_pool_layer(base_layers, rois, "pool5")
    print (pool5)



def rpn_net2(net, num_anchors, processed_images, data, cat=1):
    rpn1 = tf.layers.conv2d(net,
                                filters=512,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer ='normal' ,
                                name='npn_conv/3x3')
    classe = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name="rpn_out_class")
    reg = tf.layers.conv2d(rpn1,
                                filters=num_anchors * 4,
                                kernel_size=(1, 1),
                                activation='linear',
                                kernel_initializer='uniform',
                                name='rpn_out_regre')

    # rpn_cls_score_reshape = utils.to_NCHW_format(bottom=x_class,
    #                                                             num_dim=2,
    #                                                             name='rpn_cls_score_reshape')
    # rpn_cls_prob_reshape = tf.reshape(tf.nn.softmax(tf.reshape(rpn_cls_score_reshape,
    #                                                             [-1, tf.shape(rpn_cls_score_reshape)[-1]]),
    #                                                 name='rpn_cls_prob_reshape'),
    #                                     tf.shape(rpn_cls_score_reshape))

    # rpn_cls_prob = utils.to_NCHW_format(bottom=rpn_cls_prob_reshape,
    #                                     num_dim=num_anchors * 2,
    #                                     name='rpn_cls_prob')

    # return [rpn_cls_prob, x_regr]

    anchors = get_anchor.generate_anchors()

    num_anchors =  anchors.shape[0]
    width = int(np.shape(net)[1])
    height = int(np.shape(net)[2])
    print (net.shape)
    img_width = int(processed_images.shape[1])
    img_height = int(processed_images.shape[2])

    num_feature_map = width * height

    # Calculate output w, h stride
    w_stride = img_width / width
    h_stride = img_height / height





    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()

    all_anchors = (anchors.reshape( (1, 9, 4)) +
                                    shifts.reshape( (1, num_feature_map, 4) ).transpose((1, 0, 2)) )


    total_anchors = num_feature_map * 9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    # utils.bbox_plot(all_anchors)
    reg = np.reshape(reg, (-1, 4))
    classe = np.reshape(classe, (-1, 1))

    proposals = utils.bbox_transform_inv(all_anchors, reg)

    proposals = utils.clip_boxes(proposals, (np.array([int(processed_images.shape[1]), int(processed_images.shape[2])], dtype='float32')))
    keep = utils.filter_boxes(proposals, 40)
    proposals = proposals[keep, :]
    scores = classe[keep]


    box = np.array([data[0][0].x, data[0][0].y, data[0][0].w, data[0][0].h], dtype='float32')
    box = box.reshape(1, 4)

    pre_nms_topN = 6000
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    post_nms_topN = 300
    keep = utils.py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    FG_FRAC=.25
    FG_THRESH=.5
    BG_THRESH_HI=.5
    BG_THRESH_LO=.1
    BATCH = 256
    proposals = np.vstack((proposals, box))

    overlaps = utils.bbox_overlaps(proposals, box)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    fg_rois_per_this_image = min(int(BATCH * FG_FRAC), fg_inds.size)

    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = BATCH - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # labels = labels[keep_inds]
    rois = proposals[keep_inds]
    gt_rois = box[gt_assignment[keep_inds]]

    targets = utils.bbox_transform(rois, gt_rois)#input rois
    rois_num=targets.shape[0]
    batch_box=np.zeros((rois_num, 200, 4))
    # import pdb; pdb.set_trace()
    for i in range(rois_num):
        batch_box[i, cat] = targets[i]
    batch_box = np.reshape(batch_box, (rois_num, -1))
    # get gt category
    batch_categories = np.zeros((rois_num, 200, 1))
    for i in range(rois_num):
        batch_categories[i, cat] = 1
    batch_categories = np.reshape(batch_categories, (rois_num, -1))

    return rois, batch_box, batch_categories