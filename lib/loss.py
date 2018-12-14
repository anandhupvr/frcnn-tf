import tensorflow as tf




def smoothL1(x, sigma):
    conditional = tf.less(tf.abs(x), 1/sigma**2)
    close = 0.5 * (sigma * 2) ** 2
    far = tf.abs(x) - 0.5/sigma ** 2

    return tf.where(conditional, close, far)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
        out_loss_box,
        axis=dim
    ))
    return loss_box

# def rpn_cls(rpn_cls_score_reshape, rpn_labels):
#     rpn_cls_score = tf.reshape(rpn_cls_score_reshape, [-1, 2])
#     rpn_label = tf.reshape(rpn_labels, [-1])
#     rpn_select = tf.where(tf.not_equal(rpn_label, -1))
#     rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
#     rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
#     rpn_cross_entropy = tf.reduce_mean(
#             tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
#     return rpn_cross_entropy

# def rpn_bbox(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights):
#     # rpn_bbox_pred = rpn_bbox_pred
#     rpn_bbox_targets = tf.transpose( rpn_bbox_targets,   [ 0, 2, 3, 1])
#     rpn_bbox_inside_weights = tf.transpose( rpn_bbox_inside_weights, [ 0, 2, 3, 1])
#     rpn_bbox_outside_weights = tf.transpose( rpn_bbox_outside_weights,[ 0, 2, 3, 1])

#     rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
#                         rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

#     return rpn_loss_box

def rpn_bbox(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights):
    # rpn_bbox_pred = rpn_bbox_pred
    rpn_bbox_targets = tf.transpose( rpn_bbox_targets,   [ 0, 2, 3, 1])
    rpn_bbox_inside_weights = tf.transpose( rpn_bbox_inside_weights, [ 0, 2, 3, 1])
    rpn_bbox_outside_weights = tf.transpose( rpn_bbox_outside_weights,[ 0, 2, 3, 1])

    diff = tf.multiply(rpn_bbox_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
    diff_sL1 = smoothL1(diff, 3.0)
    rpn_bbox_reg = 10 * tf.reduce_sum(tf.multiply(tf.rpn_bbox_outside_weights, diff_sL1))



    return rpn_bbox_reg


def rpn_cls(rpn_cls_score, rpn_labels):

    shape = tf.shape(rpn_cls_score)

    rpn_cls_score = tf.transpose(rpn_cls_score,[0,3,1,2])
    rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
    rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
    rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2])


    rpn_labels = tf.reshape(rpn_labels, [-1])
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_labels, -1))),[-1, 2])
    rpn_labels = tf.reshape(tf.gather(rpn_labels, tf.where(tf.not_equal(rpn_labels, -1))), [-2])

    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

    return rpn_cross_entropy

# def rcnn_cls_loss(cls_score, labels):
#     # cls_score = cls_score
#     label = tf.reshape(labels, [-1])


#     cross_entropy = tf.reduce_mean(
#         tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=tf.reshape(cls_score, [-1, 1]), labels=label))
#     return cross_entropy



def rcnn_bbox_los(bbox_prediction, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    bbox_pred = bbox_prediction
    # bbox_targets = bbox_targets
    # bbox_inside_weights = bbox_inside_weights
    # bbox_outside_weights = bbox_outside_weights

    loss_box = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    return loss_box

def losses(rpn_cls_score_reshape, rpn_labels, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, cls_score, labels, bbox_prediction, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    # rpn_cls_loss = rpn_cls(rpn_cls_score_reshape, rpn_labels)
    rpn_bbox_loss = rpn_bbox(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    
    # rcnn_bbox = rcnn_bbox_los(bbox_prediction, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    # rcnn_cls = rcnn_cls_loss(cls_score, labels)
    # loss = rpn_bbox_loss + rpn_cls_loss + rcnn_bbox + rcnn_cls
    loss =  rpn_bbox_loss

    return loss
