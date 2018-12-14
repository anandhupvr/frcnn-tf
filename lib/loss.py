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


def rpn_cls(rpn_cls_score, rpn_labels):
    rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
    rpn_labels = tf.rshape(rpn_labels, -1)
    return [tf.shape(rpn_cls_score), tf.shape[rpn_labels]]


def losses(rpn_cls_score, rpn_labels, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, cls_score, labels, bbox_prediction, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    rpn_cls_loss = rpn_cls(rpn_cls_score, rpn_labels)
    # rpn_bbox_loss = rpn_bbox(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    
    # rcnn_bbox = rcnn_bbox_los(bbox_prediction, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    # rcnn_cls = rcnn_cls_loss(cls_score, labels)
    # loss = rpn_bbox_loss + rpn_cls_loss + rcnn_bbox + rcnn_cls
    loss =  rpn_cls_loss

    return loss
