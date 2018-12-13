import tensorflow as tf




def rpn_cls():
    rpn_cls_score = tf.reshape(rpn_cls_score_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_labels, [-1])
    rpn_select = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
    rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
    return rpn_cross_entropy

def rpn_bbox():
    rpn_bbox_pred = rpn_bbox_pred
    rpn_bbox_targets = rpn_bbox_targets
    rpn_bbox_inside_weights = rpn_bbox_inside_weights
    rpn_bbox_outside_weights = rpn_bbox_outside_weights

    rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                        rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

    return rpn_loss_box

def rcnn_cls_loss(self):
    cls_score = _predictions["cls_score"]
    label = tf.reshape(_predictions["labels"], [-1])


    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(cls_score, [-1, class_num]), labels=label))
    return cross_entropy



def rcnn_bbox_los(self):
    bbox_pred = _predictions["bbox_pred"]
    bbox_targets = _predictions["bbox_targets"]
    bbox_inside_weights = _predictions["bbox_inside_weights"]
    bbox_outside_weights = _predictions["bbox_outside_weights"]

    loss_box = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    return loss_box

def losses():
    rpn_cls_loss = rpn_cls()
    rpn_bbox_loss = rpn_bbox()
    
    rcnn_bbox = rcnn_bbox_los()
    rcnn_cls = rcnn_cls_loss()
    # loss = rpn_bbox_loss + rpn_cls_loss + rcnn_bbox + rcnn_cls
    loss = rpn_cls_loss + rpn_bbox_loss + rcnn_bbox + rcnn_cls

    return loss
