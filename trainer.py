from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
from models import net_vgg
from loader import get_anchor
from models import rpn
import loader.utils as utils
# vgg16 = net_vgg.vgg_16()
slim = tf.contrib.slim


tf.enable_eager_execution()


checkpoints_dir = 'vgg_16_2016_08_28/vgg16.ckpt'

with tf.Graph().as_default():

    img = tf.read_file('dog.jpg')
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.cast(image, tf.float32)
    resize_fn = tf.image.resize_image_with_crop_or_pad
    image_resized = resize_fn(image, 224, 224)
    processed_images = tf.expand_dims(image_resized, 0)
    # print (processed_images.shape)
    with slim.arg_scope(net_vgg.vgg_arg_scope()):
        logits, _ = net_vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
    net = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
                                            checkpoints_dir,
                                            slim.get_model_variables('vgg_16'))




    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("./graphs", sess.graph)
        init_fn(sess)
        np_image, network_input, net = sess.run([image,
                                                    processed_images,
                                                    net])
        # summary = sess.run(merged)
        # writer.add_summary(summary)

    net = np.reshape(net, [1, 14, 14, 512])
    net = tf.Variable(net)

    num_anchors = get_anchor.generate_anchors()
    utils.box_plot(num_anchors)
    num_anchors =  num_anchors.shape[0]
    # with tf.Session() as sess:
    #     clas, reg = rpn.rpn_k(net, num_anchors)
    #     print (clas)

    classe, reg = rpn.rpn_net(net, num_anchors)
    

    # clar = np.array(clas)
    # for i in range(clar.shape[3]):
    #     plt.imshow(clar[1, :, :, i])
    #     plt.show

    # print (net.shape)
    # filt = np.array(net)
    # print (filt.shape[2])

    # filt = (clas)
    # print (filt.shape[3])

    # plt.figure(1, figsize=(20,20))
    # n_columns = 6
    # n_rows = math.ceil(filt.shape[3] / n_columns) + 1
    # for i in range(filt.shape[3]):
    #     plt.subplot(n_rows, n_columns, i+1)
    #     plt.subplot(3, 3, i+)
    #     # plt.title('Filter: {0} '.format(str(i)))
    #     plt.imshow(clas[:,:,i], interpolation="nearest")
    # plt.show()
    # for fil in range(net.shape[2]):
    #     extracted_filter = net[:, :, fil]
    #     plt.imshow(extracted_filter)
    #     plt.show()
        # img.append(fil,:,:)
    #     img.append(extracted_filter)

 
