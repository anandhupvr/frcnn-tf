from loader.utils import Bbox
import loader.utils as utils
import os
from PIL import Image
import numpy as np


class load:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = ('meatballs', 'meatballs_dish')
        self._class_to_ind = dict(list(zip(self.classes, list(range(len(self.classes))))))
        self.imdb = {}
        utils.train_test_split(self.dataset_path+"images/meatballs")
        self.ptr = 0

    def _get_width(self, img):
        return(Image.open(img).size[0])

    def data_batch(self):

        def to_float(box):
            st = box.split(' ')
            return [float(i) for i in st]

        import pdb; pdb.set_trace()
        image_files = open("train.txt", "r").readlines()[self.ptr: self.ptr + 1]
        img = np.expand_dims(np.array(Image.open(image_files[0].strip()),dtype=np.uint8), axis=0).astype('float32')

        label_file = open(((image_files[0].strip()).split("images")[0] +
             "labelsbbox" + (image_files[0].strip()).split("images")[1].replace(
                "jpg","txt")).strip("\n")).readlines()

        # boxes = roidb[i]['boxes'].copy()

        labels = np.array([[float(i)] for i in label_file if len(i) < 3])
        bb = np.array([to_float(i.strip()) for i in label_file if len(i) > 3], dtype=np.float32)
        # box = bb.append(labels)
        self.ptr += 1
        feed = {0 : (img, bb, labels)}
        return feed

