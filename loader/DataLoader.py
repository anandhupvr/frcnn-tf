from loader.utils import Bbox
import loader.utils as utils
import os



class load:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = ('bg','apple')
        utils.train_test_split(self.dataset_path)

    def data(self):
        # import pdb; pdb.set_trace()
        all_imgs = {}
        box = []
        ptr = 0
        image_files = open(os.path.join(
            self.dataset_path,
            "train.txt"), "r").readlines()[ptr: ptr + 1]
        label_file = open(
            (image_files[0].split("images")[0] +
             "labels" +
             image_files[0].split("images")[1].replace(
                "jpg",
                "txt")).strip("\n")).readlines()
        print(len(label_file) - 1)
        class_name = label_file[0]
        for label in label_file[1:]:
            x, y, w, h = label.split(" ")
            obj = Bbox(x, y, w, h, class_name)
            box.append(obj)

        all_imgs[ptr] = box
        return all_imgs



