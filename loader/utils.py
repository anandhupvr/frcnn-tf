import os
import glob
import sys

class Bbox:
    def __init__(self, x, y, w, h, cat):
        self.x = x
        self.x = x
        self.w = w
        self.h = h
        self.cat = cat

    def __getitem__(self,key):
        return getattr(self, key)

def train_test_split(data_path, *args):
    if len(args) == 0:
        percentage_test = 10
    else:
        percentage_test = args[0]
    img_count = len(os.listdir(data_path+"images/apple"))
    file_train = open(data_path + 'train.txt', 'w+')  
    file_test = open(data_path + 'test.txt', 'w+')
    counter = 0
    index_test = round(img_count * percentage_test / 100)
    all_items = glob.glob(data_path+"images/apple/*.jpg")
    for i in all_items:
        if counter < index_test:
            file_test.write(i+"\n")
            counter += 1
        else:
            file_train.write(i+"\n")