import scipy
import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

class DataLoader():
    def __init__(self, dataset_name, features, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

        self.max_features = features
        vectorizer = CountVectorizer(stop_words='english', max_features=self.max_features, binary=True)
        descriptions = pd.read_csv('../fashiongen/descriptions.csv')
        self.vect_descriptions = vectorizer.fit_transform(descriptions['name'].values)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('../images/%s/*' % (data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        text_descriptions = []

        for img_path in batch_images:
            img = self.imread(img_path)
            index = re.findall('\d+', img_path)
            index = int(index[0]) - 1

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            text_descriptions.append(
                self.vect_descriptions[index].toarray().reshape(1, 1, self.max_features))

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        text_descriptions = np.array(text_descriptions)

        return imgs_A, imgs_B, text_descriptions

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('../images/%s/*' % (data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, text_descriptions = [], [], []
            for img in batch:
                index = re.findall('\d+', img)
                index = int(index[0]) - 1
                img = self.imread(img)

                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                text_descriptions.append(
                    self.vect_descriptions[index].toarray().reshape(1, 1, self.max_features))

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            text_descriptions = np.array(text_descriptions)

            yield imgs_A, imgs_B, text_descriptions


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)