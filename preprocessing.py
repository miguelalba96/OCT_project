import os
import glob
import utils

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


ALL_CLASSES = ['NORMAL', 'CNV', 'DME', 'DRUSEN']


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_label(label):
    if label == 'NORMAL':
        return 0
    if label == 'CNV':
        return 1
    if label == 'DME':
        return 2
    if label == 'DRUSEN':
        return 3


def _process_examples(example_data, filename: str, channels=3, pre_augm=True):
    """
    :param example_data: takes the list of dictionaries and transform them into Tf records, this is an special format
    of tensorflow data that makes your life easier in tf 1.x and 2.0 saving the data and load it in our training loop
    (WARNING: You have to take care of the encoding of features to not have problems when loading the data, this means
    taking into consideration that images are int or float)
    :param filename: output filename
    :param channels: number of channels of the image (RGB=3), grayscale=!
    :return: None
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for i, ex in enumerate(example_data):
            # define pre augmentation of pre image resizing
            if pre_augm:
                image = pre_augmentation(ex['image']).tostring()
            else:
                image = ex['image'].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(ex['image'].shape[0]),
                'width': _int64_feature(ex['image'].shape[1]),
                'depth': _int64_feature(channels),
                'image': _bytes_feature(image),
                'label': _int64_feature(encode_label(ex['label']))
            }))
            writer.write(example.SerializeToString())
    return None


def resize_image(img, size=(128, 128)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)
    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, size, interpolation)


def pre_augmentation(img):
    img = utils.standardize_img(img)
    return img


def shard_dataset(dataset, num_records=50):
    chunk = len(dataset) // num_records
    parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
    return chunk, parts


class Preprocessing(object):
    def __init__(self, data_path):
        self.data_path = data_path
        utils.mdir(os.path.join(data_path, 'preprocessing'))
        
    def load_images(self, filename_shard, type, label):
        data = []
        for fn in filename_shard:
            img = utils.imread(fn)
            meta = {
                'image': resize_image(img, size=(136, 136)),
                'filename': fn,
                'label': label,
                'dataset': type
            }
            data.append(meta)
        return data

    def write_data(self, filenames, type, label):
        chunk, parts = shard_dataset(filenames)
        for i, j in enumerate(tqdm(parts)):
            shard = filenames[j:(j + chunk)]
            shard_data = self.load_images(shard, type, label)
            fn = '{}_{}-{}_{:03d}-{:03d}.tfrecord'.format(type, label, 'OCT', i + 1, len(parts))
            _process_examples(shard_data, os.path.join(self.data_path, 'preprocessing', fn))
        return None

    def create_data(self, type='train'):
        data_path = os.path.join(self.data_path, type)
        for cl in ALL_CLASSES:
            sub_dir = os.path.join(data_path, cl)
            data_fns = glob.glob('{}/*'.format(sub_dir))
            self.write_data(data_fns, type, label=cl)

    def generate_example_sets(self):
        for type in ['train', 'test']:
            self.create_data(type)


if __name__ == '__main__':
    prep = Preprocessing(data_path='/media/miguel/ALICIUM/Miguel/DOWNLOADS/ZhangLabData/CellData/OCT')
    prep.generate_example_sets()