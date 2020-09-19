import os
import glob
import math
import random

import skimage
import imgaug.augmenters as iaa

import numpy as np
import tensorflow as tf


def overexpose(image):
    image = skimage.exposure.equalize_hist(image.astype(np.float32))
    image = skimage.exposure.adjust_gamma(image, 0.3)
    return image


def blur_image(image, blur_sigma):
    return skimage.filters.gaussian(image, blur_sigma)
        

def external_augmentation(crop):
    crop = crop.numpy()
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([sometimes(
        iaa.Affine(scale=(1.0, 1.1),
                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                   rotate=(-8, 8),
                   shear=(-8, 8),
                   mode='edge'))])

    # crop = overexpose(crop) if random.random() < 0.2 else crop

    # if random.random() < 0.1:
        # blur_sigma = random.uniform(0, 1.5)
        # crop = blur_image(crop, blur_sigma)

    crop = seq.augment_image(crop)
    return crop


class DataLoader(object):
    def __init__(self, data_path, training):
        self.data_path = data_path
        self.training = 'train' if training else 'test'
        self.classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
        self.seed = 1
        if self.training == 'train':
            self.batch_size = 64
            self.buffer = 5000
        else:
            self.batch_size = 128
            self.buffer = 100

    def parse_record(self, record):
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        record = tf.io.parse_single_example(record, features)
        img = tf.io.decode_raw(record['image'], tf.float32)
        img = tf.reshape(img, [record['height'], record['width'], 3])
        label = tf.one_hot(record['label'], len(self.classes), dtype=tf.float32)
        return img, label

    def random_jitteing(self, crop):
        crop = tf.image.resize(crop, [136, 136], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        crop = tf.image.random_crop(crop, size=[136, 136, 3])
        return crop

    def agumentation(self, crop, label):
        crop = self.random_jitteing(crop)
        crop = tf.image.random_flip_left_right(crop)
        return crop, label

    def load_dataset(self, label):
        files = os.path.join(self.data_path, '{}_{}*.tfrecord'.format(self.training, label))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.training == 'train':
            dataset = dataset.map(self.agumentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.repeat()
        return dataset

    def balanced_batch(self):
        datasets = []
        for cl in self.classes:
            datasets.append(self.load_dataset(cl))
        importance = [0.25, 0.25, 0.25, 0.25]
        sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=importance)
        sampled_dataset = sampled_dataset.batch(self.batch_size)
        sampled_dataset = sampled_dataset.prefetch(1)
        return sampled_dataset

    def test_dataset(self, dataset_name=None):
        """
        :param dataset_name: take an specific dataset to evaluate
        :return: all test dataset
        """
        if dataset_name:
            files = os.path.join(self.data_path, 'test_*-{}_*.tfrecord'.format(dataset_name))
        else:
            files = os.path.join(self.data_path, 'test_*.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        return dataset


class Checkpoint:
    def __init__(self, checkpoint_kwargs, out_dir, max_to_keep=5, keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, out_dir, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)


def write_tensorboard(stats_dict, step, full_eval=False):
    name = 'Epoch metrics' if full_eval else 'Metrics'
    type = stats_dict['type']
    for scope, metric in stats_dict.items():
        if scope == 'loss':
            tf.summary.scalar('{}/Loss'.format(name), metric.numpy(), step)
        if scope == 'average_loss':
            tf.summary.scalar('{}/Average Loss'.format(name), metric.numpy(), step)
        if scope == 'accuracy':
            tf.summary.scalar('{}/Accuracy'.format(name), metric.numpy(), step)


def build_graph(model, feats, log_dir, step=0):
    @tf.function
    def tracing(feats):
        pred = model(feats)
        return pred

    writer = tf.summary.create_file_writer(os.path.join(log_dir, 'model_graph'))
    tf.summary.trace_on(graph=True)
    _ = tracing(feats)
    with writer.as_default():
        tf.summary.trace_export(name="graphs", step=step)
