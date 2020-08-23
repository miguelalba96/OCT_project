import gzip
import pickle
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def slurpjson(fn):
    import json
    with open(fn, 'r') as f:
        return json.load(f)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def mdir(path):
    try:
        os.makedirs(path)
        # print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")


def save(fn, a):
    with gzip.open(fn, 'wb', compresslevel=2) as f:
        pickle.dump(a, f, 2)


def imread(fn, gray_scale=False):
    if gray_scale:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fn)
    return img


def writeimg(fn, img, aspect_ratio=1.0):
    h, w, _ = img.shape
    img = Image.fromarray(img)
    # if aspect_ratio > 1:
    #     img = img.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # elif aspect_ratio < 1:
    #     img = img.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # else:
    img.save(fn)


def standardize_img(a, axis=(0, 1)):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


def load(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def save_json(fn, data):
    import json
    with open(fn, 'wb') as outfile:
        outfile.write(json.dumps(data).encode("utf-8"))


def get_all_folder(path):
    return glob.glob('{}/*'.format(path))


def setup_gpus():
    devises = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devises[0], True)


# def run_dill_encoded(payload):
#     fun, args = dill.loads(payload)
#     return fun(*args)


# def display_image(image):
#     import matplotlib.pyplot as plt
#     from GANs.net_utils import inverse_transform
#     image = inverse_transform(image.numpy()[0])
#     image = np.clip(image, 0, 255)
#     plt.imshow(image)
#     plt.show()
