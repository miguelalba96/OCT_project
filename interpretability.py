import os
import glob
import copy

import cv2
from tqdm import tqdm
import numpy as np
import skimage.segmentation
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import tensorflow as tf

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class LIME(object):
    def __init__(self, model, areas=20, perturbations=700, **kwargs):
        self.model = model
        self.perturbations = perturbations
        self.areas = areas
        self.kernel_width = 0.25

    def create_perturbations(self, img, i, segments, create_visualization=False):
        active_pixels = np.where(i == 1)[0]
        mask = np.zeros(segments.shape)
        for act in active_pixels:
            mask[segments == act] = 1
        perturbed_img = copy.deepcopy(img)
        if create_visualization:
            mask = mask.astype(np.float32)
            mask = mask[..., None]
            mask = np.concatenate((mask, mask, mask), axis=2)
            mask *= 0.2
            green_mark = np.ones(perturbed_img.shape, dtype=np.float32) * (0, 1, 0)
            perturbed_img = green_mark * mask + perturbed_img * (1.0 - mask)
        else:
            perturbed_img = perturbed_img * mask[:, :, np.newaxis]
        return perturbed_img

    def fit_linear_model(self, img, label):
        self.model.trainable = False
        img = standardize_sample(img)
        super_pixels = skimage.segmentation.quickshift(img, kernel_size=2, ratio=0.1, max_dist=1000)
        num_super_pixels = np.unique(super_pixels).shape[0]
        perturbations = np.random.binomial(1, 0.5, size=(self.perturbations, num_super_pixels))
        preds = []
        for pert in tqdm(perturbations):
            pert = self.create_perturbations(img, pert, super_pixels)
            predictions = self.model(tf.cast([pert], tf.float32))
            preds.append(predictions.numpy()[0])
        preds = np.array(preds)
        initial_image = np.ones(num_super_pixels)[np.newaxis, :]

        distances = sklearn.metrics.pairwise_distances(perturbations, initial_image, metric='cosine').ravel()
        weights = np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))

        y = preds[:, label]  # remove one hot
        linear_model = LinearRegression()
        linear_model.fit(X=perturbations, y=y, sample_weight=weights)
        coef = linear_model.coef_
        top_super_pixels = np.argsort(coef)[-self.areas:]
        mask = np.zeros(num_super_pixels)
        mask[top_super_pixels] = True
        explainer = self.create_perturbations(img, mask, super_pixels, create_visualization=True)
        return explainer, mask, super_pixels


def prep_eval_data(prep_fns):
    prep = []
    for fn in tqdm(prep_fns):
        img = utils.imread(fn)
        img = cv2.resize(img, (136, 136), interpolation=cv2.INTER_CUBIC)
        prep.append(img)
    return prep


def standardize_sample(img):
    mean = np.mean(img)
    n = len(img.ravel())
    adjusted_stddev = max(np.std(img), 1.0 / np.sqrt(n))
    return (img-mean)/adjusted_stddev


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    utils.setup_gpus()
    dme = glob.glob('/media/miguel/ALICIUM/Miguel/DOWNLOADS/ZhangLabData/CellData/OCT/test/DRUSEN/*')
    data = prep_eval_data(dme)
    img = data[0]
    modelname = '20201011_vanilla_cnn_batch64'
    model_path = os.path.join('./trained_models', modelname, 'frozen')
    model = tf.keras.models.load_model(model_path)
    explainer = LIME(model, areas=7, perturbations=700)
    ex, mask, super_pix = explainer.fit_linear_model(img, label=3)
    show(ex)
    show(skimage.segmentation.mark_boundaries(img, super_pix))
