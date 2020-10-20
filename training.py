import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils
from net_tools import DataLoader, Checkpoint, write_tensorboard, build_graph
from conv_nets import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class OCTtraining(object):
    def __init__(self, modelname, data_path, architecture, hyperparams, img_size=[136, 136], **kwargs):
        """
        :param modelname: model name
        :param data_path: folder of the data records
        :param model: tensorflow model
        :param hyperparams: params
        :param kwargs:
        """
        # initialize GPUs
        utils.setup_gpus()
        self.modelname = modelname
        self.model_path = os.path.join('./trained_models', modelname)
        self.data_path = data_path
        # hyper parameter
        self.params = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'schedule': False,  # adaptive learning rate
            'optimizer': 'ADAM',  # SGD, SGDM
            'test_iter': 100,
            'step_size': 2000,
            'epochs': 50,
            'max_class_samples': 51140,  # the second time I see again a NORMAL sample
            'total_num_samples': 108309
        }
        self.params.update(hyperparams)
        self.img_size = img_size
        self.log_dir, self.ckpt_dir, self.train_writer, self.test_writer = self.create_dirs()

        # steps per epoch
        self.steps_epoch = np.ceil(2.0 * self.params['max_class_samples'] / self.params['batch_size'])
        # self.steps_epoch = np.ceil(self.params['total_num_samples'] / self.params['batch_size'])
        self.epochs = self.params['epochs']
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

        self.architecure_params = dict(**kwargs)
        self.model = self.build_model(architecture, **self.architecure_params)

        self.train_data = DataLoader(self.data_path, training=True)
        self.test_data = DataLoader(self.data_path, training=False)

        self.lr, self.opt = self.optimizer()
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.sample_weights = tf.constant([0.25, 0.25, 0.25, 0.25])

        self.train_loss, self.test_loss, self.train_acc, self.test_acc = self.build_metrics()

        architecture = dict(model=self.model,
                            optimizer=self.opt,
                            current_epoch=self.epoch_counter,
                            step=self.step)
        self.ckpt = Checkpoint(architecture, self.ckpt_dir, max_to_keep=3)
        try:
            self.ckpt.restore().assert_existing_objects_matched()
            print('Loading pre trained model')
        except Exception as e:
            print(e)

    def create_dirs(self):
        log_dir = os.path.join(self.model_path, 'logs')
        ckpt_dir = os.path.join(self.model_path, 'checkpoints')
        train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'opt/train'))
        test_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'opt/test'))
        utils.mdir(log_dir)
        utils.mdir(ckpt_dir)
        utils.mdir(os.path.join(self.model_path, 'weights'))
        return log_dir, ckpt_dir, train_writer, test_writer

    def build_model(self, architecture, **params):
        input_shape = [self.img_size[0], self.img_size[1], 3]
        model = getattr(models, str(architecture))(**params)
        try:
            print(model.summary())
        except ValueError:
            inputs = tf.keras.Input(shape=tuple(input_shape), name='input_img')
            model(inputs)
            print('== Model description ==')
            print(model.summary())
        return model

    def optimizer(self):
        if self.params['schedule']:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.params['learning_rate'],
                                                                decay_steps=self.params['step_size'],
                                                                decay_rate=0.5, staircase=True)
        else:
            lr = self.params['learning_rate']
        if self.params['optimizer'] == 'SGDM':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif self.params['optimizer'] == 'ADAM':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise NotImplementedError
        return lr, opt

    def build_metrics(self):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
        return train_loss, test_loss, train_acc, test_acc

    def cat_cross_entropy(self, data, labels, training):
        predictions = self.model(data, training)
        obj_loss = self.loss(y_true=labels, y_pred=predictions)
        return obj_loss

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)
            loss = self.cat_cross_entropy(data, labels, training=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))  # update
        _loss = self.train_loss(loss)
        _acc = self.train_acc(labels, predictions)
        summary = {'type': 'train', 'loss': loss, 'average_loss': _loss, 'accuracy': _acc}
        return summary

    @tf.function
    def test_step(self, data, labels):
        predictions = self.model(data, training=False)
        loss = self.cat_cross_entropy(data, labels, training=False)
        _loss = self.test_loss(loss)
        _acc = self.test_acc(labels, predictions)
        summary = {'type': 'test', 'loss': loss, 'average_loss': _loss, 'accuracy': _acc}
        return summary

    def complete_evaluation(self):
        for _test in self.test_data.test_dataset():
            test_summary = self.test_step(_test[0], _test[1])
        with self.test_writer.as_default():
            write_tensorboard(test_summary, step=self.step, full_eval=True)
            self.test_loss.reset_states()
            self.test_acc.reset_states()

    def train(self):
        print('Starting Training')
        train = self.train_data.balanced_batch()
        test = self.test_data.balanced_batch()
        data = tf.data.Dataset.zip((train, test))

        for epoch in range(int(self.epoch_counter), int(self.epochs)):
            self.epoch_counter.assign_add(1)
            step_bar = tqdm(total=self.steps_epoch, desc='Steps', position=1)
            for train_batch, test_batch in data:
                img, labels = train_batch
                test_img, test_labels = test_batch
                train_summary = self.train_step(img, labels)
                test_summary = self.test_step(test_img, test_labels)
                if int(self.step) == 0:
                    build_graph(self.model, img, self.log_dir, self.step)
                (train_loss, train_acc) = self.train_loss.result(), self.train_acc.result()
                (test_loss, test_acc) = self.test_loss.result(), self.test_acc.result()
                lr = self.lr(self.step).numpy() if self.params['schedule'] else self.lr
                if int(self.step % self.params['test_iter']) == 0:
                    with self.train_writer.as_default():
                        write_tensorboard(train_summary, step=self.step)
                        tf.summary.scalar('Metrics/Learning_rate', lr, step=self.step)
                        self.train_loss.reset_states()
                        self.train_acc.reset_states()
                    with self.test_writer.as_default():
                        write_tensorboard(test_summary, step=self.step)
                        self.test_loss.reset_states()
                        self.test_acc.reset_states()

                self.step.assign_add(1)
                step_bar.update(1)

                if int(self.step % self.steps_epoch) == 0:
                    with self.train_writer.as_default():
                        write_tensorboard(train_summary, step=self.step, full_eval=True)
                    break

            self.complete_evaluation()

            template = '{}: train loss: {}, test loss: {}, train acc: {}, test acc: {}'
            print(template.format(int(epoch), train_loss, test_summary['loss'], train_acc, test_summary['accuracy']))

            self.train_writer.flush()
            self.test_writer.flush()

            self.ckpt.save(epoch)
            # # TODO: fix serialization with low level API
            # self.model.save_weights(os.path.join(self.model_path, 'weights', 'pretrained'),
            #                         overwrite=True, save_format='tf')
            self.model.save(os.path.join(self.model_path, 'frozen'))
            self.model.save(os.path.join(self.model_path, 'model.h5'))
            if int(self.step % (self.epochs * self.steps_epoch)) == 0:
                break
        print('Finished Training')
        return None


def _20200915_first_model():
    modelname = '20201011_vanilla_cnn_batch64'
    data_path = '/media/miguel/ALICIUM/Miguel/DOWNLOADS/ZhangLabData/CellData/OCT/preprocessing'
    model = 'sequential_model_1'
    cnn = OCTtraining(modelname, data_path, model,
                      hyperparams=dict(learning_rate=0.01, epochs=100,
                                       optimizer='SGDM',
                                       schedule=True,
                                       step_size=5000),
                      crop_size=[136, 136])
    cnn.train()


def _20200923_dense_model():
    modelname = '202017_dense_net_batch64_cleaner_data'
    data_path = '/media/miguel/ALICIUM/Miguel/DOWNLOADS/ZhangLabData/CellData/OCT/preprocessing'
    model = 'dense_net_red'
    cnn = OCTtraining(modelname, data_path, model,
                      hyperparams=dict(learning_rate=0.02, epochs=100,
                                       optimizer='SGDM',
                                       schedule=True,
                                       step_size=5000),
                      crop_size=[136, 136])
    cnn.train()


def _20200929_dense_net():
    modelname = '20200929_primer_modelo_squeeze_densenet_batch64'
    data_path = '/media/miguel/ALICIUM/Miguel/DOWNLOADS/ZhangLabData/CellData/OCT/preprocessing'
    model = 'dense_net_red'
    cnn = OCTtraining(modelname, data_path, model,
                      hyperparams=dict(learning_rate=0.02, epochs=60,
                                       optimizer='SGDM',
                                       schedule=True,
                                       step_size=5000),
                      crop_size=[136, 136])
    cnn.train()


if __name__ == '__main__':
    # _20200915_first_model()
    _20200923_dense_model()
    # _20200929_dense_net()
    pass
