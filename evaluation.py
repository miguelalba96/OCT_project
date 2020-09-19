import os
import pprint
import argparse
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import utils
from net_tools import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--model-name', help="Name of the CNN model trained")
parser.add_argument('--data-path', help="Folder containing the tf records")
parser.add_argument('--save-preds', action='store_true', default=False, help='Save predictions')
parser.add_argument('--explain', action='store_true', default=False, help='Explanability methods ')

args = parser.parse_args()


class EvalDataset(object):
    def __init__(self, model_name, data_path, explain=False, save_predictions=False, **kwargs):
        utils.setup_gpus()

        self.model_path = os.path.join('./trained_models', model_name, 'frozen')

        print('Loading model from: {}'.format(self.model_path))

        self.model_name = model_name
        self.data = DataLoader(data_path, training=False).test_dataset()
        self.model = tf.keras.models.load_model(self.model_path)
        self.explain = explain
        self.outdir = os.path.join('./trained_models', model_name, 'results')
        self.class_names = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
        self.save_predictions = save_predictions
        utils.mdir(self.outdir)

    def compute_metrics(self, predictions):
        results = dict(model_name=self.model_name, report=dict())

        all_labels = [ex['label'] for ex in predictions]
        all_preds = [ex['pred_class'] for ex in predictions]
        base_conf_matrix = confusion_matrix(all_labels, all_preds)
        pprint.pprint({'baseline_conf_matrix': base_conf_matrix.tolist()})

        for t in [0.5, 0.6, 0.7, 0.85, 0.9, 0.95]:
            filtered = []
            for ex in predictions:
                probabilities = ex['probabilities']
                predicted_as = ex['pred_class']
                if probabilities[predicted_as] >= t:
                    filtered.append(ex)

            labels = [ex['label'] for ex in filtered]
            preds = [ex['pred_class'] for ex in filtered]
            conf_matrix = confusion_matrix(labels, preds)

            meta = {
                'threshold_{}'.format(t): {
                    'classes': self.class_names,
                    'confusion_matrix': conf_matrix.tolist(),
                    'class_report': classification_report(labels,
                                                          preds, target_names=self.class_names,
                                                          output_dict=True)
                }
            }
            results['report'].update(meta)
        pprint.pprint(results)
        return results

    def evaluate(self):
        predictions = []
        for batch in tqdm(self.data):
            imgs, labels = batch
            preds = self.model(imgs, training=False)
            pred_as = tf.argmax(preds, axis=1).numpy()
            labels = tf.argmax(labels, axis=1).numpy()
            for i, ex in enumerate(imgs):
                meta = {
                    'img': imgs[i][:, :, 0].numpy(),
                    'probabilities': preds[i].numpy(),
                    'pred_class': int(pred_as[i]),
                    'label': int(labels[i])
                }
                if self.explain:
                    self.explanations()

                predictions.append(meta)
        results = self.compute_metrics(predictions)
        # TODO: add a saving in tf record for filename, to be used to check the direct images
        utils.save_json(os.path.join(self.outdir, 'results.json'), results)
        if self.save_predictions:
            utils.save(os.path.join(self.outdir, 'imgs.pdata'), predictions)
        return predictions

    def explanations(self):
        raise NotImplementedError


if __name__ == '__main__':
    evaluation = EvalDataset(args.model_name, args.data_path, args.explain, args.save_preds)
    evaluation.evaluate()
