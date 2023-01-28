#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

TensorFlow Keras classifier

@author: __author__
@status: __status__
@license: __license__
'''

import signal
import os, sys, inspect

import sklearn.metrics

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(f'Adding {parentdir} to path')
import tempfile
import tarfile
import json
import codecs
import numpy as np
import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint
from transfer_model import TransferModel
from wandb.keras import WandbCallback
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from stopping import Stopping
from argparser import ArgParser
import plot
import pandas as pd
import time
import shutil
from focal_loss import focal_loss
from pathlib import Path
import conf as model_conf
from numpy import  asarray


def sigterm_handler(signal, frame):
    print('Run got SIGTERM')


class TrainOutput:

    def __init__(self, model, image_size, classes, class_size, history, image_mean,
                 image_std, best_epoch, y_test, y_pred):
        self._model = model
        self._image_size = image_size
        self._classes = classes
        self._class_size = class_size
        self._history = history
        self._image_mean = image_mean
        self._image_std = image_std
        self._best_epoch = best_epoch
        self._y_test = y_test
        self._y_pred = y_pred

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_mean(self):
        return self._image_mean

    @property
    def image_std(self):
        return self._image_std

    @property
    def classes(self):
        return self._classes

    @property
    def y_test(self):
        return self._y_test

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def history(self):
        return self._history


class Train:

    def __init__(self, out_path: Path, ckpt_path: Path):
        self._output_path = out_path
        self._checkpoint_path = ckpt_path

    def compile_and_fit_model(self, model, fine_tune_num, train_generator, validation_generator,
                              epochs, batch_size, loss, optimizer, lr, k,
                              metric_type=tf.keras.metrics.categorical_accuracy,
                              early_stop=False, has_wandb=False):

        steps_per_epoch = train_generator.n // batch_size
        validation_steps = validation_generator.n // batch_size

        # un-freeze the top layers of the model
        model.trainable = True

        # if fine tune at defined, unfreeze top `fine_tune_num` layers
        if fine_tune_num > 0:
            for layer in model.layers[-1*fine_tune_num:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

        opt_loss = loss
        if loss == 'categorical_focal_loss':
            opt_loss = focal_loss(alpha=1)

        if optimizer == 'radam':
            model.compile(loss=opt_loss,
                          optimizer=tfa.optimizers.RectifiedAdam(
                                                                learning_rate=lr,
                                                                total_steps=epochs,
                                                                warmup_proportion=0.1,
                                                                min_lr=1e-5,
                                                            ),
                          metrics=[metric_type],
                          run_eagerly=True)

        elif optimizer == 'ranger':
            opt = tfa.optimizers.Lookahead(RAdam(lr=lr), k, slow_step_size=0.5)
            model.compile(loss=opt_loss,
                          optimizer=opt,
                          metrics=[metric_type])

        elif optimizer == 'adam':
            model.compile(loss=opt_loss,
                          optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          metrics=[metric_type])
        else:
            model.compile(loss=opt_loss,
                          optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                          metrics=[metric_type])

        early = Stopping(monitor='val_categorical_accuracy', patience=3, verbose=1, restore_best_weights=True)
        checkpoint_path = self._checkpoint_path / 'checkpoints.best.h5'
        checkpoint = ModelCheckpoint(checkpoint_path.as_posix(), monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     mode='max')

        class PrintMetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epochs, logs=None):
                print(f"val_acc = {logs['categorical_accuracy']:.10f}")
                print(f"val_loss = {logs['loss']:.10f}")
                if has_wandb:
                    wandb.log({'val_acc': logs['categorical_accuracy']}, step=epochs, commit=False)
                    wandb.log({'val_loss': logs['loss']}, step=epochs, commit=False)

        callbacks = [PrintMetricsCallback(), checkpoint]
        if early_stop:
            callbacks += [early]

        if has_wandb:
            wandb_cb = WandbCallback(save_model=False, data_type="image", validation_data=validation_generator,
                                  labels=validation_generator.classes)
            callbacks += [wandb_cb]

        if checkpoint_path.exists():
            print(f'Loading model weights from {checkpoint_path}')
            model.load_weights(checkpoint_path.as_posix())

        history = model.fit(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            callbacks=callbacks)
        if early_stop:
            best_epoch = early.best_epoch
        else:
            best_epoch = min(0, len(history.history) - 1)
        return history, model, best_epoch

    def print_metrics(self, hist):
        print('===========Final metrics==========')
        val_acc = hist.history['val_categorical_accuracy'][-1]
        val_loss = hist.history['val_loss'][-1]
        print(f'validation_loss = {val_loss:6.4f}')
        print(f'validation_accuracy = {val_acc:6.4f}')

    def train_model(self, args: list):
        """
        Train the model and print out the metrics
        :param args command line argument object
        """
        # if the base model is not defined, use the first one in the list
        if not args.base_model:
            args.base_model = list(model_conf.MODEL_DICT.keys())[0]

        cfg = eval(f"model_conf.MODEL_DICT['{args.base_model}']")

        if args.has_wandb:
            wandb.sagemaker_auth(path=Path(__file__).resolve().parent)
            wandb.login()
            wandb.init(config=args)

        with open(args.train_stats) as f:
            conf_dict = json.load(f)

        classes = sorted(list(conf_dict['total_concepts'].keys())) # in ascending sorted order as presented by the generator
        print(f'Training classes {classes}')
        class_size = len(classes)
        mean = np.array([])
        std = np.array([])

        if args.preprocessor:
            # Preprocess using the same method as the base model and apply image augmentation if requested
            generator_args = dict(preprocessing_function=eval(cfg['preprocessor']),
                                  rotation_range=args.rotation_range,
                                  width_shift_range=args.augment_range,
                                  height_shift_range=args.augment_range,
                                  zoom_range=args.augment_range)
        elif args.featurewise_normalize:
            # Rescale all images by 1./255, normalize and apply image augmentation if requested
            generator_args = dict(rescale=1/255.,
                                  rotation_range=args.rotation_range,
                                  width_shift_range=args.augment_range,
                                  height_shift_range=args.augment_range,
                                  zoom_range=args.augment_range,
                                  featurewise_center=True,
                                  featurewise_std_normalization=True)
            if 'mean' not in conf_dict.keys() or 'std' not in conf_dict.keys() :
                raise Exception(f'Need to define the mean/std for featurewise_normalize in a config.json')
        else:
            generator_args = dict(preprocessing_function=eval(cfg['preprocessor']),
                                  rotation_range=args.rotation_range,
                                  width_shift_range=args.augment_range,
                                  height_shift_range=args.augment_range,
                                  zoom_range=args.augment_range) 

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(generator_args)

        if args.featurewise_normalize:
            mean = asarray(conf_dict['mean'], dtype=np.float32)  # [R, G, B] order
            std = asarray(conf_dict['std'], dtype=np.float32)
            data_gen.mean = mean
            data_gen.std = std

        train_path = self._output_path / 'train'
        val_path = self._output_path / 'val'
        image_dir = self._output_path / 'images'

        image_dir.mkdir(parents=True)

        def unpack(target_path: Path, tar_path: Path):
            print(f'Unpacking {tar_path}')
            if not target_path.exists():
                target_path.mkdir(parents=True)
            tar = tarfile.open(tar_path)
            tar.extractall(path=target_path)
            tar.close()

        try:
            unpack(train_path, args.train)
            unpack(val_path, args.eval)
        except Exception as ex:
            raise (ex)

        model, image_size = TransferModel(args.base_model, args.dropout, args.l2_weight_decay_alpha).build(class_size)

        # train/val flow from separate directories
        print('Training data:')
        training_generator = data_gen.flow_from_directory(train_path.as_posix(),
                                                               target_size=(image_size, image_size),
                                                               batch_size=args.batch_size,
                                                               class_mode='categorical',
                                                               color_mode='rgb')

        print('Validation data:')
        validation_generator = data_gen.flow_from_directory(val_path.as_posix(),
                                                                 target_size=(image_size, image_size),
                                                                 batch_size=args.batch_size,
                                                                 class_mode='categorical',
                                                                 color_mode='rgb',
                                                                 shuffle=False)
 
        batch_x, batch_y = training_generator.next()
        print(f'Training batch shape={batch_x.shape}, min={batch_x.min():0.3f}, max={batch_x.max():0.3f}')

        history, model, best_epoch = self.compile_and_fit_model(model=model, fine_tune_num=-1,
                                                                train_generator=training_generator, lr=args.lr,
                                                                k=args.k,
                                                                validation_generator=validation_generator,
                                                                epochs=args.epochs,
                                                                batch_size=args.batch_size,
                                                                loss=args.loss,
                                                                optimizer=args.optimizer,
                                                                metric_type=tf.keras.metrics.categorical_accuracy,
                                                                early_stop=args.early_stop,
                                                                has_wandb=args.has_wandb)

        if args.fine_tune_num > 0:
            history, model, best_epoch = self.compile_and_fit_model(model=model, fine_tune_num=args.fine_tune_num,
                                                                train_generator=training_generator, lr=1e-4,
                                                                k=args.k,
                                                                validation_generator=validation_generator,
                                                                epochs=args.epochs,
                                                                batch_size=args.batch_size,
                                                                loss=args.loss,
                                                                optimizer=args.optimizer,
                                                                metric_type=tf.keras.metrics.categorical_accuracy,
                                                                early_stop=args.early_stop,
                                                                has_wandb=args.has_wandb)

        # load model for best epoch and compute data for PR/CM
        checkpoint_path = self._output_path / 'checkpoints.best.h5'
        if checkpoint_path.exists():
            print(f'Loading model weights from {checkpoint_path}')
            model.load_weights(checkpoint_path)

        print('Running prediction on validation data...')
        validation_generator.reset()
        pred = model.predict_generator(validation_generator)
        y_pred = np.argmax(pred, axis=1)

        print('===========Report==========')
        print(confusion_matrix(validation_generator.classes, y_pred))
        report = classification_report(validation_generator.classes, y_pred, target_names=classes, output_dict=True)
        print(report)
        df = pd.DataFrame(report).transpose()
        auc = sklearn.metrics.roc_auc_score(validation_generator.classes, y_pred, labels=classes)
        f1 = sklearn.metrics.f1_score(validation_generator.classes, y_pred, labels=classes)
        print(f'AUC = {auc:.10f}')
        print(f'F1 = {f1:.10f}')

        if args.has_wandb:
            wandb.log({'AUC':auc}, commit=False)
            wandb.log({'F1':f1}, commit=False)
        self.print_metrics(history)

        print(f'Saving model to {args.saved_model_dir}')
        model.save(args.saved_model_dir)
        df.to_csv(f'{args.saved_model_dir}/classification_report.csv')
        return TrainOutput(model, image_size, classes, class_size, history, mean, std, best_epoch, validation_generator.classes, pred)


def log_metrics(output: TrainOutput, image_dir: Path, has_wandb: bool):
    p = plot.Plot()

    # create plot of the loss/accuracy for quick reference
    graph_image_loss_png = image_dir / 'loss.png'
    graph_image_acc_png = image_dir / 'accuracy.png'
    graph_image_roc_png = image_dir / 'roc.png'
    figure_loss = p.plot_loss_graph(output.history, 'Training and Validation Loss')
    figure_loss.savefig(graph_image_loss_png)
    figure_acc = p.plot_accuracy_graph(output.history, 'Training and Validation Accuracy')
    figure_acc.savefig(graph_image_acc_png)

    acc = output.history.history['val_categorical_accuracy']
    print(f'best_val_categorical_accuracy = {acc[output.best_epoch]}')
    if has_wandb:
        wandb.log({"best_val_categorical_accuracy": acc[output.best_epoch]})
        wandb.log({'roc': wandb.plots.ROC(output.y_test, output.y_pred, output.classes)})
        wandb.log({'pr': wandb.plots.precision_recall(output.y_test, output.y_pred, output.classes)})
        wandb.sklearn.plot_confusion_matrix(output.y_test, np.argmax(output.y_pred, axis=1), output.classes)
    figure_roc = p.plot_roc(output.classes, output.y_test, output.y_pred)
    figure_roc.savefig(graph_image_roc_png)


def main(args=None):

    params_path = Path('/opt/ml/input/config/hyperparameters.json')

    # parse arguments either in sagemaker which happens through a hyperparameters.json file, or directly if testing
    if os.path.exists(params_path):
        args = []
        with open(params_path, 'r') as tc:
            hyperparameters = json.load(tc)
            for key, value in hyperparameters.items():
                args.append('--{}'.format(key))
                args.append(value)
            parser = ArgParser(args)
    else:
        parser = ArgParser(args)

    # clean previous model if one exists
    if os.path.exists(parser.args.saved_model_dir):
        if os.listdir(parser.args.saved_model_dir):
            print(f'{parser.args.saved_model_dir} not empty. Removing...')
            shutil.rmtree(parser.args.saved_model_dir)

    try:
        start_time = time.time()

        with tf.compat.v1.Session():
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                ckpt_path = Path(parser.args.checkpoint_local_path)
                model_path = Path(parser.args.saved_model_dir)
                
                # clean-up any existing checkpoints and models
                if parser.args.disable_save:
                    if ckpt_path.exists():
                        shutil.rmtree(ckpt_path)
                    if model_path.exists():
                        shutil.rmtree(model_path)

                if not ckpt_path.exists():
                    ckpt_path.mkdir(parents=True)

                if not model_path.exists():
                    model_path.mkdir(parents=True)

                runner = Train(output_path, ckpt_path)
                train_output = runner.train_model(parser.args)

                log_metrics(train_output, model_path, parser.args.has_wandb)

                # clean-up any existing checkpoints and models
                if parser.args.disable_save:
                    if ckpt_path.exists():
                        shutil.rmtree(ckpt_path)
                    if model_path.exists():
                        shutil.rmtree(model_path)
                else:
                    # log model and normalization parameters needed for inference
                    json.dump({'image_size': f'{train_output.image_size}x{train_output.image_size}',
                           "image_mean": train_output.image_mean.tolist(),
                           "image_std": train_output.image_std.tolist(),
                           "classes": train_output.classes},
                              codecs.open(model_path / 'config.json', 'w', encoding='utf-8'),
                              separators=(',', ':'), sort_keys=True,
                              indent=4)

                
    except Exception as ex:
        print('Model training failed ' + str(ex))
        exit(-1)

    runtime = time.time() - start_time
    print(f'Model complete. Total runtime {runtime}')
    exit(0)


def sigterm_handler(sig, frame):
    print(f'===================> Sigterm {sig} {frame}')
    exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        print(f' ==============Running {sys.argv}=============')
        main()
    except KeyboardInterrupt:
        print(' ==============KeyboardInterrupt=============')
    except Exception as ex:
        print(f' ================Exception {ex} ===============')
        exit(-1)


