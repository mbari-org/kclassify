#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Saves and restores best training weights on training stop

@author: __author__
@status: __status__
@license: __license__
'''

import tensorflow.keras
import numpy as np

class Stopping(tensorflow.keras.callbacks.Callback):

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):

        self._monitor = monitor
        self._patience = patience
        self._verbose = verbose
        self._baseline = baseline
        self._min_delta = abs(min_delta)
        self._wait = 0
        self._stopped_epoch = 0
        self._best_epoch = 0
        self._restore_best_weights = restore_best_weights
        self._best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self._monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self._min_delta *= 1
        else:
            self._min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self._wait = 0
        self._stopped_epoch = 0
        if self._baseline is not None:
            self.best = self._baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self._min_delta, self.best):
            self.best = current
            self._best_epoch = epoch
            self._wait = 0
            if self._restore_best_weights:
                self._best_weights = self.model.get_weights()
        else:
            self._wait += 1
            print(f'{self._wait} epochs since improvement to {self._monitor}')
            if self._wait >= self._patience:
                self._stopped_epoch = epoch
                print('Model training state previously: {}'.format(self.model.stop_training))
                self.model.stop_training = True
                print('Model training state now: {}'.format(self.model.stop_training))
                if self._restore_best_weights:
                    if self._verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self._best_weights)

    def on_train_end(self, logs=None):
        if self._stopped_epoch > 0 and self._verbose > 0:
            print('Epoch %05d: early stopping' % (self._stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self._monitor)
        return monitor_value

    @property
    def best_epoch(self):
        return self._best_epoch
