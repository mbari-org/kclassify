#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Plot utility for generating performance plots

@author: __author__
@status: __status__
@license: __license__
'''

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import preprocessing
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

class Plot():

    @staticmethod
    def plot_loss_graph(history, title):
        """
        Generate a matplotlib graph for the loss and accuracy metrics
        :param args:
        :param history: dictionary of performance data
        :return: instance of a graph
        """
        acc = history.history['categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, loss, 'bo')

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        return fig

    @staticmethod
    def plot_accuracy_graph(history, title):
        plt.clf()
        acc = history.history['categorical_accuracy']
        val_acc = history.history['val_categorical_accuracy'] 

        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, acc, 'bo')

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        return fig

    @staticmethod
    def plot_roc(labels, y_test, y_prob):
        plt.clf()
        fig, ax = plt.subplots()
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.arange(0, y_prob.shape[1] + 1))
        y_test_bin = lb.transform(y_test)
        fpr, tpr, _ = roc_curve(y_test_bin[:, np.arange(0, y_prob.shape[1])].ravel(), y_prob.ravel())
        auc_final = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{labels}, AUC={auc_final:.3f}")

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve', fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 13}, loc='lower right')
        return fig