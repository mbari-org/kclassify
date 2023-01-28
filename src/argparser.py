#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Argument parser

@author: __author__
@status: __status__
@license: __license__
'''

import os
import argparse
import sys
import conf as model_conf
from argparse import RawTextHelpFormatter
from pathlib import Path

# If running in AWS, we must define the inputs/outputs per the spec
if 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI' in os.environ:
    default_train = Path('/opt/ml/input/data/training/catsdogstrain.tar.gz')
    default_eval = Path('/opt/ml/input/data/training/catsdogsval.tar.gz')
    default_stats = Path('/opt/ml/input/data/training/train_stats.json')
    default_saved_model_dir = Path('/opt/ml/model')
    default_model_dir = Path('/opt/ml/model')
    default_checkpoint_path = Path('/opt/ml/checkpoints')
else:
    default_train = Path(__file__).parent.parent / 'data' / 'catsdogstrain.tar.gz'
    default_eval = Path(__file__).parent.parent / 'data' / 'catsdogsval.tar.gz'
    default_stats = Path(__file__).parent.parent / 'data' / 'train_stats.json'
    default_saved_model_dir = Path(__file__).parent.parent / 'model'
    default_model_dir = default_saved_model_dir
    default_checkpoint_path = Path(__file__).parent.parent / 'checkpoints'

class ArgParser:

    def __init__(self, args):
        """
        Parse the arguments.
        """
        examples = 'Examples:' + '\n\n'
        examples += sys.argv[0] + " --train trainimages.tar.gz" \
                                  " --eval valimages.tar.gz"
        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                              description='Run transfer learning on folder of images organized by label',
                                              epilog=examples)
        parser.add_argument('--train', type=str, help="Path to training compressed data", default=default_train.as_posix())
        parser.add_argument('--eval', type=str, default=default_eval.as_posix())
        parser.add_argument('--train_stats', type=str, default=default_stats)
        parser.add_argument('--model-dir', type=str, default=default_model_dir.as_posix())
        # location to store the saved model after training
        parser.add_argument('--saved-model-dir', type=str, default=default_saved_model_dir.as_posix())
        parser.add_argument('--checkpoint_local_path', type=str, default=default_checkpoint_path.as_posix(), help='Path to write checkpoints and wandb logs to')
        parser.add_argument("--base_model", choices=model_conf.MODEL_DICT.keys(),  help='Enter the network you want as your base feature extractor')
        parser.add_argument("--fine_tune_num", default=-1, type=int, help='Enter top n layers to unfreeze for fine-tuning')
        parser.add_argument("--batch_size", default=32, type=int,  help='Enter the batch size that must be used to train')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
        parser.add_argument('--l2_weight_decay_alpha', type=float, default=0.0,
                            help='weight decay if using l2 regularlization to reduce overfitting (default: 0.0 which disabled)')
        parser.add_argument('--has_wandb', type=self.boolean_string, default=False,
                                 help='true if logging to wandb server')
        parser.add_argument('--preprocessor', type=self.boolean_string, default=False,
                                 help='use the model preprocessor on the inputs')
        parser.add_argument('--featurewise_normalize', type=self.boolean_string, default=False,
                                 help='use featurewise centering and std normalizing')
        parser.add_argument('--dropout', type=self.boolean_string, default=False,
                                 help='add dropout layer')
        parser.add_argument('--horizontal_flip', type=self.boolean_string, default=False,
                                 help='add horizontal flip augmentation')
        parser.add_argument('--vertical_flip', type=self.boolean_string, default=False,
                                 help='add vertical flip augmentation')
        parser.add_argument('--early_stop', type=self.boolean_string, default=False,
                                 help='apply early stopping to model')
        parser.add_argument('--disable_save', type=self.boolean_string, default=False,
                                 help='disable model and checkpoint saving')
        parser.add_argument('--rotation_range', type=float, default=0.0, help='rotation range between 0-1 to apply rotation augmentation during training')
        parser.add_argument('--augment_range', type=float, default=0.0, help='range between 0-1 to apply width, shift, and zoom augmentation during training')
        parser.add_argument('--k', type=int, default=5, help='1-5 batch interval for look-ahead')
        parser.add_argument('--shear_range', type=float, default=0.0, help='range between 0-1 to apply shear augmentation during training')
        parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=1,
                                 type=int)
        parser.add_argument("--loss",
                                 help="Loss function for the gradients categorical_crossentropy, or categorical_focal_loss",
                                 nargs='?', action='store',
                                 default='categorical_crossentropy', type=str)
        parser.add_argument("--optimizer", help="optimizer: adam, radam, ranger", default='adam')
        parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)
        print('======================')
        print(vars(parser.parse_args(args)))
        self._args = parser.parse_args(args)

    @property
    def args(self):
        return self._args

    def boolean_string(self, s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
