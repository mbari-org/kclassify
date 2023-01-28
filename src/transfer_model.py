#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Surgery code to freeze, slim by dropping layers, etc.

@author: __author__
@status: __status__
@license: __license__
'''

import tensorflow as tf
import conf as model_conf


class TransferModel:

    def __init__(self, base_model_name:str="vgg16",  dropout:bool=False, l2_weight_decay_alpha=0.):
        """
        :param base_model_name: base model name as defined in conf.py
        :param dropout:  if true, add dropout layer
        :param l2_weight_decay_alpha: if >0, adds loss decay layers
        """
        self._base_model_name = base_model_name
        self._dropout = dropout
        self._l2_weight_decay_alpha=l2_weight_decay_alpha
        if base_model_name not in model_conf.MODEL_DICT.keys():
            raise (f'{base_model_name} not in {model_conf.MODEL_DICT.keys()}')

    def build(self, class_size:int):
        """
        Build base model from the pre-trained model
        :param class_size: number of classes in Dense layer
        :return: a Keras network model
        """
        cfg = eval(f"model_conf.MODEL_DICT['{self._base_model_name}']")
        image_size = cfg['image_size']
        img_shape = (image_size, image_size, 3)
        model_instance = cfg['model_instance']

        base_model = eval(f"{model_instance}(input_shape={img_shape},include_top=False,weights='imagenet')")

        if self._l2_weight_decay_alpha > 0.:
            if cfg['has_depthwise_layers']:
                for layer in base_model.layers:
                    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                        layer.add_loss(tf.keras.regularizers.l2(self._l2_weight_decay_alpha)(l.depthwise_kernel))
                    elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                        layer.add_loss(tf.keras.regularizers.l2(self._l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(tf.keras.regularizers.l2(self._l2_weight_decay_alpha)(layer.bias))
            else:
                for layer in base_model.layers:
                    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                        layer.add_loss(tf.keras.regularizers.l2(self._l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(tf.keras.regularizers.l2(self._l2_weight_decay_alpha)(layer.bias))

        base_model.trainable = False

        if self._dropout:
            print('add dropout)')
            model_sequential = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(class_size, activation='softmax')
            ])
        else:
            model_sequential = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(class_size, activation='softmax')
            ])

        return model_sequential, image_size


if __name__ == '__main__':
    mmaker = TransferModel()
    # build the basic model
    model = mmaker.build()
    model.summary()
