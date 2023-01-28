#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Configuration file for model specifications. 
These models are the ones supported in this code.
More details on the models can be found at https://keras.io/api/applications/

@author: __author__
@status: __status__
@license: __license__
'''

import tensorflow as tf

MODEL_DICT = {}

efficientnetB0 = dict(
    image_size=224,
    model_instance="tf.keras.applications.EfficientNetB0",
    preprocessor="tf.keras.applications.efficientnet.preprocess_input",
    has_depthwise_layers=False
)
resnet50 = dict(
    image_size=224,
    model_instance="tf.keras.applications.ResNet50",
    preprocessor="tf.keras.applications.resnet.preprocess_input",
    has_depthwise_layers=False
)
vgg16 = dict(
    image_size=224,
    model_instance="tf.keras.applications.VGG16",
    preprocessor="tf.keras.applications.vgg16.preprocess_input",
    has_depthwise_layers=False
)
vgg19 = dict(
    image_size=224,
    model_instance="tf.keras.applications.VGG19",
    preprocessor="tf.keras.applications.vgg19.preprocess_input",
    has_depthwise_layers=False
)
mobilenetv2 = dict(
    image_size=224,
    model_instance="tf.keras.applications.MobileNetV2",
    preprocess="tf.keras.applications.mobilenet_v2.preprocess_input",
    model_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2",
    feature_extractor_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2",
    # fine_tune_at=100,
    has_depthwise_layers=True
)

MODEL_DICT["efficientnetB0"] = efficientnetB0
MODEL_DICT["resnet50"] = resnet50
MODEL_DICT["mobilenetv2"] = mobilenetv2
MODEL_DICT["vgg16"] = vgg16
MODEL_DICT["vgg19"] = vgg19
