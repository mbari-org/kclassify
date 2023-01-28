[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)

[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
![Supported Platforms](https://img.shields.io/badge/Supported%20Platforms-Windows%20%7C%20macOS%20%7C%20Linux-green)
![license-GPL](https://img.shields.io/badge/license-GPL-blue)

Author: Danelle Cline, [dcline@mbari.org](mailto:dcline@mbari.org)

# About

* kclassify * Tensorflow-Keras image classifier for transfer-learning training either locally or in AWS SageMaker

<a href="https://aws.amazon.com/what-is-cloud-computing"><img src="https://d0.awsstatic.com/logos/powered-by-aws.png" alt="Powered by AWS Cloud Computing"></a>


This trains an image classifier using transfer learning with the Tensorflow Keras library with choices of

Optimizers

* Radam
* Adam
* Ranger (not working as of 7-20-21)

Models

* efficientnetB0 EfficientNetB0
* resnet50
* vgg16
* vgg19
* mobilenetv2
 
Augmentations

* width, shift, and zoom  
* horizontal/vertical flip

and all the typical hyperparameters needed for model training like 
learning rate,  batch size, etc.
 
Documentation can be found [here](http://docs.mbari.org/kclassify).

## Requirements
 - Docker
 - AWS Account 
   
# Questions?

If you have any questions or are interested in contributing, please contact me at dcline@mbari.org.

*Danelle Cline*
https://www.mbari.org/cline-danelle-e/

---
