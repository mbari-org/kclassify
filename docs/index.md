---
description: Train an image classifier either locally or in AWS. Supports Resnet50, EfficientNetB0, VGG16, VGG19 or MobileNetV2 models
---

Author: Danelle Cline, [dcline@mbari.org](mailto:dcline@mbari.org)

# About

*kclassify* runs an image classification model using transfer-learning.  This runs on one or many GPUS either locally or in AWS.

<a href="https://aws.amazon.com/what-is-cloud-computing"><img src="https://d0.awsstatic.com/logos/powered-by-aws.png" alt="Powered by AWS Cloud Computing"></a>

Currently, supports the following:

Optimizers

* Radam
* Adam
* Ranger (not working as of 7-20-21)

Models

* efficientnetB0
* resnet50
* vgg16
* vgg19
* mobilenetv2

Augmentations

* width, shift, and zoom
* horizontal/vertical flip

and all the typical hyperparameters needed for model training like learning rate,  batch size, etc.

## Requirements
- Docker
- SageMaker SDK version 2.20.0
- One or more GPUs
- Training and validation images (JPEG or PNG) images compressed into tar.gz files. See [Data organization](data.md) for details 
- AWS Account (only needed if modifying the model)
    - Your AWS account must support the role *ecr:InitiateLayerUpload* to push the docker image this creates 

## [Data organization](data.md)
## [How to run in AWS](run_aws.md)
## [License](http://www.gnu.org/licenses/gpl-3.0.en.html)
