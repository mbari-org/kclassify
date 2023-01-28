#!/bin/bash

algorithm_name=mbari/kclassify:1.0.1
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)

# Build the docker image locally
docker build -t ${algorithm_name} -f docker/Dockerfile .
docker tag ${algorithm_name} ${algorithm_name}

# Run a quick test of training 3 epochs; if this works then the cloud training will also work
echo --augment_range 0.2 --optimizer adam --base_model efficientnetB0 --epoch 3 --early_stop True --preprocessor False --disable_save True --featurewise_normalize True --lr .001  --train /opt/ml/input/data/training/catsdogstrain.tar.gz --eval /opt/ml/input/data/training/catsdogsval.tar.gz --train_stats /opt/ml/input/data/training/train_stats.json --has_wandb False
docker run -v $PWD/data:/opt/ml/input/data/training -it --rm  ${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name} --augment_range 0.2 --optimizer adam --base_model efficientnetB0 --epoch 3 --early_stop True --preprocessor False --disable_save True --featurewise_normalize True --lr .001  --train /opt/ml/input/data/training/catsdogstrain.tar.gz --eval /opt/ml/input/data/training/catsdogsval.tar.gz --train_stats /opt/ml/input/data/training/train_stats.json

# With wandb key
#docker run -v $PWD/data:/opt/ml/input/data/training -e WANDB_API_KEY=......yourkey  -it --rm  ${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name} --augment_range 0.2 --optimizer adam --base_model efficientnetB0 --epoch 3 --early_stop True --preprocessor False --disable_save True --featurewise_normalize True --lr .001  --train /opt/ml/input/data/training/catsdogstrain.tar.gz --eval /opt/ml/input/data/training/catsdogsval.tar.gz --train_stats /opt/ml/input/data/training/train_stats.json --has_wandb False

