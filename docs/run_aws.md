# AWS Sagemaker On-Demand Training 

This simple example will train for 2 epochs in SageMaker. 

# Setup a minimal python environment

```
virtualenv --python=/usr/bin/python3.6 .venv
source .venv/bin/activate
pip install sagemaker==2.24.5

```
 
# Setup

Download the latest docker image mbari/kclassify

```

docker pull mbari/kclassify

```

Push that to your ECR repository

```
docker login -u AWS -p $(aws ecr get-login-password --region us-west-2) 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker push mbari/kclassify
```

# Run

!!! danger "Alert"
    Check for image before starting to make sure the image is available in your account, e.g. 
    
    $ aws ecr describe-repositories --query "repositories[].repositoryName" --output text --region us-west-2* 


In this example the image *kclassify:1.0.1* is used.

The example below stores the required training data and other artifacts that are generated during training in the following
organization:

```
~~~
│   └── your bucket/your prefix
│       └── training/ 
│                   └──train_stats.json (optional)
│                   └──train.tar.gz 
│                   └──val.tar.gz
│       └── checkpoint/  (optional)
...
│                   └──0f110283-1d0d-41ed-a336-b997bfec0658/
│                   └──1c264240-62b4-4342-9f31-80b6a5d69b14/
```


Available options for hyperparameters can be found [here](arguments.md)

```python 

import boto3
import botocore
import datetime as dt
import json
import uuid
import sagemaker
from sagemaker.estimator import Estimator
from pathlib import Path

#################################################################################################
# Setup default locations, model parameters and needed globals
#################################################################################################
# For s3 operations
s3_client = boto3.client('s3') 
s3_resource = boto3.resource('s3')
# client for sagemaker operations
sagemaker_session = sagemaker.Session() 
# This is set to the IAM role in SageMaker configured for the VAA project. If you are running this outside of a sagemaker notebook, you must set the role
role = 'arn:aws:iam::872338704006:role/service-role/AmazonSageMaker-ExecutionRole-20201012T164265'
# Region to run this in
region = 'us-west-2'
# Docker name in ECR of the training image
image_uri = '872338704006.dkr.ecr.us-west-2.amazonaws.com/kclassify-v1.0.1'
# The root path your training data is in locally
training_path = Path.cwd() / 'data' 
# The root bucket to store your training data and models in.
bucket = f'902005-videolab-test-sagemaker' 
# This can be anything you want - just a placeholder to separate this data from other training jobs in the bucket
prefix = 'test512x512' 
# Training data location
training_channel = prefix + '/training'
s3_train_data = f's3://{bucket}/{training_channel}'
# location to store checkpoints between jobs; optional - use if you want to run this again after a training job completes with the previous checkpoints
checkpoint_s3_bucket = f's3://{bucket}/{prefix}/checkpoint/{uuid.uuid4()}'
 
print(f'Model output {bucket_path}')
print(f'Training data in: {s3_train_data}')

#################################################################################################
# Configure tags. These are optional, but useful for later cost accounting and clean-up.
#################################################################################################
user = getpass.getuser() # this will grab the system user
deletion_date = (dt.datetime.utcnow() + dt.timedelta(days=90)).strftime('%Y%m%dT%H%M%SZ')
tag_dict = [{'Key': 'mbari:project-number', 'Value': '902005'},
        {'Key': 'mbari:owner', 'Value': user},
        {'Key': 'mbari:description', 'Value': 'test kclassify training'},
        {'Key': 'mbari:customer-project', 'Value': '902005'},
        {'Key': 'mbari:stage', 'Value': 'test'},
        {'Key': 'mbari:application', 'Value': 'detection'},
        {'Key': 'mbari:deletion-date', 'Value': deletion_date},
        {'Key': 'mbari:created-by', 'Value': user}]

#################################################################################################
# Create and tag the bucket to the bucket (only need to do this once)
#################################################################################################
try:
    response = s3_client.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': region },)
    print(response)
    # latest exceptions https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
except botocore.exceptions.ClientError as error:
    if 'BucketAlreadyOwnedByYou' not in str(error):
        raise error
try:
    bucket_tagging = s3_resource.BucketTagging(bucket)
    bucket_tagging.put(Tagging={'TagSet': tag_dict})
except Exception as error:
    raise error


#################################################################################################
# Upload the data  (only need to do this once, unless the data has changed)
#################################################################################################
print(f'Uploading data to {bucket}...')
for file in training_path.glob('*.*'): 
    sagemaker_session.upload_data(path=f'{training_path}/{file.name}', bucket=bucket, key_prefix=training_channel)
print('Done')

#################################################################################################
# Define the metrics to log
#################################################################################################
metric_definitions = [{'Name': 'validation_accuracy:', 'Regex': 'validation_accuracy = ([0-9.]+)'},
                      {'Name': 'validation_loss', 'Regex': 'validation_loss = ([0-9.]+)'},
                     {'Name': 'best_val_categorical_accuracy', 'Regex': 'best_val_categorical_accuracy = ([0-9.]+)'}]
print(metric_definitions) 

#################################################################################################
# Run training job
#################################################################################################
estimator = Estimator(base_job_name='bluewhale-a-efnetb0',
                       role=role,
                       tags=tag_dict,
                       image_uri=image_uri,
                       volume_size = 10,
                       enable_sagemaker_metrics=True,
                       instance_count=1,
                       instance_type='ml.p2.xlarge',
                       sagemaker_session=sagemaker_session,
                       input_mode= 'File',
                       metric_definitions=metric_definitions,
                       hyperparameters={
                           'epochs': 10,
                           'early_stop': True,
                           'horizontal_flip': True,
                           'vertical_flip': False,
                           'batch_size': 64,
                           'optimizer': 'adam',
                           'base_model': 'efficientnetB0',
                           'train_stats': '/opt/ml/input/data/training/train_stats.json',
                           'train': '/opt/ml/input/data/training/train.tar.gz',  # these must match the training files in the bucket an must be in /opt/ml/input/data/training as specified in the bucket prefix
                           'eval': '/opt/ml/input/data/training/val.tar.gz',
                           'saved-model-dir': '/opt/ml/model/1' # this must be in /opt/ml/model /opt/ml/model/1 puts in a version 1 of the model
                        })


train_data = sagemaker.inputs.TrainingInput(f'{s3_train_data}', distribution='FullyReplicated', content_type='text/plain', s3_data_type='S3Prefix')
data = {'training': train_data}
# Finally, run!
estimator.fit(inputs=data)
```
