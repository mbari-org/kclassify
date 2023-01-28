# ================================================================
#  Docker Image for  mbari/kclassify:x.x.x
#  ================================================================
ARG REGION
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/tensorflow-training:2.4.1-gpu-py37-cu110-ubuntu18.04

MAINTAINER Danelle Cline dcline@mbari.org

# install dependencies needed
RUN apt-get update && apt-get install -y git && python -m pip install --upgrade pip

RUN git clone https://github.com/mbari-org/kclassify /usr/src/kclassify
WORKDIR /usr/src/kclassify

# Install pip packages
COPY requirements.txt .
RUN pip install --no-cache tensorflow-gpu==2.6.1
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train.py"]
