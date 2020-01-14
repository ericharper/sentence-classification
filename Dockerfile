FROM nvcr.io/nvidia/nemo:v0.9

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsndfile1 sox \
    python-setuptools python-dev && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/src/tensorrt/bin

# install additional jupyter lab extensions
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

# RUN jupyter labextension install jupyterlab_vim

RUN pip install jupyterlab-nvdashboard

RUN jupyter labextension install jupyterlab-nvdashboard

# RUN pip install flake8 && jupyter labextension install jupyterlab-flake8

RUN jupyter lab build

RUN mkdir /sentence-classification

COPY . /sentence-classification

WORKDIR /sentence-classification
