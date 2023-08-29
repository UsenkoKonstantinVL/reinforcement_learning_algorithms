FROM nvidia/cudagl:11.0.3-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" | tee /etc/apt/sources.list.d/cuda.list
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    apt-utils \
    mesa-utils \
    gnupg2 \
    net-tools \
    wget \
    unzip \
    curl \
    git \
    mc \
    vim

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip

RUN python3 --version

# RUN python3 -m pip install numpy 
RUN python3 -m pip install numpy gym rllib torch
