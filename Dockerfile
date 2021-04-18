# FROM python:3.8
FROM ubuntu:groovy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python3-opengl \
       python3-pyglet \
       python3-pip

RUN mkdir /ml_ws
WORKDIR /ml_ws
       
COPY requirements.txt /tmp/

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# RUN pip3 install –trusted-host pypi.python.org -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt