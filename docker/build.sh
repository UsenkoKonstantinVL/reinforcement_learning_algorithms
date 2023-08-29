#!/bin/bash
DOCKER_CONTAINER_NAME="rl-algorithms"
DEV_DOCKER_IMAGE_NAME="rl-algorithms"

echo "## BUILD CONTAINER '$DOCKER_CONTAINER_NAME'"
docker build -t $DOCKER_CONTAINER_NAME .

PROJECT_VOLUME="$(pwd)/../:/rl_ws"

echo "## RUN CONTAINER '$DOCKER_CONTAINER_NAME'"
docker run \
	-d \
	-it \
	--name $DOCKER_CONTAINER_NAME \
	--gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-v $PROJECT_VOLUME \
    $DEV_DOCKER_IMAGE_NAME bash
