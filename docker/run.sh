#!/bin/bash
DOCKER_CONTAINER_NAME="rl-algorithms"

if [ "$( docker container inspect -f '{{.State.Running}}' $DOCKER_CONTAINER_NAME )" == "false" ]; then
    echo Start $DOCKER_CONTAINER_NAME ...
    xhost +local:root
    docker start $DOCKER_CONTAINER_NAME
fi

docker exec -ti $DOCKER_CONTAINER_NAME bash