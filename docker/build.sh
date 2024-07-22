#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./build.sh IMAGE_NAME"
  return 1
fi

# the Dockerfile can only copy files from the same folder it is located at
cp ../requirements.txt ./requirements.txt

# Build the docker image
docker build\
  -t $1 \
  -f Dockerfile . \
  # --build-arg user=$USER\
  # --build-arg uid=$UID