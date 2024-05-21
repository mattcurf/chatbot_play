#!/bin/bash

docker run -it --rm -v `pwd`:/project -w /project --device /dev/dri:/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY chatbot-ipex /bin/bash _run.sh 

