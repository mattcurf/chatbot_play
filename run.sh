#!/bin/bash

docker run -it --rm -v `pwd`:/project -w /home/user --device /dev/dri:/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY chatbot_ubuntu.jammy_$USER python3 /project/chatbot_demo.py

