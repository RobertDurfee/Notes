FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install \
    jupyter \
    numpy \
    tensorflow==2.0.0-alpha0 \
    matplotlib

RUN useradd -ms /bin/bash u6036
USER u6036

WORKDIR /home/u6036/work