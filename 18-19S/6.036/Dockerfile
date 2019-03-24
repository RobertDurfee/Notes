FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install \
    jupyter \
    numpy \
    tensorflow \
    matplotlib

RUN useradd -ms /bin/bash _6036
USER _6036

WORKDIR /home/_6036/work