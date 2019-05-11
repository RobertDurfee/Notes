FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    texlive-xetex \
    pandoc

RUN pip3 install \
    pylint \
    jupyter \
    numpy \
    scipy \
    matplotlib \
    pandas \
    seaborn \
    statsmodels \
    scikit-learn \
    tensorflow

RUN useradd -ms /bin/bash u6s077
USER u6s077

WORKDIR /home/u6s077/work