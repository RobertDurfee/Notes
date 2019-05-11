FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    texlive-xetex \
    pandoc

RUN pip3 install \
    jupyter \
    numpy \
    scipy \
    matplotlib \
    xlrd \
    pandas \
    seaborn \
    statsmodels \
    linearmodels \
    scikit-learn

RUN useradd -ms /bin/bash u1432
USER u1432

WORKDIR /home/u1432/work