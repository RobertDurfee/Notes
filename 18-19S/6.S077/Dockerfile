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
    pandas \
    seaborn \
    statsmodels \
    scikit-learn

RUN useradd -ms /bin/bash jovyan
USER jovyan

WORKDIR /home/jovyan/work

ENTRYPOINT [ "jupyter", "notebook", "--ip=0.0.0.0" ]