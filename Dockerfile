FROM continuumio/miniconda3:latest

VOLUME /usr/src/ucsgnet

WORKDIR /root
COPY env.yml /root
RUN conda env create -f env.yml
RUN conda init bash
ENV PATH /opt/conda/envs/ucsg/bin:$PATH

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    libgl1-mesa-glx

WORKDIR /usr/src/ucsgnet
RUN echo "conda activate ucsg" >> ~/.bashrc
CMD bash
