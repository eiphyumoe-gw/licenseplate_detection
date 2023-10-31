FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y \
    python3-pip ffmpeg libsm6 libxext6 -y


RUN pip install --upgrade pip 

WORKDIR /Test-docker/

RUN pip3 install Cython
RUN pip3 install torch torchvision

#install dependencies
RUN apt-get update && apt-get install -y \
  zlib1g-dev \
  libjpeg-dev \
  gcc \
  libtiff5-dev
  
COPY src/ src/
COPY test/ test/
COPY weights/ weights/
COPY requirements.txt ./

RUN pip3 install -r requirements.txt

# RUN pip3 install -v -e src/YOLOX/.