ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev 
RUN pip install opencv-python scipy tensorboardX tensorboard scikit-image pyyaml
