ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:3.1.0-torch_sdaa3.1.0
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/
ADD requirements.txt /workspace/
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh
RUN source activate torch27_env_py310 && pip install --no-cache-dir -r requirements.txt
ADD . /workspace/Classification/ResNet/
WORKDIR /workspace/Classification/ResNet/
