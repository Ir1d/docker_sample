FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive


ADD ./ /keras-retinanet/

WORKDIR /keras-retinanet

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libfontconfig1 && pip install --no-cache-dir -r requirements.txt  && apt-get clean \
  && rm -rf \
    /tmp/hsperfdata* \
    /var/*/apt/*/partial \
    /var/lib/apt/lists/* \
    /var/log/apt/term*
RUN pip install --no-cache-dir pycocotools==2.0 && cd keras-retinanet

ENTRYPOINT ["python", "demo.py"]
