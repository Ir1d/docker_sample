FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive


ADD ./ /keras-retinanet/

WORKDIR /keras-retinanet

RUN pip install -r requirements.txt && cd keras-retinanet

ENTRYPOINT ["python", "demo.py"]
