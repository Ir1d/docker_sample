FROM ufoym/deepo:pytorch-py36-cu90
# FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive

ADD ./ /

WORKDIR /

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libfontconfig1 && apt-get clean \
  && rm -rf \
    /tmp/hsperfdata* \
    /var/*/apt/*/partial \
    /var/lib/apt/lists/* \
    /var/log/apt/term*

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt

ENTRYPOINT ["python", "args.py"]
