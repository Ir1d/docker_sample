FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive


ADD ./ /DSFD/

WORKDIR /DSFD

# RUN ls >> /tmp/foo.txt

# RUN conda env create -n dsfd -f environment.yml && conda activate dsfd

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libfontconfig1 && apt-get clean \
  && rm -rf \
    /tmp/hsperfdata* \
    /var/*/apt/*/partial \
    /var/lib/apt/lists/* \
    /var/log/apt/term*

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "args.py"]
