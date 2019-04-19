FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive


ADD ./* DSFD/

WORKDIR /DSFD

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "args.py"]
