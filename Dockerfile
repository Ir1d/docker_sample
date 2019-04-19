FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

LABEL maintainer="Ir1d <sirius.caffrey@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get --no-install-recommends --yes install \
    autoconf \
    automake \
    build-essential \
    curl \
    default-jre-headless \
    epstool \
    g++ \
    gcc \
    gdb \
    gfortran \
    ghostscript \
    gnuplot-qt \
    info \
    less \
    libamd2.4.1 \
    libarpack2 \
    libblas-dev \
    libcamd2.4.1 \
    libccolamd2.9.1 \
    libcholmod3.0.6 \
    libcolamd2.9.1 \
    libcurl3-gnutls \
    libcxsparse3.1.4 \
    libfftw3-dev \
    libfltk-gl1.3 \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-glx \
    libgl2ps0 \
    libglpk36 \
    libglu1-mesa \
    libgraphicsmagick++3 \
    libhdf5-10 \
    libhdf5-dev \
    libhdf5-serial-dev \
    liblapack-dev \
    libncurses5 \
    libopenblas-dev \
    libpcre3 \
    libportaudio2 \
    libqhull7 \
    libqrupdate1 \
    libqscintilla2-12v5 \
    libqt4-network \
    libqt4-opengl \
    libreadline-dev \
    libsndfile1 \
    libsundials-ida2 \
    libsundials-nvecserial0 \
    libtool \
    libumfpack5.7.1 \
    make \
    pstoedit \
    texinfo \
    transfig \
    unzip \
    xz-utils \
    zip \
  && apt-get clean \
  && rm -rf \
    /tmp/hsperfdata* \
    /var/*/apt/*/partial \
    /var/lib/apt/lists/* \
    /var/log/apt/term*

ADD https://s3.amazonaws.com/octave-snapshot/public/SHA256SUMS /
RUN curl -SL https://s3.amazonaws.com/octave-snapshot/public/octave-ubuntu-trusty-snapshot.tar.xz \
  | tar -xJC /usr/local --strip-components=1

# COPY --from=mtmiller/octave:latest /octave/ /octave/

ADD ./eval_tools /tools
VOLUME [ "/tools/data", "/tools/output", "/root/UG2/Sub_challenge2_1/output/0001/output/" ]

WORKDIR /tools

ENTRYPOINT ["octave", "df_eval.m"]
