FROM tensorflow/tensorflow:latest-gpu-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /usr/src/app /usr/src/app/datasets
COPY cullpdb+profile_6133.npy /usr/src/app/datasets

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app

RUN pip install -r requirements.txt

COPY embedding.py /usr/src/app

ENTRYPOINT [ "python", "embedding.py" ]