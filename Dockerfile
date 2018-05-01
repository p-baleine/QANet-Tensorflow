FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt-get update -y \
  && apt-get install language-pack-ja -y \
  && locale-gen ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:en
ENV LC_ALL ja_JP.UTF-8

COPY requirements.txt /tmp/
RUN grep -v tensorflow /tmp/requirements.txt | pip install -r /dev/stdin

WORKDIR /qanet

CMD ["/run_jupyter.sh", "--allow-root"]
