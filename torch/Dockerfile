FROM pytorch/torchserve:latest

USER root

RUN apt-get update && \
    apt-get install -y \
        curl \
    && pip --no-cache-dir install --upgrade awscli \
    && apt-get clean

RUN mkdir -p /home/model-server/model-store
RUN chown -R model-server:model-server /home/model-server

# USER model-server

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
