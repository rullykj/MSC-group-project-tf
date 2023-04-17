FROM ubuntu:focal

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN ln -sf /usr/share/zoneinfo/Europe/London /etc/localtime

EXPOSE 8000
WORKDIR /api

VOLUME ["/api"]

COPY req/. /api

RUN apt-get update && \
    apt-get install -yq tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get autoremove -y && \
    apt-get install build-essential cmake -y && \
	apt-get install libopenblas-dev liblapack-dev -y && \
	apt-get install libx11-dev libgtk-3-dev -y && \
	apt-get install python3-pip -y
	
RUN pip3 install -U pip

RUN pip3 install -r requirements.txt

COPY src/. /api

CMD ["sanic", "main.app", "--host=0.0.0.0", "--port=8000", "--reload"]