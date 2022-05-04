#FROM python:3.7-slim
FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

SHELL ["/bin/bash", "-c"]

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

RUN apt-get update
RUN apt-get install -y python python3-pip python3-dev
RUN apt-get -y install git-core
RUN apt-get install -y liblzma-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install torch==1.6.0 tensorflow==2.4.0 transformers==4.1.1 \
python-crfsuite colorama coloredlogs requests flask flask_sqlalchemy flask_wtf \
python-telegram-bot pyconll pyyaml ufal.udpipe terminaltables

WORKDIR /home
ADD ruword2tags.tar.gz /home
WORKDIR /home/ruword2tags
RUN pip install .

RUN pip install git+https://github.com/Koziev/rutokenizer
RUN pip install git+https://github.com/Koziev/rupostagger
RUN pip install git+https://github.com/Koziev/ruword2tags
RUN pip install git+https://github.com/Koziev/rusyllab

RUN apt-get clean

WORKDIR /chatbot/ruchatbot/bot
COPY ./ruchatbot/bot/*.py ./

#WORKDIR /chatbot/ruchatbot/frontend
#COPY ./ruchatbot/frontend/*.py ./

#WORKDIR /chatbot/ruchatbot/bot_service
#COPY ./ruchatbot/bot_service/*.py ./

#WORKDIR /chatbot/ruchatbot/bot_service/static/img
#COPY ./ruchatbot/bot_service/static/img/*.* ./

#WORKDIR /chatbot/ruchatbot/bot_service/templates
#COPY ./ruchatbot/bot_service/templates/*.* ./

WORKDIR /chatbot/ruchatbot/utils
COPY ./ruchatbot/utils/*.py ./

WORKDIR /chatbot/data
COPY ./data/*.* ./

WORKDIR /chatbot/tmp/rugpt_chitchat
COPY ./tmp/rugpt_chitchat/*.* ./

WORKDIR /chatbot/tmp/ruBert-base
COPY ./tmp/ruBert-base/*.* ./

WORKDIR /chatbot/tmp
COPY ./tmp/*.* ./

WORKDIR /chatbot/scripts
#COPY ./scripts/console_bot.sh ./
#COPY ./scripts/flask_bot.sh ./
COPY ./scripts/tg_bot.sh ./

WORKDIR /chatbot
COPY CHANGELOG.txt ./

#EXPOSE 9001
WORKDIR /chatbot/scripts
CMD "./tg_bot.sh"
