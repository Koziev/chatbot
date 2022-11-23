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

RUN pip install torch==1.8.0 tensorflow==2.4.0 transformers==4.11.3 \
python-crfsuite colorama coloredlogs requests flask flask_sqlalchemy flask_wtf \
python-telegram-bot pyconll pyyaml ufal.udpipe terminaltables networkx ufal.udpipe sentence_transformers

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

WORKDIR /chatbot/ruchatbot/scripting
COPY ./ruchatbot/scripting/*.py ./

WORKDIR /chatbot/ruchatbot/scripting/generator
COPY ./ruchatbot/scripting/generator/*.py ./

WORKDIR /chatbot/ruchatbot/scripting/matcher
COPY ./ruchatbot/scripting/matcher/*.py ./

WORKDIR /chatbot/ruchatbot/frontend
COPY ./ruchatbot/frontend/*.py ./

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

WORKDIR /chatbot/tmp/sbert_pq
COPY ./tmp/sbert_pq .

WORKDIR /chatbot/tmp/sbert_synonymy
COPY ./tmp/sbert_synonymy .

WORKDIR /chatbot/tmp/rugpt_npqa
COPY ./tmp/rugpt_npqa/*.* ./

#WORKDIR /chatbot/tmp/rugpt_interpreter
#COPY ./tmp/rugpt_interpreter/*.* ./

WORKDIR /chatbot/tmp/t5_interpreter
COPY ./tmp/t5_interpreter/*.* ./

WORKDIR /chatbot/tmp/rugpt_premise4question
COPY ./tmp/rugpt_premise4question/*.* ./

WORKDIR /chatbot/tmp/rubert-tiny
COPY ./tmp/rubert-tiny/*.* ./

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
