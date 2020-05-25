FROM python:3.6-slim

SHELL ["/bin/bash", "-c"]

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8


RUN apt-get update
RUN apt-get install -y python python-pip
RUN apt-get -y install git-core
RUN apt-get install -y liblzma-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install git+https://github.com/Koziev/rulemma
RUN pip uninstall -y numpy
RUN yes | apt-get install python-numpy

RUN pip install sentencepiece
RUN pip install lightgbm
RUN pip install keras==2.2.4
RUN pip install --upgrade scikit-learn
RUN pip install gensim
RUN pip install pathlib
RUN pip install python-crfsuite
RUN pip install tensorflow==1.14
RUN pip install colorama
RUN pip install coloredlogs
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git

WORKDIR /home
ADD ruword2tags.tar.gz /home
WORKDIR /home/ruword2tags
RUN pip install .

RUN pip install git+https://github.com/Koziev/rutokenizer
RUN pip install git+https://github.com/Koziev/rupostagger
#RUN pip install git+https://github.com/Koziev/ruword2tags
RUN pip install git+https://github.com/Koziev/rusyllab
RUN pip install git+https://github.com/Koziev/ruchunker

RUN apt-get clean

WORKDIR /chatbot/ruchatbot/bot
COPY ./ruchatbot/bot/*.py ./

WORKDIR /chatbot/ruchatbot/frontend
COPY ./ruchatbot/frontend/*.py ./

WORKDIR /chatbot/ruchatbot/bot_service
COPY ./ruchatbot/bot_service/*.py ./

WORKDIR /chatbot/ruchatbot/utils
COPY ./ruchatbot/utils/*.py ./

WORKDIR /chatbot/ruchatbot/layers
COPY ./ruchatbot/layers/*.py ./

WORKDIR /chatbot/ruchatbot/scenarios
COPY ./ruchatbot/scenarios/*.py ./

WORKDIR /chatbot/ruchatbot/generative_grammar
COPY ./ruchatbot/generative_grammar/*.py ./

WORKDIR /chatbot/ruchatbot
COPY ./ruchatbot/__init__.py ./
COPY ./ruchatbot/qa_machine.py ./

WORKDIR /chatbot/data
COPY ./data/*.* ./

WORKDIR /chatbot/tmp
COPY ./tmp/*.* ./

WORKDIR /chatbot/scripts
COPY ./scripts/console_bot.sh ./
COPY ./scripts/test_console.sh ./

WORKDIR /chatbot/scripts
CMD "./console_bot.sh"
#CMD "./test_console.sh"
