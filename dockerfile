FROM nitincypher/docker-ubuntu-python-pip
#FROM ubuntu

RUN apt-get update
RUN apt-get install -y python python-pip 
RUN apt-get -y install git-core
#RUN apt-get install python-pip
RUN apt-get install -y liblzma-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install xgboost
RUN pip install lightgbm
RUN pip install keras
RUN pip install --upgrade scikit-learn
RUN pip install pymystem3
RUN pip install gensim
RUN pip install pathlib
RUN pip install python-crfsuite
RUN pip install tensorflow
RUN pip install colorama
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install git+https://github.com/Koziev/rutokenizer
RUN pip install git+https://github.com/Koziev/rupostagger
RUN pip install git+https://github.com/Koziev/ruword2tags
RUN pip install git+https://github.com/Koziev/rusyllab

RUN apt-get clean 

WORKDIR /chatbot/PyModels/bot
COPY ./PyModels/bot/*.py ./

WORKDIR /chatbot/PyModels/bot_service
COPY ./PyModels/bot_service/*.py ./

WORKDIR /chatbot/PyModels/order_translator
COPY ./PyModels/order_translator/*.py ./

WORKDIR /chatbot/PyModels/trainers
COPY ./PyModels/trainers/*.py ./

WORKDIR /chatbot/PyModels/utils
COPY ./PyModels/utils/*.py ./

WORKDIR /chatbot/PyModels/generative_grammar
COPY ./PyModels/generative_grammar/*.py ./

WORKDIR /chatbot/PyModels
COPY ./PyModels/console_chatbot.py ./

WORKDIR /chatbot/data
COPY ./data/*.* ./

WORKDIR /chatbot/tmp
COPY ./tmp/*.* ./

WORKDIR /chatbot/scripts
COPY ./scripts/console_bot.sh ./

CMD "/chatbot/scripts/console_bot.sh"
