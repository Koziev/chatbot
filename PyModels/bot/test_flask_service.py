# -*- coding: utf-8 -*-
'''
Вопрос-ответная машина реализована в виде REST сервиса на Flask.

https://flask-restful.readthedocs.io/en/latest/index.html
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import logging

from flask import Flask
from flask_restful import Resource, Api

from files3_facts_storage import Files3FactsStorage
from text_utils import TextUtils
from simple_answering_machine import SimpleAnsweringMachine


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

app = Flask(__name__)
api = Api(app)

# https://stackoverflow.com/questions/24251307/flask-creating-objects-that-remain-over-multiple-requests
@app.before_first_request
def load_answering_machine():
    global answering_machine
    logging.info('Loading answering machine models...')
    text_utils = TextUtils()
    facts_storage = Files3FactsStorage(text_utils=text_utils, facts_folder='/home/eek/polygon/paraphrasing/data')
    answering_machine = SimpleAnsweringMachine(facts_storage=facts_storage, text_utils=text_utils)
    answering_machine.load_models('/home/eek/polygon/paraphrasing/tmp')

class AnsweringMachineStartResource(Resource):
    def get(self):
        return {"started": 1}


class AnsweringMachineQueryResource(Resource):
    def get(self, sender_id, question):
        logging.info(u'Start processing query {} from sender={}'.format(question, sender_id))
        total_answer = u''
        answering_machine.push_phrase(sender_id, question+u'?')
        while True:
            answer = answering_machine.pop_phrase(sender_id)
            if len(answer)==0:
                break
            else:
                if len(total_answer)>0:
                    total_answer += u'\n'
                total_answer += answer;

        return {'answer': total_answer}

# Предзагрузка словарей и моделей по урлу http://127.0.0.1:5000/start
api.add_resource(AnsweringMachineStartResource, '/start')

# Пример обработки вопроса: http://127.0.0.1:5000/answer/1/%D0%BA%D0%B0%D0%BA%20%D1%82%D0%B5%D0%B1%D1%8F%20%D0%B7%D0%BE%D0%B2%D1%83%D1%82
api.add_resource(AnsweringMachineQueryResource, '/answer/<sender_id>/<question>')

if __name__ == '__main__':
    app.run(debug=True)
