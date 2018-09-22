# -*- coding: utf-8 -*-

"""
Привязка урлов к функциям сервиса чатбота https://github.com/Koziev/chatbot.
"""

from __future__ import print_function

import sys

from flask import request
from flask import render_template
from flask import flash
from flask import redirect
from flask import jsonify
from sqlalchemy import func
import json

from bot_service import flask_app
#from bot_service import db
from bot_service import rest_service_core
from bot_service.dialog_form import DialogForm
from dialog_phrase import DialogPhrase

#from db_mappings.mappings import ML_Solver, ML_Attr, ML_Sample

user_id = 'test_user'

@flask_app.route('/start', methods=["GET"])
def start():
    # Чтобы заранее заставить бота загрузить все модели с диска.
    return redirect('/index')
    pass


@flask_app.route('/', methods=["GET", "POST"])
@flask_app.route('/index', methods=["GET", "POST"])
def index():
    # покажем веб-форму, в которой пользователь сможет ввести
    # свою реплику и увидеть ответ бота
    form = DialogForm()
    bot = flask_app.config['bot']

    # todo - сохранять и восстанавливать историю диалога через БД...
    DIALOG_HISTORY = 'dialog_history'
    if DIALOG_HISTORY not in flask_app.config:
        flask_app.config[DIALOG_HISTORY] = dict()

    if user_id not in flask_app.config[DIALOG_HISTORY]:
        flask_app.config[DIALOG_HISTORY][user_id] = []

    phrases = flask_app.config[DIALOG_HISTORY][user_id][:]

    if form.validate_on_submit():
        # Пользователь ввел свою реплику, обрабатываем ее.
        #flash('full_name={}'.format(form.full_name.data))
        utterance = form.utterance.data
        if len(utterance) > 0:
            phrases.append(DialogPhrase(utterance, user_id, False))
            bot.push_phrase(user_id, utterance)

    if request.method == 'GET':
        bot.start_conversation(user_id)

    while True:
        answer = bot.pop_phrase(user_id)
        if len(answer) == 0:
            break
        phrases.append(DialogPhrase(answer, u'chatbot', True))

    flask_app.config[DIALOG_HISTORY][user_id] = phrases

    return render_template('dialog_form.html', hphrases=phrases, form=form)
