# -*- coding: utf-8 -*-

from __future__ import print_function

import logging

from flask import Flask
from flask import request
from flask import jsonify

from sqlalchemy.sql import text

from .config import Config


flask_app = Flask(__name__)

#flask_app.config['SECRET_KEY'] = 'ChatBot-service-key'
flask_app.config.from_object(Config)

# TODO ...
