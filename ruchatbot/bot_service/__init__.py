from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from .config import Config
from .global_params import profile_path, models_folder, data_folder, w2v_folder

from .rest_service_core import flask_app

#flask_app = Flask(__name__)

#flask_app.config['SECRET_KEY'] = 'ChatBot-service-key'
#flask_app.config.from_object(Config)

#db = SQLAlchemy(flask_app)

import ruchatbot.bot_service.routes
