from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from config import Config
from bot_service.global_params import facts_folder, models_folder,\
    data_folder, w2v_folder


flask_app = Flask(__name__)

#flask_app.config['SECRET_KEY'] = 'ChatBot-service-key'
flask_app.config.from_object(Config)

#db = SQLAlchemy(flask_app)

import routes
