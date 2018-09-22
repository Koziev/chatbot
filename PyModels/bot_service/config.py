import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ChatBot-service-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, '../../tmp/kb.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
