from flask import render_template
import connexion
import logging

from utils.logging_helpers import init_trainer_logging
from bot.text_utils import TextUtils
from global_vars import set_engine
from order_translator_engine import OrderTranslatorEngine


init_trainer_logging('../../tmp/order_translator_srv.log')  # TODO: set folder for logging by command line

# app = Flask(__name__, template_folder="templates")

# To enable swagger UI console:
# pip install connexion[swagger-ui]

options = {"swagger_ui": True}
app = connexion.App(__name__, specification_dir='./', options=options)
app.add_api('swagger.yml')

logging.info('Initialize parser')

text_utils = TextUtils()
# text_utils.load_dictionaries('../')

engine = OrderTranslatorEngine(text_utils)
engine.load_model('../../tmp')
engine.load_anchor_orders('../../data/test_orders.txt')
set_engine(engine)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
