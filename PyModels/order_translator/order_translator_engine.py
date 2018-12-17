import operator

from bot.lgb_synonymy_detector import LGB_SynonymyDetector
from bot.order_comprehension_table import OrderComprehensionTable


class OrderTranslatorEngine(object):
    def __init__(self, text_utils):
        self.detector = LGB_SynonymyDetector()
        self.text_utils = text_utils

    def load_model(self, model_folder):
        self.detector.load(model_folder)

    def load_anchor_orders(self, filepath):
        self.orders_table = OrderComprehensionTable()
        self.orders_table.load_file(filepath)

    def list_anchors(self):
        orders = list(map(operator.itemgetter(0), self.orders_table.get_templates()))
        anchors = set(self.orders_table.get_order_anchor(o) for o in orders)
        return list(anchors)

    def find_nearest_order(self, order_str):
        orders = [(f[0],) for f in self.orders_table.get_templates()]
        nearest_order, similarity = self.detector.get_most_similar(self.text_utils.wordize_text(order_str),
                                                                   orders,
                                                                   self.text_utils,
                                                                   word_embeddings=None)
        return self.orders_table.get_order_anchor(nearest_order), similarity
