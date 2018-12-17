order_translator_engine = None


def set_engine(engine):
    global order_translator_engine
    order_translator_engine = engine


def get_engine():
    return order_translator_engine
