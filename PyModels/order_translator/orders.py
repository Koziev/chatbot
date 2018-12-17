# -*- coding: utf-8 -*-

import traceback
from global_vars import get_engine


def read_all():
    return get_engine().list_anchors()


def translate_order(raw):
    try:
        anchor, similarity = get_engine().find_nearest_order(raw)
        return {'ok': True,
                'error_str': u'',
                'anchor_order': anchor,
                'similarity': similarity}
    except Exception as e:
        err_msg = str(e) + u' ' + traceback.format_exc()
        # flask.abort(500, err_msg)
        return {'ok': False,
                'error_str': err_msg,
                'anchor_order': u'',
                'similarity': 0.0}
