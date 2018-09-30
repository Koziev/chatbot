# -*- coding: utf-8 -*-

import logging
import logging.handlers

def init_trainer_logging(logfile_path):
    # настраиваем логирование в файл и эхо-печать в консоль
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(logfile_path, mode='w')

    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)
