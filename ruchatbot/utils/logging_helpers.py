# -*- coding: utf-8 -*-

import logging
import logging.handlers
import coloredlogs


def init_trainer_logging(logfile_path, debugging=True):
    # настраиваем логирование в файл и эхо-печать в консоль
    log_level = logging.DEBUG if debugging else logging.ERROR
    logging.basicConfig(level=log_level, format='%(asctime)s %(message)s')

    lf = logging.FileHandler(logfile_path, mode='w')
    lf.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=log_level,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )

