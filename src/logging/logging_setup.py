# logging_setup.py
import logging
import logging.config
from datetime import datetime

def time_namer(default_name):
    import os
    dirname, basename = os.path.split(default_name)
    name, ext, date = basename.split(".", 2)
    newname = f"{date}.{name}.{ext}"
    return os.path.join(dirname, newname)

def setup_logging():
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(asctime)s %(levelname)s: %(message)s"}
        },
        "handlers": {
            "timed": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "when": "midnight",
                "interval": 1,
                "backupCount": 7,
                "filename": "./log/app.log",
                "formatter": "simple",
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["timed"]
        },
    }
    
    logging.config.dictConfig(cfg)
    root = logging.getLogger()
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.TimedRotatingFileHandler):
            h.namer = time_namer
            h.doRollover()
