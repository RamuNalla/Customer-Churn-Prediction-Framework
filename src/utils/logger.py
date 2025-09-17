import logging
import logging.config
from pathlib import Path
import yaml
from datetime import datetime

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:   # setup logger configuration

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)    

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': f'logs/{name}_{datetime.now().strftime("%Y%m%d")}.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            name: {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file']
        }
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger(name)