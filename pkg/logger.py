import logging
import os
from datetime import datetime

def get_logger(profile_name: str = "default"):
    logs_dir = os.path.join("logs", profile_name)
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(
        logs_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logger = logging.getLogger(profile_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
