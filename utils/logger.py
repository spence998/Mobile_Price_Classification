import logging
import sys
from datetime import date


def get_configured_logger(name):
    today_date = date.today()
    severity = "INFO"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Saved log formatting
    logging.basicConfig(
        filename=f"log_{today_date}.log",
        format=format,
        level=severity
    )

    #
    logger = logging.getLogger(name)
    logger.setLevel(severity)
    formatter = logging.Formatter(format)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(severity)
    logger.addHandler(ch)
    return logger


# import logging
# from logger import get_configured_logger(__name__)
logger = get_configured_logger("Spencer")

logger.info("hello")
